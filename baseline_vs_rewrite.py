"""
Synthetic dataset generation pipeline + scalar reward model training
for RLHF, comparing a rewrite-augmented model against a baseline.

Requirements:
    pip install google-generativeai datasets transformers torch tqdm \
                pandas scikit-learn accelerate groq
"""

import os
import json
import shutil

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from synthetic_data_gen import PipelineConfig, SyntheticDataPipeline, print_dataset_stats
from reward_model import (
    RewardModelConfig,
    RewardModelTrainer,
    ScalarRewardModel,
    RLHFPairDataset,
)

def check_environment():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT found"
    print("GPU:", gpu)

    for fname in ["synthetic_data_gen.py", "reward_model.py"]:
        status = "✓" if os.path.exists(fname) else "✗ MISSING:"
        print(status, fname)

# 1. Generate synthetic dataset (test run — 10 samples)

def generate_test_dataset(api_key: str, output_dir: str = "./rlhf_data_test"):
    config = PipelineConfig(
        groq_api_key=api_key,
        num_samples=10,
        output_dir=output_dir,
    )
    pipeline = SyntheticDataPipeline(config)
    pipeline.run()

    with open(f"{output_dir}/rlhf_dataset.json") as f:
        data = json.load(f)

    sample = data[0]
    print("PROMPT   :", sample["prompt"])
    print("\nCHOSEN   :", sample["response_chosen"][:100])
    print("\nREJECTED :", sample["response_rejected"][:100])
    print("\nFEEDBACK :", sample["rewrite_feedback"])

# 2. Generate full dataset (100 samples)

def generate_full_dataset(api_key: str, output_dir: str = "./rlhf_data"):
    config = PipelineConfig(
        groq_api_key=api_key,
        num_samples=100,
        output_dir=output_dir,
        delay_between_calls=1.0,
    )
    pipeline = SyntheticDataPipeline(config)
    pipeline.run()
    print_dataset_stats(output_dir)

# 3. Inspect dataset

def inspect_dataset(data_dir: str = "./rlhf_data"):
    df = pd.read_csv(f"{data_dir}/rlhf_dataset.csv")
    cols = ["prompt", "rewrite_feedback", "quality_score_chosen", "quality_score_rejected"]
    print(df[cols].head(3).to_string())

# 4. Baseline reward model (no rewrite augmentation)

class BaselineRLHFPairDataset(RLHFPairDataset):
    def __init__(self, samples, tokenizer, max_length):
        super().__init__(samples, tokenizer, max_length, augment_with_rewrite=False)

class BaselineRewardModelTrainer(RewardModelTrainer):
    def _load_data(self):
        with open(self.config.data_path) as f:
            samples = json.load(f)

        split = int(0.9 * len(samples))
        train_samples = samples[:split]
        val_samples = samples[split:]
        print(f"[BASELINE] Train: {len(train_samples)} | Val: {len(val_samples)}")

        train_dataset = BaselineRLHFPairDataset(
            train_samples, self.tokenizer, self.config.max_length
        )
        val_dataset = BaselineRLHFPairDataset(
            val_samples, self.tokenizer, self.config.max_length
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2
        )
        return train_loader, val_loader


def train_baseline_reward_model(data_path: str = "./rlhf_data/rlhf_dataset.json"):
    config = RewardModelConfig(
        data_path=data_path,
        output_dir="./reward_model_baseline",
        learning_rate=1e-5,
        epochs=3,
        batch_size=8,
    )
    trainer = BaselineRewardModelTrainer(config)
    trainer.train()
    
# 5. Rewrite reward model

def train_rewrite_reward_model(data_path: str = "./rlhf_data/rlhf_dataset.json"):
    config = RewardModelConfig(
        data_path=data_path,
        output_dir="./reward_model",
        base_model="microsoft/deberta-v3-small",
        epochs=3,
        batch_size=8,
        learning_rate=2e-5,
    )
    trainer = RewardModelTrainer(config)
    trainer.train()

# 6. Scoring helper

def score_response(model_dir: str, prompt: str, response: str, config: RewardModelConfig) -> float:
    model_dir = os.path.abspath(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = ScalarRewardModel(config)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pt"), map_location=config.device)
    )
    model.float().to(config.device)
    model.eval()

    text = f"[PROMPT] {prompt} [RESPONSE] {response}"
    enc = tokenizer(
        text,
        max_length=config.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        score = model(
            enc["input_ids"].to(config.device),
            enc["attention_mask"].to(config.device),
        )
    return score.item()

# 7. In-distribution evaluation

IN_DIST_TEST_CASES = [
    {
        "prompt": "Explain gradient descent.",
        "good": (
            "Gradient descent iteratively updates parameters by moving in the direction "
            "of steepest loss decrease, scaled by a learning rate."
        ),
        "bad": "Gradient descent is when the model learns by going downhill somehow.",
    },
    {
        "prompt": "What is the difference between precision and recall?",
        "good": (
            "Precision = TP/(TP+FP): of all predicted positives, how many were correct. "
            "Recall = TP/(TP+FN): of all actual positives, how many were caught. "
            "High recall matters in medical diagnosis; high precision in spam filtering."
        ),
        "bad": "Precision is about being correct and recall is about finding everything.",
    },
    {
        "prompt": "How does BERT differ from GPT?",
        "good": (
            "BERT is a bidirectional encoder pretrained with masked language modeling, "
            "making it strong for classification and QA. GPT is a unidirectional decoder "
            "pretrained with causal language modeling, making it strong for text generation."
        ),
        "bad": "BERT and GPT are both language models but they work differently.",
    },
]


def evaluate_in_distribution():
    rm_config = RewardModelConfig()
    header = f"{'Prompt':<35} {'Model':<12} {'Good':>8} {'Bad':>8} {'Gap':>8}"
    print(header)
    print("-" * 75)

    for tc in IN_DIST_TEST_CASES:
        for label, model_dir in [
            ("Rewrite", "./reward_model/best_model"),
            ("Baseline", "./reward_model_baseline/best_model"),
        ]:
            g = score_response(model_dir, tc["prompt"], tc["good"], rm_config)
            b = score_response(model_dir, tc["prompt"], tc["bad"], rm_config)
            print(f"{tc['prompt'][:34]:<35} {label:<12} {g:>8.4f} {b:>8.4f} {g - b:>8.4f}")
        print()

# 8. Out-of-distribution evaluation

OOD_TEST_CASES = [
    {
        "prompt": "Write a haiku about machine learning.",
        "good": "Weights adjust slowly / Loss descends like autumn leaves / Model learns the truth",
        "bad": "Data flows through nodes / Computers think and learn fast / AI is the future",
    },
    {
        "prompt": "Explain how a refrigerator works.",
        "good": (
            "A refrigerator uses a compressor to pressurize refrigerant gas, which then "
            "expands and absorbs heat from inside the fridge, cooling it down. "
            "The heat is expelled outside via condenser coils."
        ),
        "bad": "A refrigerator keeps things cold using electricity and some chemical stuff inside.",
    },
    {
        "prompt": "What is the capital of Australia?",
        "good": "The capital of Australia is Sydney",
        "bad": (
            "The capital of Australia is Canberra, not Sydney as commonly assumed. "
            "It was purpose-built as a compromise between Sydney and Melbourne "
            "when Australia federated in 1901."
        ),
    },
]


def evaluate_ood():
    rm_config = RewardModelConfig()
    header = f"{'Prompt':<35} {'Model':<12} {'Good':>8} {'Bad':>8} {'Gap':>8}"
    print(header)
    print("-" * 75)

    for tc in OOD_TEST_CASES:
        for label, model_dir in [
            ("Rewrite", "./reward_model/best_model"),
            ("Baseline", "./reward_model_baseline/best_model"),
        ]:
            g = score_response(model_dir, tc["prompt"], tc["good"], rm_config)
            b = score_response(model_dir, tc["prompt"], tc["bad"], rm_config)
            print(f"{tc['prompt'][:34]:<35} {label:<12} {g:>8.4f} {b:>8.4f} {g - b:>8.4f}")
        print()

 # Export models

def export_models():
    shutil.make_archive("reward_model_rewrite",  "zip", "./reward_model/best_model")
    shutil.make_archive("reward_model_baseline", "zip", "./reward_model_baseline/best_model")
    shutil.make_archive("rlhf_dataset",          "zip", "./rlhf_data")
    print("Exported: reward_model_rewrite.zip, reward_model_baseline.zip, rlhf_dataset.zip")

# Entry point

if __name__ == "__main__":
    API_KEY = os.environ.get("GROQ_API_KEY", "")

    check_environment()
    generate_test_dataset(API_KEY)
    generate_full_dataset(API_KEY)
    inspect_dataset()
    train_baseline_reward_model()
    train_rewrite_reward_model()
    evaluate_in_distribution()
    evaluate_ood()
    print_comparison_table()
    export_models()
