"""
Synthetic Dataset Generation Pipeline for Rewrite-Based RLHF
============================================================
Generates paired (prompt, response_a, response_b, rewrite, preference) data
using Groq API (LLaMA-3) for use in reward model training.

Compatible with: Google Colab / Kaggle
Requirements: pip install groq datasets transformers tqdm pandas
"""

import os
import json
import time
import random
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional
from groq import Groq

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    groq_api_key: str = ""                        # Set via env or directly
    model_name: str = "llama-3.3-70b-versatile"   # Fast + high quality on Groq
    num_samples: int = 500
    batch_size: int = 10
    output_dir: str = "./rlhf_data"
    seed: int = 42
    temperature: float = 0.9
    max_retries: int = 3
    delay_between_calls: float = 1.0              # Groq free tier: ~30 req/min


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class RLHFSample:
    sample_id: str
    prompt: str
    response_chosen: str
    response_rejected: str
    rewrite_feedback: str
    rewritten_response: str
    preference_label: int
    quality_score_chosen: float
    quality_score_rejected: float
    domain: str


# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────
SEED_PROMPTS = {
    "instruction_following": [
        "Explain the concept of gradient descent in simple terms.",
        "Write a Python function to check if a string is a palindrome.",
        "Summarize the key differences between supervised and unsupervised learning.",
        "What are the main causes of climate change?",
        "How does the transformer architecture work in NLP?",
        "Explain the bias-variance tradeoff.",
        "Write a function that returns the nth Fibonacci number.",
        "What is the difference between precision and recall?",
        "Explain backpropagation step by step.",
        "How does BERT differ from GPT?",
    ],
    "open_ended_qa": [
        "What makes a good research paper?",
        "How should I approach learning a new programming language?",
        "What are the ethical considerations in AI development?",
        "How does reinforcement learning from human feedback work?",
        "What is the role of attention mechanisms in modern LLMs?",
    ],
    "creative_writing": [
        "Write a short paragraph about the future of AI.",
        "Describe a world where machines and humans collaborate seamlessly.",
        "Write a brief story about a scientist making a breakthrough discovery.",
    ],
    "reasoning": [
        "If all A are B, and some B are C, what can we conclude about A and C?",
        "A car travels 60 mph for 2 hours then 40 mph for 3 hours. What is the average speed?",
        "Explain why correlation does not imply causation with an example.",
    ]
}

SYSTEM_PROMPT = """You are a data generation assistant for RLHF research.
Your job is to generate training pairs for reward model training.
Always respond with valid JSON only. No extra text, no markdown, no explanation outside the JSON."""

USER_TEMPLATE = """Given the following prompt, generate:
1. A HIGH-QUALITY response (detailed, accurate, well-structured, helpful)
2. A LOW-QUALITY response (vague, incomplete, slightly off-topic, or poorly structured)
3. REWRITE FEEDBACK: 2-3 sentences explaining what is wrong with the low-quality response and how to fix it
4. A REWRITTEN response applying that feedback

Prompt: {prompt}

Respond ONLY in this exact JSON format:
{{
  "response_chosen": "...",
  "response_rejected": "...",
  "rewrite_feedback": "...",
  "rewritten_response": "...",
  "quality_score_chosen": <float between 0.7 and 1.0>,
  "quality_score_rejected": <float between 0.1 and 0.5>
}}"""


# ─────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────
class GroqGenerator:
    def __init__(self, config: PipelineConfig):
        api_key = config.groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Groq API key not found.\n"
                "Get a free key at https://console.groq.com\n"
                "Then set: os.environ['GROQ_API_KEY'] = 'your_key'"
            )
        self.client = Groq(api_key=api_key)
        self.config = config

    def generate(self, prompt_text: str) -> Optional[dict]:
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt_text},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=1500,
                    response_format={"type": "json_object"},  # Groq JSON mode
                )
                raw = response.choices[0].message.content.strip()
                return json.loads(raw)

            except json.JSONDecodeError as e:
                print(f"  [Attempt {attempt+1}] JSON parse error: {e}")
            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    wait = 60
                    print(f"  [Attempt {attempt+1}] Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [Attempt {attempt+1}] API error: {e}")
                    time.sleep(2 ** attempt)
        return None


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
class SyntheticDataPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.generator = GroqGenerator(config)
        self.samples = []
        random.seed(config.seed)
        os.makedirs(config.output_dir, exist_ok=True)

    def _get_all_prompts(self) -> list[tuple[str, str]]:
        all_prompts = []
        for domain, prompts in SEED_PROMPTS.items():
            for p in prompts:
                all_prompts.append((domain, p))
        while len(all_prompts) < self.config.num_samples:
            all_prompts.extend(all_prompts)
        random.shuffle(all_prompts)
        return all_prompts[:self.config.num_samples]

    def generate_sample(self, sample_id: str, domain: str, prompt: str) -> Optional[RLHFSample]:
        filled_prompt = USER_TEMPLATE.format(prompt=prompt)
        result = self.generator.generate(filled_prompt)
        if result is None:
            return None

        required_keys = ["response_chosen", "response_rejected", "rewrite_feedback",
                         "rewritten_response", "quality_score_chosen", "quality_score_rejected"]
        if not all(k in result for k in required_keys):
            print(f"  Missing keys in response for sample {sample_id}")
            return None

        return RLHFSample(
            sample_id=sample_id,
            prompt=prompt,
            response_chosen=result["response_chosen"],
            response_rejected=result["response_rejected"],
            rewrite_feedback=result["rewrite_feedback"],
            rewritten_response=result["rewritten_response"],
            preference_label=1,
            quality_score_chosen=float(result.get("quality_score_chosen", 0.8)),
            quality_score_rejected=float(result.get("quality_score_rejected", 0.3)),
            domain=domain
        )

    def run(self):
        prompts = self._get_all_prompts()
        print(f"\n{'='*55}")
        print(f"  Synthetic RLHF Data Generation Pipeline")
        print(f"  Provider : Groq")
        print(f"  Model    : {self.config.model_name}")
        print(f"  Samples  : {self.config.num_samples}")
        print(f"{'='*55}\n")

        failed = 0
        for i, (domain, prompt) in enumerate(tqdm(prompts, desc="Generating")):
            sample_id = f"sample_{i:04d}"
            sample = self.generate_sample(sample_id, domain, prompt)
            if sample:
                self.samples.append(asdict(sample))
            else:
                failed += 1
            time.sleep(self.config.delay_between_calls)

            if (i + 1) % 50 == 0:
                self._save_checkpoint(i + 1)

        print(f"\n✓ Generated: {len(self.samples)} | Failed: {failed}")
        self._save_final()

    def _save_checkpoint(self, step: int):
        path = os.path.join(self.config.output_dir, f"checkpoint_{step}.json")
        with open(path, "w") as f:
            json.dump(self.samples, f, indent=2)
        print(f"  Checkpoint saved → {path}")

    def _save_final(self):
        json_path = os.path.join(self.config.output_dir, "rlhf_dataset.json")
        with open(json_path, "w") as f:
            json.dump(self.samples, f, indent=2)

        df = pd.DataFrame(self.samples)
        csv_path = os.path.join(self.config.output_dir, "rlhf_dataset.csv")
        df.to_csv(csv_path, index=False)
        self._save_hf_format(df)

        print(f"\n✓ Final dataset saved:")
        print(f"   JSON → {json_path}")
        print(f"   CSV  → {csv_path}")
        print(f"   HF   → {self.config.output_dir}/hf_dataset/")

    def _save_hf_format(self, df: pd.DataFrame):
        from datasets import Dataset, DatasetDict
        dataset = Dataset.from_pandas(df)
        split = dataset.train_test_split(test_size=0.1, seed=self.config.seed)
        dd = DatasetDict({"train": split["train"], "validation": split["test"]})
        hf_path = os.path.join(self.config.output_dir, "hf_dataset")
        dd.save_to_disk(hf_path)
        print(f"   Train: {len(split['train'])} | Val: {len(split['test'])}")


# ─────────────────────────────────────────────
# DATASET STATS
# ─────────────────────────────────────────────
def print_dataset_stats(output_dir: str):
    csv_path = os.path.join(output_dir, "rlhf_dataset.csv")
    df = pd.read_csv(csv_path)
    print(f"\n{'='*55}")
    print("  Dataset Statistics")
    print(f"{'='*55}")
    print(f"  Total samples          : {len(df)}")
    print(f"  Domain breakdown       :")
    for domain, count in df["domain"].value_counts().items():
        print(f"    {domain:<30} {count}")
    print(f"  Avg quality (chosen)   : {df['quality_score_chosen'].mean():.3f}")
    print(f"  Avg quality (rejected) : {df['quality_score_rejected'].mean():.3f}")
    print(f"  Quality gap            : {(df['quality_score_chosen']-df['quality_score_rejected']).mean():.3f}")
    print(f"  Avg prompt length      : {df['prompt'].str.len().mean():.0f} chars")
    print(f"  Avg chosen length      : {df['response_chosen'].str.len().mean():.0f} chars")
    print(f"  Avg feedback length    : {df['rewrite_feedback'].str.len().mean():.0f} chars")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    config = PipelineConfig(
        groq_api_key=os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE"),
        num_samples=200,
        output_dir="./rlhf_data",
        delay_between_calls=1.0,
    )
    pipeline = SyntheticDataPipeline(config)
    pipeline.run()
    print_dataset_stats(config.output_dir)
