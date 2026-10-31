# %% Cell 2
# Run this setup cell before the imports below. It installs only missing packages.
import importlib.util
import subprocess
import sys

required_packages = ['datasets', 'transformers', 'accelerate', 'matplotlib',
                     'pandas', 'numpy', 'torch', 'peft', 'trl']
missing_packages = [name for name in required_packages
                    if importlib.util.find_spec(name) is None]
if missing_packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
    importlib.invalidate_caches()
    print('Installed:', ', '.join(missing_packages))
else:
    print('All required packages are installed.')

# %% Cell 3
import re
import gc
import json
import math
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, Dataset, DatasetDict

# %% Cell 4
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs_old_adapters")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Device:", DEVICE)

# %% Cell 5
def clear_memory(*names):
    """Delete large variables by name and clear Python/CUDA memory."""
    for name in names:
        if name in globals():
            del globals()[name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# %% Cell 6
# You may add creative instructions above the final-answer rule.
# If you do nothing, this minimal prompt still runs.
# Keep the final line parseable, or the evaluator may mark correct reasoning as wrong. 
# Do not remove the boxed answer rule. Please edit the first line of the prompt and add your instructions (however long you want).
SYSTEM_PROMPT = """
Your task is to solve the math problem.

Final answer rule:
The last line of your response must be exactly:
\\boxed{ANSWER}
""".strip()

# %% Cell 7
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import SFTTrainer, SFTConfig

MATH_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_EVAL = 100

DATA_DIR = Path("data/gsm8k")
if all((DATA_DIR / f"gsm8k-{split}.arrow").exists() for split in ("train", "test")):
    gsm8k = DatasetDict({
        split: Dataset.from_file(str(DATA_DIR / f"gsm8k-{split}.arrow"))
        for split in ("train", "test")
    })
else:
    gsm8k = load_dataset("openai/gsm8k", "main")
gsm8k_test_100 = gsm8k["test"].select(range(N_EVAL))

def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return text[start:pos - 1]


def extract_gt(raw_answer):
    match = re.search(r"####\s*(.+)", raw_answer)
    return match.group(1).strip().replace(",", "") if match else raw_answer.strip()


def extract_model_answer(text):
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def normalize_answer(ans):
    s = str(ans).strip().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    # fixed regex
    s = re.sub(r"\\(?:left|right|displaystyle)", "", s)
    s = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    
    try:
        v = float(s)
        if math.isfinite(v):
            return str(int(v)) if v == int(v) else str(v)
    except (ValueError, OverflowError):
        pass
    if "/" in s and s.count("/") == 1:
        try:
            a, b = s.split("/")
            return f"{float(a) / float(b):.10g}"
        except (ValueError, ZeroDivisionError):
            pass
    return s


def answers_match(pred, gt):
    p, g = normalize_answer(pred), normalize_answer(gt)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


def load_math_model(model_name=MATH_MODEL, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def make_gsm_prompts(tokenizer, questions, system_prompt=SYSTEM_PROMPT, few_shot_messages=None):
    prompts = []
    for q in questions:
        messages = [{"role": "system", "content": system_prompt}]
        if few_shot_messages:
            messages.extend(few_shot_messages)
        messages.append({"role": "user", "content": q})
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def generate_math(model, tokenizer, questions, system_prompt=SYSTEM_PROMPT, few_shot_messages=None, batch_size=16, max_new_tokens=512):
    responses = []
    for start in range(0, len(questions), batch_size):
        batch_questions = questions[start:start + batch_size]
        prompts = make_gsm_prompts(tokenizer, batch_questions, system_prompt=system_prompt, few_shot_messages=few_shot_messages)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        for i in range(len(batch_questions)):
            responses.append(tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip())
    return responses


def evaluate_gsm8k(model, tokenizer, split, num_samples=100, system_prompt=SYSTEM_PROMPT, few_shot_messages=None, batch_size=16, output_name=None):
    n = min(num_samples, len(split))
    records = []
    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        questions = [split[i]["question"] for i in range(start, end)]
        gts = [extract_gt(split[i]["answer"]) for i in range(start, end)]
        responses = generate_math(model, tokenizer, questions, system_prompt=system_prompt, few_shot_messages=few_shot_messages, batch_size=batch_size)
        for offset, (q, gt, resp) in enumerate(zip(questions, gts, responses)):
            pred = extract_model_answer(resp)
            ok = answers_match(pred, gt)
            correct += int(ok)
            records.append({
                "index": start + offset,
                "question": q,
                "ground_truth": gt,
                "model_response": resp,
                "extracted_answer": pred,
                "correct": ok,
            })
    result = {"accuracy": correct / n, "correct": correct, "total": n, "records": records}
    if output_name:
        path = OUTPUT_DIR / output_name
        path.write_text(json.dumps(result, indent=2))
    return result

# %% Cell 8

LORA_DEFAULTS = {
    "rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 2e-4,
    "epochs": 1,
    "batch_size": 8,
    "grad_accum": 4,
    "max_seq_len": 1024,
}


def build_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_DEFAULTS["rank"],
        lora_alpha=LORA_DEFAULTS["lora_alpha"],
        lora_dropout=LORA_DEFAULTS["lora_dropout"],
        target_modules=LORA_DEFAULTS["target_modules"],
        bias="none",
    )


def format_gsm8k_train_example(example):
    reasoning = example["answer"].split("####")[0].strip()
    final = extract_gt(example["answer"])
    assistant = f"{reasoning}\n\n\\boxed{{{final}}}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": assistant},
        ]
    }


def train_lora_sft(num_samples, output_dir, model_name=MATH_MODEL):
    output_dir = Path(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()

    train_data = gsm8k["train"].select(range(min(num_samples, len(gsm8k["train"]))))
    train_data = train_data.map(format_gsm8k_train_example, remove_columns=train_data.column_names)

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=LORA_DEFAULTS["epochs"],
        per_device_train_batch_size=LORA_DEFAULTS["batch_size"],
        gradient_accumulation_steps=LORA_DEFAULTS["grad_accum"],
        learning_rate=LORA_DEFAULTS["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        max_length=LORA_DEFAULTS["max_seq_len"],
        completion_only_loss=True,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_data, processing_class=tokenizer)
    trainer.train()

    adapter_path = output_dir / "final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    return str(adapter_path)

# %% Cell 12
math_model, math_tokenizer = load_math_model()

# %% Cell 13
# small test run

small_base_result = evaluate_gsm8k(
    math_model,
    math_tokenizer,
    gsm8k_test_100,
    num_samples=3,
    system_prompt=SYSTEM_PROMPT,
    few_shot_messages=None,
    batch_size=3,
    output_name="b1_base_model_small_test.json"
)

small_base_result["accuracy"], small_base_result["correct"], small_base_result["total"]

# %% Cell 14
pd.DataFrame(small_base_result["records"])[
    ["index", "question", "ground_truth", "extracted_answer", "correct", "model_response"]
]

# %% Cell 15
def sync_cuda():    
    if torch.cuda.is_available():        
        torch.cuda.synchronize()

# %% Cell 16
from tqdm.auto import tqdm

def evaluate_gsm8k_progress(
    model,
    tokenizer,
    split,
    num_samples=100,
    system_prompt=SYSTEM_PROMPT,
    few_shot_messages=None,
    batch_size=16,
    output_name=None,
):
    n = min(num_samples, len(split))
    records = []
    correct = 0

    batch_starts = range(0, n, batch_size)

    for start in tqdm(batch_starts, total=math.ceil(n / batch_size), desc="Evaluating GSM8K"):
        end = min(start + batch_size, n)

        questions = [split[i]["question"] for i in range(start, end)]
        gts = [extract_gt(split[i]["answer"]) for i in range(start, end)]

        responses = generate_math(
            model,
            tokenizer,
            questions,
            system_prompt=system_prompt,
            few_shot_messages=few_shot_messages,
            batch_size=batch_size,
        )

        for offset, (q, gt, resp) in enumerate(zip(questions, gts, responses)):
            pred = extract_model_answer(resp)
            ok = answers_match(pred, gt)
            correct += int(ok)

            records.append({
                "index": start + offset,
                "question": q,
                "ground_truth": gt,
                "model_response": resp,
                "extracted_answer": pred,
                "correct": ok,
            })

    result = {
        "accuracy": correct / n,
        "correct": correct,
        "total": n,
        "records": records,
    }

    if output_name:
        path = OUTPUT_DIR / output_name
        path.write_text(json.dumps(result, indent=2))

    return result

# %% Cell 17
start = time.perf_counter()

b1_base_result = evaluate_gsm8k_progress(
    math_model,
    math_tokenizer,
    gsm8k_test_100,
    num_samples=100,
    system_prompt=SYSTEM_PROMPT,
    few_shot_messages=None,
    batch_size=16,
    output_name="b1_base_model_100_gsm8k.json",
)

sync_cuda()
b1_base_runtime = time.perf_counter() - start

print(f"B.1 base model accuracy: {b1_base_result['accuracy']:.4f}")
print(f"Correct: {b1_base_result['correct']}/{b1_base_result['total']}")
print(f"Runtime: {b1_base_runtime:.2f} sec")

# %% Cell 18
b1_base_accuracy = b1_base_result["accuracy"]
b1_base_correct = b1_base_result["correct"]
b1_base_total = b1_base_result["total"]

print(f"B.1 base model accuracy: {b1_base_accuracy:.4f}")
print(f"Correct: {b1_base_correct}/{b1_base_total}")
print(f"Runtime: {b1_base_runtime:.2f} sec")

# %% Cell 19
b1_base_summary_df = pd.DataFrame([{
    "setting": "base_model_zero_shot",
    "model": MATH_MODEL,
    "num_eval_questions": b1_base_total,
    "correct": b1_base_correct,
    "accuracy": b1_base_accuracy,
    "runtime_sec": b1_base_runtime,
    "system_prompt": SYSTEM_PROMPT,
    "few_shot": False,
    "batch_size": 16,
}])

display(b1_base_summary_df)

b1_base_summary_df.to_csv(
    OUTPUT_DIR / "b1_base_model_summary.csv",
    index=False
)

# %% Cell 20
b1_records_df = pd.DataFrame(b1_base_result["records"])

display(
    b1_records_df[
        ["index", "ground_truth", "extracted_answer", "correct", "question", "model_response"]
    ].head(10)
)

display(
    b1_records_df[
        ~b1_records_df["correct"]
    ][
        ["index", "ground_truth", "extracted_answer", "question", "model_response"]
    ].head(5)
)

# %% Cell 22
b1_records_df = pd.DataFrame(b1_base_result["records"])

b2_wrong_df = b1_records_df[~b1_records_df["correct"]].copy()

print(f"Number of incorrect base-model answers: {len(b2_wrong_df)}")
print(f"Number of correct base-model answers: {b1_records_df['correct'].sum()}")
print(f"Total evaluated examples: {len(b1_records_df)}")

display(
    b2_wrong_df[
        ["index", "ground_truth", "extracted_answer", "question", "model_response"]
    ].head(10)
)

# %% Cell 23
def compact_text(text, max_chars=1200):
    text = str(text).replace("\n\n", "\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."

def print_gsm8k_case(row, max_chars=1600):
    print("=" * 100)
    print(f"Index: {row['index']}")
    print(f"Ground truth answer: {row['ground_truth']}")
    print(f"Extracted model answer: {row['extracted_answer']}")
    print("-" * 100)
    print("Question:")
    print(row["question"])
    print("-" * 100)
    print("Model response excerpt:")
    print(compact_text(row["model_response"], max_chars=max_chars))
    print("=" * 100)
    print()

# %% Cell 24
# ---- IMPORTANT ----
# Note that in reruns, these two indexes are not guaranteed to be the reponses that are "boxed but wrong",
# since they depend on the random sample of 100 test questions and the model's responses.
# Therefore we commented out the assertions that check for specific indexes and correctness, 
# which allows the notebook to run through even if the exact failure cases differ.

b2_selected_indices = [2, 4, 6]

b2_selected_wrong = b1_records_df[
    b1_records_df["index"].isin(b2_selected_indices)
].copy()

# assert len(b2_selected_wrong) == len(b2_selected_indices)
# assert (~b2_selected_wrong["correct"]).all()

for _, row in b2_selected_wrong.iterrows():
    print_gsm8k_case(row, max_chars=2500)

# %% Cell 25
def make_excerpt(text, max_chars=500):
    text = str(text).strip()
    text = text.replace("\n\n", "\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."

b2_selected_wrong["model_excerpt"] = b2_selected_wrong["model_response"].apply(
    lambda x: make_excerpt(x, max_chars=600)
)

display(
    b2_selected_wrong[
        ["index", "ground_truth", "extracted_answer", "correct", "question", "model_excerpt"]
    ]
)

# %% Cell 28
lora_defaults_df = pd.DataFrame([
    {
        "hyperparameter": key,
        "value": value,
    }
    for key, value in LORA_DEFAULTS.items()
])

display(lora_defaults_df)

# %% Cell 29
b3_example_raw = gsm8k["train"][0]
b3_example_formatted = format_gsm8k_train_example(b3_example_raw)

print("Raw question:")
print(b3_example_raw["question"])
print("\nRaw answer:")
print(b3_example_raw["answer"])

print("\nFormatted chat messages:")
for msg in b3_example_formatted["messages"]:
    print("=" * 80)
    print("Role:", msg["role"])
    print(msg["content"])

# %% Cell 31
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


clear_memory("math_model", "math_tokenizer")

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

param_model = AutoModelForCausalLM.from_pretrained(
    MATH_MODEL,
    torch_dtype=dtype,
    device_map={"": "cpu"},
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)

base_total_params, base_trainable_before_lora = count_parameters(param_model)

print(f"Base model total parameters: {base_total_params:,}")
print(f"Base model trainable parameters before LoRA: {base_trainable_before_lora:,}")

# %% Cell 32
param_lora_model = get_peft_model(param_model, build_lora_config())

lora_total_params, lora_trainable_params = count_parameters(param_lora_model)

percent_trainable_vs_base = 100 * lora_trainable_params / base_total_params
reduction_factor = base_total_params / lora_trainable_params

print(f"Total parameters after adding LoRA adapters: {lora_total_params:,}")
print(f"Trainable LoRA parameters: {lora_trainable_params:,}")
print(f"Percentage of base parameters trained: {percent_trainable_vs_base:.4f}%")
print(f"Reduction factor compared with full fine-tuning: {reduction_factor:.1f}x")

# %% Cell 33
b4_param_summary_df = pd.DataFrame([{
    "model": MATH_MODEL,
    "lora_rank": LORA_DEFAULTS["rank"],
    "lora_alpha": LORA_DEFAULTS["lora_alpha"],
    "lora_dropout": LORA_DEFAULTS["lora_dropout"],
    "target_modules": ", ".join(LORA_DEFAULTS["target_modules"]),
    "base_total_parameters": base_total_params,
    "trainable_lora_parameters": lora_trainable_params,
    "percent_trainable_vs_base": percent_trainable_vs_base,
    "full_finetuning_reduction_factor": reduction_factor,
}])

display(b4_param_summary_df)

b4_param_summary_df.to_csv(
    OUTPUT_DIR / "b4_lora_parameter_count_summary.csv",
    index=False
)

# %% Cell 34
clear_memory(
    "math_model",
    "math_tokenizer",
    "param_model",
    "param_lora_model",
    "lora_1k_model",
    "lora_1k_tokenizer",
)

# %% Cell 37
from transformers import set_seed

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %% Cell 39
LORA_DEFAULTS = {
    "rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 2e-4,
    "epochs": 1,
    "batch_size": 2,       # reduced from 8 to avoid CUDA OOM
    "grad_accum": 32,      # keeps effective batch size at 32
    "max_seq_len": 1024,
}

# %% Cell 40
lora_1k_output_dir = OUTPUT_DIR / "lora_gsm8k_1k"
lora_1k_adapter_path = lora_1k_output_dir / "final_adapter"

start = time.perf_counter()

if lora_1k_adapter_path.exists():
    print(f"Found existing 1k LoRA adapter at: {lora_1k_adapter_path}")
    print("Skipping training and using the existing adapter.")
    lora_1k_adapter_path = str(lora_1k_adapter_path)
else:
    lora_1k_adapter_path = train_lora_sft(
        num_samples=1000,
        output_dir=lora_1k_output_dir,
        model_name=MATH_MODEL,
    )

sync_cuda()
b5_train_runtime = time.perf_counter() - start

print(f"1k LoRA adapter path: {lora_1k_adapter_path}")
print(f"B.5 1k LoRA training/runtime cell time: {b5_train_runtime:.2f} sec")

# %% Cell 41
clear_memory("math_model", "math_tokenizer")

lora_1k_model, lora_1k_tokenizer = load_math_model(
    model_name=MATH_MODEL,
    lora_path=lora_1k_adapter_path,
)

# %% Cell 42
small_lora_1k_result = evaluate_gsm8k_progress(
    lora_1k_model,
    lora_1k_tokenizer,
    gsm8k_test_100,
    num_samples=3,
    system_prompt=SYSTEM_PROMPT,
    few_shot_messages=None,
    batch_size=3,
    output_name="b5_lora_1k_small_test.json",
)

print(
    small_lora_1k_result["accuracy"],
    small_lora_1k_result["correct"],
    small_lora_1k_result["total"],
)

pd.DataFrame(small_lora_1k_result["records"])[
    ["index", "ground_truth", "extracted_answer", "correct", "question", "model_response"]
]

# %% Cell 43
b5_small_df = pd.DataFrame(small_lora_1k_result["records"]).copy()

b5_small_df["has_boxed"] = b5_small_df["model_response"].str.contains(
    r"\\boxed\{",
    regex=True
)

b5_small_df["last_line"] = b5_small_df["model_response"].apply(
    lambda x: str(x).strip().splitlines()[-1] if str(x).strip() else ""
)

display(
    b5_small_df[
        [
            "index",
            "ground_truth",
            "extracted_answer",
            "correct",
            "has_boxed",
            "last_line",
        ]
    ]
)

for _, row in b5_small_df.iterrows():
    print("=" * 100)
    print(f"Index: {row['index']}")
    print(f"Ground truth: {row['ground_truth']}")
    print(f"Extracted answer: {row['extracted_answer']}")
    print(f"Correct: {row['correct']}")
    print(f"Has boxed answer: {row['has_boxed']}")
    print("-" * 100)
    print(row["model_response"])
    print()

# %% Cell 44
start = time.perf_counter()

b5_lora_1k_result = evaluate_gsm8k_progress(
    lora_1k_model,
    lora_1k_tokenizer,
    gsm8k_test_100,
    num_samples=100,
    system_prompt=SYSTEM_PROMPT,
    few_shot_messages=None,
    batch_size=16,
    output_name="b5_lora_1k_100_gsm8k.json",
)

sync_cuda()
b5_eval_runtime = time.perf_counter() - start

b5_lora_1k_accuracy = b5_lora_1k_result["accuracy"]
b5_lora_1k_correct = b5_lora_1k_result["correct"]
b5_lora_1k_total = b5_lora_1k_result["total"]

print(f"B.5 1k LoRA accuracy: {b5_lora_1k_accuracy:.4f}")
print(f"Correct: {b5_lora_1k_correct}/{b5_lora_1k_total}")
print(f"Evaluation runtime: {b5_eval_runtime:.2f} sec")
print(f"Improvement over B.1 baseline: {b5_lora_1k_accuracy - b1_base_accuracy:+.4f}")

# %% Cell 45
b5_lora_1k_summary_df = pd.DataFrame([{
    "setting": "lora_sft_1k",
    "model": MATH_MODEL,
    "adapter_path": lora_1k_adapter_path,
    "num_train_examples": 1000,
    "num_eval_questions": b5_lora_1k_total,
    "correct": b5_lora_1k_correct,
    "accuracy": b5_lora_1k_accuracy,
    "baseline_accuracy": b1_base_accuracy,
    "accuracy_improvement_over_baseline": b5_lora_1k_accuracy - b1_base_accuracy,
    "train_runtime_sec": b5_train_runtime,
    "eval_runtime_sec": b5_eval_runtime,
    "lora_rank": LORA_DEFAULTS["rank"],
    "lora_alpha": LORA_DEFAULTS["lora_alpha"],
    "lora_dropout": LORA_DEFAULTS["lora_dropout"],
    "target_modules": ", ".join(LORA_DEFAULTS["target_modules"]),
    "learning_rate": LORA_DEFAULTS["learning_rate"],
    "epochs": LORA_DEFAULTS["epochs"],
    "batch_size": LORA_DEFAULTS["batch_size"],
    "grad_accum": LORA_DEFAULTS["grad_accum"],
    "max_seq_len": LORA_DEFAULTS["max_seq_len"],
    "system_prompt": SYSTEM_PROMPT,
    "few_shot": False,
}])

display(b5_lora_1k_summary_df)

b5_lora_1k_summary_df.to_csv(
    OUTPUT_DIR / "b5_lora_1k_summary.csv",
    index=False,
)

# %% Cell 46
b5_lora_1k_records_df = pd.DataFrame(b5_lora_1k_result["records"])

display(
    b5_lora_1k_records_df[
        ["index", "ground_truth", "extracted_answer", "correct", "question", "model_response"]
    ].head(10)
)

display(
    b5_lora_1k_records_df[
        ~b5_lora_1k_records_df["correct"]
    ][
        ["index", "ground_truth", "extracted_answer", "question", "model_response"]
    ].head(5)
)

# %% Cell 47
b5_baseline_vs_lora_1k = pd.DataFrame([
    {
        "setting": "base_model",
        "num_train_examples": 0,
        "correct": b1_base_correct,
        "total": b1_base_total,
        "accuracy": b1_base_accuracy,
    },
    {
        "setting": "lora_sft_1k",
        "num_train_examples": 1000,
        "correct": b5_lora_1k_correct,
        "total": b5_lora_1k_total,
        "accuracy": b5_lora_1k_accuracy,
    },
])

b5_baseline_vs_lora_1k["delta_vs_base"] = (
    b5_baseline_vs_lora_1k["accuracy"] - b1_base_accuracy
)

display(b5_baseline_vs_lora_1k)

b5_baseline_vs_lora_1k.to_csv(
    OUTPUT_DIR / "b5_baseline_vs_lora_1k.csv",
    index=False,
)

# %% Cell 54
clear_memory(
    "math_model",
    "math_tokenizer",
    "lora_1k_model",
    "lora_1k_tokenizer",
    "lora_2k_model",
    "lora_2k_tokenizer",
)

# %% Cell 55
LORA_DEFAULTS = {
    "rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 2e-4,
    "epochs": 1,
    "batch_size": 2,       # OOM-safe micro-batch size
    "grad_accum": 32,      # keeps effective batch size at 32
    "max_seq_len": 1024,
}

print("Effective train batch size:", LORA_DEFAULTS["batch_size"] * LORA_DEFAULTS["grad_accum"])

# %% Cell 56
import shutil

lora_2k_output_dir = OUTPUT_DIR / "lora_gsm8k_2k"
lora_2k_adapter_path = lora_2k_output_dir / "final_adapter"

if lora_2k_output_dir.exists() and not lora_2k_adapter_path.exists():
    print("Removing incomplete 2k LoRA output directory.")
    shutil.rmtree(lora_2k_output_dir)

start = time.perf_counter()

if lora_2k_adapter_path.exists():
    print(f"Found existing 2k LoRA adapter at: {lora_2k_adapter_path}")
    print("Skipping training and using the existing adapter.")
    lora_2k_adapter_path = str(lora_2k_adapter_path)
else:
    lora_2k_adapter_path = train_lora_sft(
        num_samples=2000,
        output_dir=lora_2k_output_dir,
        model_name=MATH_MODEL,
    )

# %% Cell 57

sync_cuda()
b7_train_runtime = time.perf_counter() - start

print(f"2k LoRA adapter path: {lora_2k_adapter_path}")
print(f"B.7 2k LoRA training/runtime cell time: {b7_train_runtime:.2f} sec")

# %% Cell 58
clear_memory("math_model", "math_tokenizer", "lora_1k_model", "lora_1k_tokenizer")

lora_2k_model, lora_2k_tokenizer = load_math_model(
    model_name=MATH_MODEL,
    lora_path=lora_2k_adapter_path,
)

# %% Cell 59
small_lora_2k_result = evaluate_gsm8k_progress(
    lora_2k_model,
    lora_2k_tokenizer,
    gsm8k_test_100,
    num_samples=3,
    system_prompt=SYSTEM_PROMPT,
    few_shot_messages=None,
    batch_size=3,
    output_name="b7_lora_2k_small_test.json",
)

print(
    small_lora_2k_result["accuracy"],
    small_lora_2k_result["correct"],
    small_lora_2k_result["total"],
)

b7_small_df = pd.DataFrame(small_lora_2k_result["records"]).copy()
b7_small_df["has_boxed"] = b7_small_df["model_response"].str.contains(
    r"\\boxed\{",
    regex=True
)
b7_small_df["last_line"] = b7_small_df["model_response"].apply(
    lambda x: str(x).strip().splitlines()[-1] if str(x).strip() else ""
)

display(
    b7_small_df[
        [
            "index",
            "ground_truth",
            "extracted_answer",
            "correct",
            "has_boxed",
            "last_line",
        ]
    ]
)

for _, row in b7_small_df.iterrows():
    print("=" * 100)
    print(f"Index: {row['index']}")
    print(f"Ground truth: {row['ground_truth']}")
    print(f"Extracted answer: {row['extracted_answer']}")
    print(f"Correct: {row['correct']}")
    print(f"Has boxed answer: {row['has_boxed']}")
    print("-" * 100)
    print(row["model_response"])
    print()

# %% Cell 60
start = time.perf_counter()

b7_lora_2k_result = evaluate_gsm8k_progress(
    lora_2k_model,
    lora_2k_tokenizer,
    gsm8k_test_100,
    num_samples=100,
    system_prompt=SYSTEM_PROMPT,
    few_shot_messages=None,
    batch_size=16,
    output_name="b7_lora_2k_100_gsm8k.json",
)

sync_cuda()
b7_eval_runtime = time.perf_counter() - start

b7_lora_2k_accuracy = b7_lora_2k_result["accuracy"]
b7_lora_2k_correct = b7_lora_2k_result["correct"]
b7_lora_2k_total = b7_lora_2k_result["total"]

print(f"B.7 2k LoRA accuracy: {b7_lora_2k_accuracy:.4f}")
print(f"Correct: {b7_lora_2k_correct}/{b7_lora_2k_total}")
print(f"Evaluation runtime: {b7_eval_runtime:.2f} sec")
print(f"Improvement over B.1 baseline: {b7_lora_2k_accuracy - b1_base_accuracy:+.4f}")
print(f"Improvement over B.5 1k LoRA: {b7_lora_2k_accuracy - b5_lora_1k_accuracy:+.4f}")

# %% Cell 61
b7_lora_2k_summary_df = pd.DataFrame([{
    "setting": "lora_sft_2k",
    "model": MATH_MODEL,
    "adapter_path": lora_2k_adapter_path,
    "num_train_examples": 2000,
    "num_eval_questions": b7_lora_2k_total,
    "correct": b7_lora_2k_correct,
    "accuracy": b7_lora_2k_accuracy,
    "baseline_accuracy": b1_base_accuracy,
    "lora_1k_accuracy": b5_lora_1k_accuracy,
    "accuracy_improvement_over_baseline": b7_lora_2k_accuracy - b1_base_accuracy,
    "accuracy_improvement_over_1k": b7_lora_2k_accuracy - b5_lora_1k_accuracy,
    "train_runtime_sec": b7_train_runtime,
    "eval_runtime_sec": b7_eval_runtime,
    "lora_rank": LORA_DEFAULTS["rank"],
    "lora_alpha": LORA_DEFAULTS["lora_alpha"],
    "lora_dropout": LORA_DEFAULTS["lora_dropout"],
    "target_modules": ", ".join(LORA_DEFAULTS["target_modules"]),
    "learning_rate": LORA_DEFAULTS["learning_rate"],
    "epochs": LORA_DEFAULTS["epochs"],
    "train_batch_size": LORA_DEFAULTS["batch_size"],
    "grad_accum": LORA_DEFAULTS["grad_accum"],
    "effective_train_batch_size": LORA_DEFAULTS["batch_size"] * LORA_DEFAULTS["grad_accum"],
    "max_seq_len": LORA_DEFAULTS["max_seq_len"],
    "system_prompt": SYSTEM_PROMPT,
    "few_shot": False,
}])

display(b7_lora_2k_summary_df)

b7_lora_2k_summary_df.to_csv(
    OUTPUT_DIR / "b7_lora_2k_summary.csv",
    index=False,
)

# %% Cell 62
b7_lora_2k_records_df = pd.DataFrame(b7_lora_2k_result["records"])

display(
    b7_lora_2k_records_df[
        ["index", "ground_truth", "extracted_answer", "correct", "question", "model_response"]
    ].head(10)
)

display(
    b7_lora_2k_records_df[
        ~b7_lora_2k_records_df["correct"]
    ][
        ["index", "ground_truth", "extracted_answer", "question", "model_response"]
    ].head(5)
)

# %% Cell 63
b7_scaling_results_df = pd.DataFrame([
    {
        "setting": "base_model",
        "num_train_examples": 0,
        "correct": b1_base_correct,
        "total": b1_base_total,
        "accuracy": b1_base_accuracy,
    },
    {
        "setting": "lora_sft_1k",
        "num_train_examples": 1000,
        "correct": b5_lora_1k_correct,
        "total": b5_lora_1k_total,
        "accuracy": b5_lora_1k_accuracy,
    },
    {
        "setting": "lora_sft_2k",
        "num_train_examples": 2000,
        "correct": b7_lora_2k_correct,
        "total": b7_lora_2k_total,
        "accuracy": b7_lora_2k_accuracy,
    },
])

b7_scaling_results_df["delta_vs_base"] = (
    b7_scaling_results_df["accuracy"] - b1_base_accuracy
)

display(b7_scaling_results_df)

b7_scaling_results_df.to_csv(
    OUTPUT_DIR / "b7_accuracy_vs_training_size.csv",
    index=False,
)

# %% Cell 64
plt.figure(figsize=(6, 4))
plt.plot(
    b7_scaling_results_df["num_train_examples"],
    b7_scaling_results_df["accuracy"],
    marker="o",
)

plt.xlabel("Number of GSM8K training examples")
plt.ylabel("Accuracy on 100 GSM8K test questions")
plt.title("LoRA SFT scaling on GSM8K")
plt.xticks([0, 1000, 2000])
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = OUTPUT_DIR / "b7_accuracy_vs_training_size.png"
plt.savefig(plot_path, dpi=200)
plt.show()

print(f"Saved plot to: {plot_path}")

# %% Cell 65
print("B.7 scaling summary")
print("-" * 80)
print(f"Base model:    {b1_base_correct}/{b1_base_total}, accuracy = {b1_base_accuracy:.4f}")
print(f"LoRA 1k model: {b5_lora_1k_correct}/{b5_lora_1k_total}, accuracy = {b5_lora_1k_accuracy:.4f}")
print(f"LoRA 2k model: {b7_lora_2k_correct}/{b7_lora_2k_total}, accuracy = {b7_lora_2k_accuracy:.4f}")
print()
print(f"1k gain over base: {b5_lora_1k_accuracy - b1_base_accuracy:+.4f}")
print(f"2k gain over base: {b7_lora_2k_accuracy - b1_base_accuracy:+.4f}")
print(f"2k gain over 1k:   {b7_lora_2k_accuracy - b5_lora_1k_accuracy:+.4f}")
print()
print(f"2k training runtime: {b7_train_runtime:.2f} sec")
print(f"2k evaluation runtime: {b7_eval_runtime:.2f} sec")

# %% Cell 67
b8_selected_indices = [2, 4, 6]

b1_records_df = pd.DataFrame(b1_base_result["records"])
b7_lora_2k_records_df = pd.DataFrame(b7_lora_2k_result["records"])

best_sft_name = "LoRA SFT 2k"
best_sft_records_df = b7_lora_2k_records_df.copy()
best_sft_accuracy = b7_lora_2k_accuracy

print(f"Best SFT model used for B.8: {best_sft_name}")
print(f"Best SFT accuracy: {best_sft_accuracy:.4f}")

# %% Cell 68
def make_response_excerpt(text, max_chars=900):
    text = str(text).strip().replace("\n\n", "\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."

def get_record_by_index(records_df, idx):
    rows = records_df[records_df["index"] == idx]
    assert len(rows) == 1, f"Expected exactly one row for index {idx}, found {len(rows)}"
    return rows.iloc[0]

# %% Cell 69
b8_rows = []

for idx in b8_selected_indices:
    base_row = get_record_by_index(b1_records_df, idx)
    sft_row = get_record_by_index(best_sft_records_df, idx)

    assert base_row["question"] == sft_row["question"], f"Question mismatch at index {idx}"
    assert str(base_row["ground_truth"]) == str(sft_row["ground_truth"]), f"Ground truth mismatch at index {idx}"

    b8_rows.append({
        "index": idx,
        "question": base_row["question"],
        "ground_truth": base_row["ground_truth"],

        "base_extracted_answer": base_row["extracted_answer"],
        "base_correct": base_row["correct"],
        "base_response_excerpt": make_response_excerpt(base_row["model_response"], max_chars=900),
        "base_full_response": base_row["model_response"],

        "sft_model": best_sft_name,
        "sft_extracted_answer": sft_row["extracted_answer"],
        "sft_correct": sft_row["correct"],
        "sft_response_excerpt": make_response_excerpt(sft_row["model_response"], max_chars=900),
        "sft_full_response": sft_row["model_response"],

        "fixed_by_sft": (not bool(base_row["correct"])) and bool(sft_row["correct"]),
    })

b8_comparison_df = pd.DataFrame(b8_rows)

display_cols = [
    "index",
    "ground_truth",
    "base_extracted_answer",
    "base_correct",
    "sft_extracted_answer",
    "sft_correct",
    "fixed_by_sft",
    "question",
    "base_response_excerpt",
    "sft_response_excerpt",
]

pd.set_option("display.max_colwidth", 250)
display(b8_comparison_df[display_cols])

# %% Cell 70

b8_num_fixed = int(b8_comparison_df["fixed_by_sft"].sum())
b8_total = len(b8_comparison_df)

print(f"B.8 selected failure cases: {b8_total}")
print(f"Cases fixed by {best_sft_name}: {b8_num_fixed}/{b8_total}")

b8_fix_summary_df = pd.DataFrame([{
    "best_sft_model": best_sft_name,
    "base_accuracy": b1_base_accuracy,
    "best_sft_accuracy": best_sft_accuracy,
    "selected_failure_cases": b8_total,
    "fixed_by_sft": b8_num_fixed,
    "not_fixed_by_sft": b8_total - b8_num_fixed,
}])

display(b8_fix_summary_df)

# %% Cell 71
def print_b8_case(row):
    print("=" * 120)
    print(f"Index: {row['index']}")
    print(f"Ground truth: {row['ground_truth']}")
    print("-" * 120)
    print("Question:")
    print(row["question"])
    print("-" * 120)
    print(f"BASE MODEL")
    print(f"Extracted answer: {row['base_extracted_answer']}")
    print(f"Correct: {row['base_correct']}")
    print()
    print(row["base_full_response"])
    print("-" * 120)
    print(f"{row['sft_model']}")
    print(f"Extracted answer: {row['sft_extracted_answer']}")
    print(f"Correct: {row['sft_correct']}")
    print()
    print(row["sft_full_response"])
    print("-" * 120)
    print(f"Fixed by SFT: {row['fixed_by_sft']}")
    print("=" * 120)
    print()

for _, row in b8_comparison_df.iterrows():
    print_b8_case(row)

# %% Cell 72
b8_comparison_df.to_csv(
    OUTPUT_DIR / "b8_base_vs_best_sft_selected_failures.csv",
    index=False,
)

b8_fix_summary_df.to_csv(
    OUTPUT_DIR / "b8_base_vs_best_sft_fix_summary.csv",
    index=False,
)

# Optional: save a text file with full responses, easier to copy into the report.
b8_text_path = OUTPUT_DIR / "b8_base_vs_best_sft_full_responses.txt"

with open(b8_text_path, "w", encoding="utf-8") as f:
    for _, row in b8_comparison_df.iterrows():
        f.write("=" * 120 + "\n")
        f.write(f"Index: {row['index']}\n")
        f.write(f"Ground truth: {row['ground_truth']}\n")
        f.write("-" * 120 + "\n")
        f.write("Question:\n")
        f.write(str(row["question"]) + "\n")
        f.write("-" * 120 + "\n")
        f.write("BASE MODEL\n")
        f.write(f"Extracted answer: {row['base_extracted_answer']}\n")
        f.write(f"Correct: {row['base_correct']}\n\n")
        f.write(str(row["base_full_response"]) + "\n")
        f.write("-" * 120 + "\n")
        f.write(f"{row['sft_model']}\n")
        f.write(f"Extracted answer: {row['sft_extracted_answer']}\n")
        f.write(f"Correct: {row['sft_correct']}\n\n")
        f.write(str(row["sft_full_response"]) + "\n")
        f.write("-" * 120 + "\n")
        f.write(f"Fixed by SFT: {row['fixed_by_sft']}\n")
        f.write("=" * 120 + "\n\n")

print(f"Saved B.8 comparison table to: {OUTPUT_DIR / 'b8_base_vs_best_sft_selected_failures.csv'}")
print(f"Saved B.8 full responses to: {b8_text_path}")

# %% Cell 74
best_sft_name = "LoRA SFT 2k"
best_sft_accuracy = b7_lora_2k_accuracy
best_sft_records_df = pd.DataFrame(b7_lora_2k_result["records"]).copy()

print(f"Best SFT model for B.9: {best_sft_name}")
print(f"Best SFT accuracy: {best_sft_accuracy:.4f}")

# %% Cell 75
b9_failures_df = best_sft_records_df[
    ~best_sft_records_df["correct"]
].copy()

print(f"Number of failures for {best_sft_name}: {len(b9_failures_df)}")
print(f"Number correct for {best_sft_name}: {best_sft_records_df['correct'].sum()}")
print(f"Total evaluated examples: {len(best_sft_records_df)}")

b9_failures_df["has_boxed"] = b9_failures_df["model_response"].str.contains(
    r"\\boxed\{",
    regex=True
)

b9_failures_df["last_line"] = b9_failures_df["model_response"].apply(
    lambda x: str(x).strip().splitlines()[-1] if str(x).strip() else ""
)

display(
    b9_failures_df[
        [
            "index",
            "ground_truth",
            "extracted_answer",
            "correct",
            "has_boxed",
            "last_line",
            "question",
            "model_response",
        ]
    ].head(10)
)

# %% Cell 76
def compact_text(text, max_chars=1600):
    text = str(text).replace("\n\n", "\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]..."

def print_b9_failure_case(row, max_chars=2500):
    print("=" * 120)
    print(f"Index: {row['index']}")
    print(f"Ground truth answer: {row['ground_truth']}")
    print(f"Extracted SFT answer: {row['extracted_answer']}")
    print(f"Correct: {row['correct']}")
    print(f"Has boxed answer: {row.get('has_boxed', 'N/A')}")
    print("-" * 120)
    print("Question:")
    print(row["question"])
    print("-" * 120)
    print(f"{best_sft_name} response:")
    print(compact_text(row["model_response"], max_chars=max_chars))
    print("=" * 120)
    print()

# %% Cell 77
for _, row in b9_failures_df.head(8).iterrows():
    print_b9_failure_case(row, max_chars=2200)

# %% Cell 78
# ---- IMPORTANT ----
# Note that in reruns, these two indexes are not guaranteed to be the reponses that are "boxed but wrong",
# since they depend on the random sample of 100 test questions and the model's responses.
# Therefore we commented out the assertions that check for specific indexes and correctness, 
# which allows the notebook to run through even if the exact failure cases differ.

b9_selected_indices = [5, 7]

b9_selected_failures = best_sft_records_df[
    best_sft_records_df["index"].isin(b9_selected_indices)
].copy()

# assert len(b9_selected_failures) == len(b9_selected_indices)
# assert (~b9_selected_failures["correct"]).all()

b9_selected_failures["has_boxed"] = b9_selected_failures["model_response"].str.contains(
    r"\\boxed\{",
    regex=True
)

b9_selected_failures["last_line"] = b9_selected_failures["model_response"].apply(
    lambda x: str(x).strip().splitlines()[-1] if str(x).strip() else ""
)

for _, row in b9_selected_failures.iterrows():
    print_b9_failure_case(row, max_chars=4000)

# %% Cell 80

def few_shot_messages_from_train(train_split, indices=(0, 1, 2)):
    messages = []
    for idx in indices:
        ex = train_split[int(idx)]
        reasoning = ex["answer"].split("####")[0].strip()
        final = extract_gt(ex["answer"])
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": f"{reasoning}\n\n\\boxed{{{final}}}"})
    return messages

# %% Cell 84
def few_shot_messages_from_train(train_split, indices=(0, 1, 2)):
    messages = []
    for idx in indices:
        ex = train_split[int(idx)]
        reasoning = ex["answer"].split("####")[0].strip()
        final = extract_gt(ex["answer"])

        messages.append({
            "role": "user",
            "content": ex["question"],
        })

        messages.append({
            "role": "assistant",
            "content": f"{reasoning}\n\n\\boxed{{{final}}}",
        })

    return messages


B10_FEW_SHOT_INDICES = (0, 1, 2)
b10_few_shot_messages = few_shot_messages_from_train(
    gsm8k["train"],
    indices=B10_FEW_SHOT_INDICES,
)

print(f"Using {len(B10_FEW_SHOT_INDICES)} fixed few-shot demonstrations:")
print("Training indices:", B10_FEW_SHOT_INDICES)

for i, idx in enumerate(B10_FEW_SHOT_INDICES, start=1):
    ex = gsm8k["train"][idx]
    print("=" * 100)
    print(f"Demonstration {i}, train index {idx}")
    print("Question:")
    print(ex["question"])
    print("\nGround-truth final answer:", extract_gt(ex["answer"]))
    print("\nAssistant demonstration:")
    print(b10_few_shot_messages[2 * (i - 1) + 1]["content"])

# %% Cell 85
b10_few_shot_examples_df = pd.DataFrame([
    {
        "demo_number": i + 1,
        "train_index": idx,
        "question": gsm8k["train"][idx]["question"],
        "raw_answer": gsm8k["train"][idx]["answer"],
        "final_answer": extract_gt(gsm8k["train"][idx]["answer"]),
        "assistant_demo": b10_few_shot_messages[2 * i + 1]["content"],
    }
    for i, idx in enumerate(B10_FEW_SHOT_INDICES)
])

display(b10_few_shot_examples_df)

b10_few_shot_examples_df.to_csv(
    OUTPUT_DIR / "b10_fixed_3shot_demonstrations.csv",
    index=False,
)

# %% Cell 86
if "lora_2k_model" not in globals() or "lora_2k_tokenizer" not in globals():
    clear_memory("math_model", "math_tokenizer", "lora_1k_model", "lora_1k_tokenizer")
    lora_2k_model, lora_2k_tokenizer = load_math_model(
        model_name=MATH_MODEL,
        lora_path=lora_2k_adapter_path,
    )

start = time.perf_counter()

b10_lora_2k_3shot_result = evaluate_gsm8k_progress(
    lora_2k_model,
    lora_2k_tokenizer,
    gsm8k_test_100,
    num_samples=100,
    system_prompt=SYSTEM_PROMPT,
    few_shot_messages=b10_few_shot_messages,
    batch_size=8,   # safer than 16 because the prompt is longer
    output_name="b10_lora_2k_3shot_100_gsm8k.json",
)

sync_cuda()
b10_lora_2k_3shot_runtime = time.perf_counter() - start

b10_lora_2k_3shot_accuracy = b10_lora_2k_3shot_result["accuracy"]
b10_lora_2k_3shot_correct = b10_lora_2k_3shot_result["correct"]
b10_lora_2k_3shot_total = b10_lora_2k_3shot_result["total"]

print(f"B.10 2k LoRA 3-shot accuracy: {b10_lora_2k_3shot_accuracy:.4f}")
print(f"Correct: {b10_lora_2k_3shot_correct}/{b10_lora_2k_3shot_total}")
print(f"Runtime: {b10_lora_2k_3shot_runtime:.2f} sec")
print(f"Delta vs 2k LoRA no-demo baseline: {b10_lora_2k_3shot_accuracy - b7_lora_2k_accuracy:+.4f}")

# %% Cell 87
b10_lora_2k_3shot_records_df = pd.DataFrame(
    b10_lora_2k_3shot_result["records"]
)

b10_lora_2k_3shot_records_df["has_boxed"] = (
    b10_lora_2k_3shot_records_df["model_response"]
    .str.contains(r"\\boxed\{", regex=True)
)

display(
    b10_lora_2k_3shot_records_df[
        ["index", "ground_truth", "extracted_answer", "correct", "has_boxed", "question", "model_response"]
    ].head(10)
)

print("Number with boxed answer:",
      int(b10_lora_2k_3shot_records_df["has_boxed"].sum()),
      "/",
      len(b10_lora_2k_3shot_records_df))

# %% Cell 88
clear_memory(
    "lora_2k_model",
    "lora_2k_tokenizer",
    "lora_1k_model",
    "lora_1k_tokenizer",
    "math_model",
    "math_tokenizer",
)

math_model, math_tokenizer = load_math_model(
    model_name=MATH_MODEL,
    lora_path=None,
)

# %% Cell 89
start = time.perf_counter()

b10_base_3shot_result = evaluate_gsm8k_progress(
    math_model,
    math_tokenizer,
    gsm8k_test_100,
    num_samples=100,
    system_prompt=SYSTEM_PROMPT,
    few_shot_messages=b10_few_shot_messages,
    batch_size=8,   # safer than 16 because the prompt is longer
    output_name="b10_base_3shot_100_gsm8k.json",
)

sync_cuda()
b10_base_3shot_runtime = time.perf_counter() - start

b10_base_3shot_accuracy = b10_base_3shot_result["accuracy"]
b10_base_3shot_correct = b10_base_3shot_result["correct"]
b10_base_3shot_total = b10_base_3shot_result["total"]

print(f"B.10 base model 3-shot accuracy: {b10_base_3shot_accuracy:.4f}")
print(f"Correct: {b10_base_3shot_correct}/{b10_base_3shot_total}")
print(f"Runtime: {b10_base_3shot_runtime:.2f} sec")
print(f"Delta vs base no-demo baseline: {b10_base_3shot_accuracy - b1_base_accuracy:+.4f}")

# %% Cell 90
b10_base_3shot_records_df = pd.DataFrame(
    b10_base_3shot_result["records"]
)

b10_base_3shot_records_df["has_boxed"] = (
    b10_base_3shot_records_df["model_response"]
    .str.contains(r"\\boxed\{", regex=True)
)

display(
    b10_base_3shot_records_df[
        ["index", "ground_truth", "extracted_answer", "correct", "has_boxed", "question", "model_response"]
    ].head(10)
)

print("Number with boxed answer:",
      int(b10_base_3shot_records_df["has_boxed"].sum()),
      "/",
      len(b10_base_3shot_records_df))

# %% Cell 91
b10_comparison_df = pd.DataFrame([
    {
        "model_setting": "Base model",
        "model": MATH_MODEL,
        "adapter": "none",
        "prompting": "no demonstrations",
        "k": 0,
        "correct": b1_base_correct,
        "total": b1_base_total,
        "accuracy": b1_base_accuracy,
        "baseline_accuracy": b1_base_accuracy,
        "delta_vs_no_demo": 0.0,
        "runtime_sec": b1_base_runtime,
    },
    {
        "model_setting": "Base model",
        "model": MATH_MODEL,
        "adapter": "none",
        "prompting": "3-shot",
        "k": 3,
        "correct": b10_base_3shot_correct,
        "total": b10_base_3shot_total,
        "accuracy": b10_base_3shot_accuracy,
        "baseline_accuracy": b1_base_accuracy,
        "delta_vs_no_demo": b10_base_3shot_accuracy - b1_base_accuracy,
        "runtime_sec": b10_base_3shot_runtime,
    },
    {
        "model_setting": "LoRA SFT 2k",
        "model": MATH_MODEL,
        "adapter": str(lora_2k_adapter_path),
        "prompting": "no demonstrations",
        "k": 0,
        "correct": b7_lora_2k_correct,
        "total": b7_lora_2k_total,
        "accuracy": b7_lora_2k_accuracy,
        "baseline_accuracy": b7_lora_2k_accuracy,
        "delta_vs_no_demo": 0.0,
        "runtime_sec": b7_eval_runtime,
    },
    {
        "model_setting": "LoRA SFT 2k",
        "model": MATH_MODEL,
        "adapter": str(lora_2k_adapter_path),
        "prompting": "3-shot",
        "k": 3,
        "correct": b10_lora_2k_3shot_correct,
        "total": b10_lora_2k_3shot_total,
        "accuracy": b10_lora_2k_3shot_accuracy,
        "baseline_accuracy": b7_lora_2k_accuracy,
        "delta_vs_no_demo": b10_lora_2k_3shot_accuracy - b7_lora_2k_accuracy,
        "runtime_sec": b10_lora_2k_3shot_runtime,
    },
])

display(b10_comparison_df)

b10_comparison_df.to_csv(
    OUTPUT_DIR / "b10_3shot_vs_no_demo_comparison.csv",
    index=False,
)

# %% Cell 92
b10_base_3shot_records_df.to_csv(
    OUTPUT_DIR / "b10_base_3shot_predictions.csv",
    index=False,
)

b10_lora_2k_3shot_records_df.to_csv(
    OUTPUT_DIR / "b10_lora_2k_3shot_predictions.csv",
    index=False,
)

# %% Cell 93
print("B.10 few-shot prompting summary")
print("-" * 80)

print(
    f"Base no-demo:   {b1_base_correct}/{b1_base_total}, "
    f"accuracy = {b1_base_accuracy:.4f}"
)
print(
    f"Base 3-shot:    {b10_base_3shot_correct}/{b10_base_3shot_total}, "
    f"accuracy = {b10_base_3shot_accuracy:.4f}, "
    f"delta = {b10_base_3shot_accuracy - b1_base_accuracy:+.4f}"
)

print()

print(
    f"2k SFT no-demo: {b7_lora_2k_correct}/{b7_lora_2k_total}, "
    f"accuracy = {b7_lora_2k_accuracy:.4f}"
)
print(
    f"2k SFT 3-shot:  {b10_lora_2k_3shot_correct}/{b10_lora_2k_3shot_total}, "
    f"accuracy = {b10_lora_2k_3shot_accuracy:.4f}, "
    f"delta = {b10_lora_2k_3shot_accuracy - b7_lora_2k_accuracy:+.4f}"
)

# %% Cell 95
b11_summary_df = pd.DataFrame([
    {
        "model_setting": "Base model",
        "no_demo_correct": b1_base_correct,
        "no_demo_total": b1_base_total,
        "no_demo_accuracy": b1_base_accuracy,
        "three_shot_correct": b10_base_3shot_correct,
        "three_shot_total": b10_base_3shot_total,
        "three_shot_accuracy": b10_base_3shot_accuracy,
        "absolute_gain": b10_base_3shot_accuracy - b1_base_accuracy,
        "relative_gain_percent": 100 * (b10_base_3shot_accuracy - b1_base_accuracy) / b1_base_accuracy,
    },
    {
        "model_setting": "LoRA SFT 2k",
        "no_demo_correct": b7_lora_2k_correct,
        "no_demo_total": b7_lora_2k_total,
        "no_demo_accuracy": b7_lora_2k_accuracy,
        "three_shot_correct": b10_lora_2k_3shot_correct,
        "three_shot_total": b10_lora_2k_3shot_total,
        "three_shot_accuracy": b10_lora_2k_3shot_accuracy,
        "absolute_gain": b10_lora_2k_3shot_accuracy - b7_lora_2k_accuracy,
        "relative_gain_percent": 100 * (b10_lora_2k_3shot_accuracy - b7_lora_2k_accuracy) / b7_lora_2k_accuracy,
    },
])

display(b11_summary_df)

b11_summary_df.to_csv(
    OUTPUT_DIR / "b11_few_shot_effect_summary.csv",
    index=False,
)

# %% Cell 96
best_benefit_row = b11_summary_df.sort_values(
    "absolute_gain",
    ascending=False,
).iloc[0]

print("Model benefiting most from 3-shot prompting:")
print(f"{best_benefit_row['model_setting']}")
print(f"Absolute gain: {best_benefit_row['absolute_gain']:+.4f}")
print(f"Relative gain: {best_benefit_row['relative_gain_percent']:+.2f}%")

# %% Cell 97
b1_records_df = pd.DataFrame(b1_base_result["records"]).copy()
b10_base_3shot_records_df = pd.DataFrame(b10_base_3shot_result["records"]).copy()

b11_base_compare_df = b1_records_df[
    ["index", "question", "ground_truth", "extracted_answer", "correct", "model_response"]
].rename(columns={
    "extracted_answer": "no_demo_answer",
    "correct": "no_demo_correct",
    "model_response": "no_demo_response",
}).merge(
    b10_base_3shot_records_df[
        ["index", "extracted_answer", "correct", "model_response"]
    ].rename(columns={
        "extracted_answer": "three_shot_answer",
        "correct": "three_shot_correct",
        "model_response": "three_shot_response",
    }),
    on="index",
    how="inner",
)

b11_base_compare_df["effect"] = np.select(
    [
        (~b11_base_compare_df["no_demo_correct"]) & (b11_base_compare_df["three_shot_correct"]),
        (b11_base_compare_df["no_demo_correct"]) & (~b11_base_compare_df["three_shot_correct"]),
        (b11_base_compare_df["no_demo_correct"]) & (b11_base_compare_df["three_shot_correct"]),
        (~b11_base_compare_df["no_demo_correct"]) & (~b11_base_compare_df["three_shot_correct"]),
    ],
    [
        "fixed_by_3shot",
        "broken_by_3shot",
        "correct_both",
        "wrong_both",
    ],
    default="unknown",
)

b11_base_effect_counts = (
    b11_base_compare_df["effect"]
    .value_counts()
    .rename_axis("effect")
    .reset_index(name="count")
)

b11_base_effect_counts["percent"] = 100 * b11_base_effect_counts["count"] / len(b11_base_compare_df)

display(b11_base_effect_counts)

b11_base_compare_df.to_csv(
    OUTPUT_DIR / "b11_base_no_demo_vs_3shot_predictions.csv",
    index=False,
)

b11_base_effect_counts.to_csv(
    OUTPUT_DIR / "b11_base_3shot_effect_counts.csv",
    index=False,
)

# %% Cell 98
b7_lora_2k_records_df = pd.DataFrame(b7_lora_2k_result["records"]).copy()
b10_lora_2k_3shot_records_df = pd.DataFrame(b10_lora_2k_3shot_result["records"]).copy()

b11_sft_compare_df = b7_lora_2k_records_df[
    ["index", "question", "ground_truth", "extracted_answer", "correct", "model_response"]
].rename(columns={
    "extracted_answer": "no_demo_answer",
    "correct": "no_demo_correct",
    "model_response": "no_demo_response",
}).merge(
    b10_lora_2k_3shot_records_df[
        ["index", "extracted_answer", "correct", "model_response"]
    ].rename(columns={
        "extracted_answer": "three_shot_answer",
        "correct": "three_shot_correct",
        "model_response": "three_shot_response",
    }),
    on="index",
    how="inner",
)

b11_sft_compare_df["effect"] = np.select(
    [
        (~b11_sft_compare_df["no_demo_correct"]) & (b11_sft_compare_df["three_shot_correct"]),
        (b11_sft_compare_df["no_demo_correct"]) & (~b11_sft_compare_df["three_shot_correct"]),
        (b11_sft_compare_df["no_demo_correct"]) & (b11_sft_compare_df["three_shot_correct"]),
        (~b11_sft_compare_df["no_demo_correct"]) & (~b11_sft_compare_df["three_shot_correct"]),
    ],
    [
        "fixed_by_3shot",
        "broken_by_3shot",
        "correct_both",
        "wrong_both",
    ],
    default="unknown",
)

b11_sft_effect_counts = (
    b11_sft_compare_df["effect"]
    .value_counts()
    .rename_axis("effect")
    .reset_index(name="count")
)

b11_sft_effect_counts["percent"] = 100 * b11_sft_effect_counts["count"] / len(b11_sft_compare_df)

display(b11_sft_effect_counts)

b11_sft_compare_df.to_csv(
    OUTPUT_DIR / "b11_sft_2k_no_demo_vs_3shot_predictions.csv",
    index=False,
)

b11_sft_effect_counts.to_csv(
    OUTPUT_DIR / "b11_sft_2k_3shot_effect_counts.csv",
    index=False,
)

# %% Cell 99
def effect_count(df, effect_name):
    return int((df["effect"] == effect_name).sum())

b11_effect_breakdown_df = pd.DataFrame([
    {
        "model_setting": "Base model",
        "correct_both": effect_count(b11_base_compare_df, "correct_both"),
        "fixed_by_3shot": effect_count(b11_base_compare_df, "fixed_by_3shot"),
        "broken_by_3shot": effect_count(b11_base_compare_df, "broken_by_3shot"),
        "wrong_both": effect_count(b11_base_compare_df, "wrong_both"),
        "net_gain": effect_count(b11_base_compare_df, "fixed_by_3shot") - effect_count(b11_base_compare_df, "broken_by_3shot"),
    },
    {
        "model_setting": "LoRA SFT 2k",
        "correct_both": effect_count(b11_sft_compare_df, "correct_both"),
        "fixed_by_3shot": effect_count(b11_sft_compare_df, "fixed_by_3shot"),
        "broken_by_3shot": effect_count(b11_sft_compare_df, "broken_by_3shot"),
        "wrong_both": effect_count(b11_sft_compare_df, "wrong_both"),
        "net_gain": effect_count(b11_sft_compare_df, "fixed_by_3shot") - effect_count(b11_sft_compare_df, "broken_by_3shot"),
    },
])

display(b11_effect_breakdown_df)

b11_effect_breakdown_df.to_csv(
    OUTPUT_DIR / "b11_fixed_broken_effect_breakdown.csv",
    index=False,
)

# %% Cell 100
def print_b11_case(row, model_name):
    print("=" * 120)
    print(f"Model: {model_name}")
    print(f"Index: {row['index']}")
    print(f"Ground truth: {row['ground_truth']}")
    print(f"Effect: {row['effect']}")
    print("-" * 120)
    print("Question:")
    print(row["question"])
    print("-" * 120)
    print("No-demo answer:", row["no_demo_answer"])
    print("No-demo correct:", row["no_demo_correct"])
    print("No-demo response:")
    print(row["no_demo_response"])
    print("-" * 120)
    print("3-shot answer:", row["three_shot_answer"])
    print("3-shot correct:", row["three_shot_correct"])
    print("3-shot response:")
    print(row["three_shot_response"])
    print("=" * 120)
    print()

# %% Cell 101
print("Base model: examples fixed by 3-shot")
for _, row in b11_base_compare_df[b11_base_compare_df["effect"] == "fixed_by_3shot"].head(3).iterrows():
    print_b11_case(row, "Base model")

print("Base model: examples broken by 3-shot")
for _, row in b11_base_compare_df[b11_base_compare_df["effect"] == "broken_by_3shot"].head(3).iterrows():
    print_b11_case(row, "Base model")

# %% Cell 102
print("LoRA SFT 2k: examples fixed by 3-shot")
for _, row in b11_sft_compare_df[b11_sft_compare_df["effect"] == "fixed_by_3shot"].head(3).iterrows():
    print_b11_case(row, "LoRA SFT 2k")

print("LoRA SFT 2k: examples broken by 3-shot")
for _, row in b11_sft_compare_df[b11_sft_compare_df["effect"] == "broken_by_3shot"].head(3).iterrows():
    print_b11_case(row, "LoRA SFT 2k")

# %% Cell 103
def boxed_rate(records_df):
    tmp = records_df.copy()
    tmp["has_boxed"] = tmp["model_response"].str.contains(r"\\boxed\{", regex=True)
    return int(tmp["has_boxed"].sum()), len(tmp), float(tmp["has_boxed"].mean())

base_no_boxed, base_no_total, base_no_boxed_rate = boxed_rate(b1_records_df)
base_3_boxed, base_3_total, base_3_boxed_rate = boxed_rate(b10_base_3shot_records_df)

sft_no_boxed, sft_no_total, sft_no_boxed_rate = boxed_rate(b7_lora_2k_records_df)
sft_3_boxed, sft_3_total, sft_3_boxed_rate = boxed_rate(b10_lora_2k_3shot_records_df)

b11_formatting_df = pd.DataFrame([
    {
        "model_setting": "Base model",
        "prompting": "no demonstrations",
        "boxed_count": base_no_boxed,
        "total": base_no_total,
        "boxed_rate": base_no_boxed_rate,
    },
    {
        "model_setting": "Base model",
        "prompting": "3-shot",
        "boxed_count": base_3_boxed,
        "total": base_3_total,
        "boxed_rate": base_3_boxed_rate,
    },
    {
        "model_setting": "LoRA SFT 2k",
        "prompting": "no demonstrations",
        "boxed_count": sft_no_boxed,
        "total": sft_no_total,
        "boxed_rate": sft_no_boxed_rate,
    },
    {
        "model_setting": "LoRA SFT 2k",
        "prompting": "3-shot",
        "boxed_count": sft_3_boxed,
        "total": sft_3_total,
        "boxed_rate": sft_3_boxed_rate,
    },
])

display(b11_formatting_df)

b11_formatting_df.to_csv(
    OUTPUT_DIR / "b11_boxed_answer_formatting_rates.csv",
    index=False,
)

# %% Cell 104
print("B.11 Few-shot prompting effect")
print("-" * 80)

for _, row in b11_summary_df.iterrows():
    print(f"{row['model_setting']}:")
    print(f"  No-demo accuracy:  {row['no_demo_accuracy']:.4f} ({row['no_demo_correct']}/{row['no_demo_total']})")
    print(f"  3-shot accuracy:   {row['three_shot_accuracy']:.4f} ({row['three_shot_correct']}/{row['three_shot_total']})")
    print(f"  Absolute gain:     {row['absolute_gain']:+.4f}")
    print(f"  Relative gain:     {row['relative_gain_percent']:+.2f}%")
    print()

print("Fixed/broken breakdown:")
display(b11_effect_breakdown_df)

print()
print("Formatting rates:")
display(b11_formatting_df)

print()
print(
    f"Model benefiting most by absolute accuracy gain: "
    f"{best_benefit_row['model_setting']} "
    f"({best_benefit_row['absolute_gain']:+.4f})"
)

# %% Cell 106
clear_memory(
    "lora_1k_model",
    "lora_1k_tokenizer",
    "lora_2k_model",
    "lora_2k_tokenizer",
    "math_model",
    "math_tokenizer",
)

# %% Cell 107

math_model, math_tokenizer = load_math_model(
    model_name=MATH_MODEL,
    lora_path=None,
)

print("Loaded base model for B.12:", MATH_MODEL)

# %% Cell 108
PROMPT_0 = SYSTEM_PROMPT

PROMPT_1 = """
Solve the math word problem. Be concise.
Use only the calculations needed to get the answer.

Final answer rule:
The last line of your response must be exactly:
\\boxed{ANSWER}
""".strip()

PROMPT_2 = """
Solve the math word problem using this structure:
1. Briefly identify what the question is asking for.
2. List the necessary quantities from the problem.
3. Write the calculation step by step.
4. Give only the final numeric answer in the required format.

Do not invent extra assumptions. Use the numbers exactly as stated in the problem.

Final answer rule:
The last line of your response must be exactly:
\\boxed{ANSWER}
""".strip()

PROMPT_3 = """
You are a careful math checker.
First write the key equations, then compute the answer, then check that the result answers the question.
Be especially careful with units, totals, twice/as-many relationships, percentage increases, and remaining quantities.
Give only the final numeric answer in the required format.

Final answer rule:
The last line of your response must be exactly:
\\boxed{ANSWER}
""".strip()

# %% Cell 109
def evaluate_one_prompt(model, tokenizer, prompt, name="prompt", n=50):
    result = evaluate_gsm8k(
        model,
        tokenizer,
        gsm8k["test"].select(range(n)),
        num_samples=n,
        system_prompt=prompt,
        batch_size=16,
    )
    return {"prompt_name": name, "accuracy": result["accuracy"], "correct": result["correct"], "total": result["total"]}, result

# Baseline bonus run using the old prompt. After this, run your own prompt variants.
if "math_model" not in globals() or "math_tokenizer" not in globals():
    math_model, math_tokenizer = load_math_model()

baseline_row, baseline_result = evaluate_one_prompt(math_model, math_tokenizer, PROMPT_0, name="old_prompt")
baseline_row

# %% Cell 110
b12_prompt_specs = [
    ("short_direct", PROMPT_1),
    ("step_by_step", PROMPT_2),
    ("equations_and_check", PROMPT_3),
]

b12_rows = [baseline_row]
b12_results = {
    "old_prompt": baseline_result,
}

for prompt_name, prompt in b12_prompt_specs:
    print("=" * 100)
    print(f"Evaluating B.12 prompt: {prompt_name}")
    print("=" * 100)

    start = time.perf_counter()
    row, result = evaluate_one_prompt(
        math_model,
        math_tokenizer,
        prompt,
        name=prompt_name,
        n=50,
    )
    sync_cuda()
    runtime_sec = time.perf_counter() - start

    row["runtime_sec"] = runtime_sec
    b12_rows.append(row)
    b12_results[prompt_name] = result

    print(row)
    print(f"Runtime: {runtime_sec:.2f} sec")

# %% Cell 111
b12_summary_df = pd.DataFrame(b12_rows)

if "runtime_sec" not in b12_summary_df.columns:
    b12_summary_df["runtime_sec"] = np.nan

b12_summary_df = b12_summary_df[
    ["prompt_name", "correct", "total", "accuracy", "runtime_sec"]
]

display(b12_summary_df)

b12_summary_df.to_csv(
    OUTPUT_DIR / "b12_prompt_playground_summary.csv",
    index=False,
)

# %% Cell 112
b12_prompt_text_df = pd.DataFrame([
    {
        "prompt_name": "old_prompt",
        "prompt_text": PROMPT_0,
    },
    {
        "prompt_name": "short_direct",
        "prompt_text": PROMPT_1,
    },
    {
        "prompt_name": "step_by_step",
        "prompt_text": PROMPT_2,
    },
    {
        "prompt_name": "equations_and_check",
        "prompt_text": PROMPT_3,
    },
])

display(b12_prompt_text_df)

b12_prompt_text_df.to_csv(
    OUTPUT_DIR / "b12_prompt_texts.csv",
    index=False,
)

# %% Cell 113
b12_format_rows = []

for prompt_name, result in b12_results.items():
    records_df = pd.DataFrame(result["records"]).copy()

    records_df["prompt_name"] = prompt_name
    records_df["has_boxed"] = records_df["model_response"].str.contains(
        r"\\boxed\{",
        regex=True,
    )
    records_df["response_chars"] = records_df["model_response"].astype(str).str.len()
    records_df["last_line"] = records_df["model_response"].apply(
        lambda x: str(x).strip().splitlines()[-1] if str(x).strip() else ""
    )

    records_df.to_csv(
        OUTPUT_DIR / f"b12_{prompt_name}_records.csv",
        index=False,
    )

    b12_format_rows.append({
        "prompt_name": prompt_name,
        "boxed_count": int(records_df["has_boxed"].sum()),
        "total": len(records_df),
        "boxed_rate": float(records_df["has_boxed"].mean()),
        "avg_response_chars": float(records_df["response_chars"].mean()),
        "median_response_chars": float(records_df["response_chars"].median()),
    })

b12_format_df = pd.DataFrame(b12_format_rows)

display(b12_format_df)

b12_format_df.to_csv(
    OUTPUT_DIR / "b12_prompt_formatting_diagnostics.csv",
    index=False,
)

# %% Cell 114
old_records_df = pd.DataFrame(b12_results["old_prompt"]["records"])[
    ["index", "question", "ground_truth", "extracted_answer", "correct", "model_response"]
].rename(columns={
    "extracted_answer": "old_extracted_answer",
    "correct": "old_correct",
    "model_response": "old_response",
})

b12_effect_rows = []

for prompt_name in ["short_direct", "step_by_step", "equations_and_check"]:
    variant_records_df = pd.DataFrame(b12_results[prompt_name]["records"])[
        ["index", "extracted_answer", "correct", "model_response"]
    ].rename(columns={
        "extracted_answer": "variant_extracted_answer",
        "correct": "variant_correct",
        "model_response": "variant_response",
    })

    compare_df = old_records_df.merge(
        variant_records_df,
        on="index",
        how="inner",
    )

    compare_df["prompt_name"] = prompt_name

    compare_df["effect_vs_old"] = np.select(
        [
            (~compare_df["old_correct"]) & (compare_df["variant_correct"]),
            (compare_df["old_correct"]) & (~compare_df["variant_correct"]),
            (compare_df["old_correct"]) & (compare_df["variant_correct"]),
            (~compare_df["old_correct"]) & (~compare_df["variant_correct"]),
        ],
        [
            "fixed_by_variant",
            "broken_by_variant",
            "correct_both",
            "wrong_both",
        ],
        default="unknown",
    )

    compare_df.to_csv(
        OUTPUT_DIR / f"b12_old_vs_{prompt_name}_comparison.csv",
        index=False,
    )

    b12_effect_rows.append({
        "prompt_name": prompt_name,
        "fixed_by_variant": int((compare_df["effect_vs_old"] == "fixed_by_variant").sum()),
        "broken_by_variant": int((compare_df["effect_vs_old"] == "broken_by_variant").sum()),
        "correct_both": int((compare_df["effect_vs_old"] == "correct_both").sum()),
        "wrong_both": int((compare_df["effect_vs_old"] == "wrong_both").sum()),
        "net_gain_vs_old": int((compare_df["effect_vs_old"] == "fixed_by_variant").sum())
                         - int((compare_df["effect_vs_old"] == "broken_by_variant").sum()),
    })

b12_effect_df = pd.DataFrame(b12_effect_rows)

display(b12_effect_df)

b12_effect_df.to_csv(
    OUTPUT_DIR / "b12_effect_vs_old_prompt.csv",
    index=False,
)

# %% Cell 115
print("B.12 Prompt Playground Summary")
print("-" * 80)

display(b12_summary_df)

print("\nFormatting diagnostics:")
display(b12_format_df)

print("\nEffect compared with old prompt:")
display(b12_effect_df)

best_b12_row = b12_summary_df.sort_values(
    ["accuracy", "correct"],
    ascending=False,
).iloc[0]

print()
print(f"Best B.12 prompt: {best_b12_row['prompt_name']}")
print(f"Accuracy: {best_b12_row['accuracy']:.4f}")
print(f"Correct: {int(best_b12_row['correct'])}/{int(best_b12_row['total'])}")

