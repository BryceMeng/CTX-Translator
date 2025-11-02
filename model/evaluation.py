
from datetime import timedelta
import gc
import json
import math
import os
import random
import time
import colorama
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedModel, PreTrainedTokenizer

from train_lora import load_model, build_gemma_inference_prompt

model_id = "ModelSpace/GemmaX2-28-2B-v0.1"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")
DATA_FILE_PATH = os.path.join(DATA_PATH, "final_data.json")

BATCH_SIZE = 1

FULL_PARAMETER = False
IS_ONLY_QKV = True
IS_ONLY_QV = True

tag = "default"
if FULL_PARAMETER:
    tag = "Full"
elif IS_ONLY_QKV:
    if IS_ONLY_QV:
        tag = "LoRA_QV"
    else:
        tag = "LoRA_QKV"
else:
    tag = "LoRA_Linear"

MODEL_FILE_PATH = os.path.join(BASE_DIR, "archive_4000_v2", f"weights_{tag}.pt")

def evaluate_model_on_data(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_file: str,
    random_seed: int = 42
) -> float:
    """
    Evaluate a given model on a prompt-response dataset and return average loss.

    Args:
        model (PreTrainedModel): A Hugging Face CausalLM model (already loaded and on device).
        tokenizer (PreTrainedTokenizer): Corresponding tokenizer.
        eval_file (str): Path to JSONL file with {"prompt": ..., "response": ...}.
        max_samples (int): Optionally limit number of examples for quick eval.

    Returns:
        float: Average loss over dataset.
    """
    device = next(model.parameters()).device

    # Load evaluation data
    samples = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)

    random.seed(random_seed) # got the same data everytime
    samples = random.sample(samples, 500)

    losses = []

    for sample in tqdm(samples, desc="Evaluating"):
        prompt = sample["prompt"].strip()
        response = sample["response"].strip()

        bos = tokenizer.bos_token
        eos = tokenizer.eos_token

        full_text = f"[INST] {prompt} [/INST] {response}{eos}"
        prompt_prefix = f"[INST] {prompt} [/INST]"

        # Tokenize
        input_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        # Mask prompt region in labels
        prefix_len = len(tokenizer(prompt_prefix, add_special_tokens=True)["input_ids"])
        labels[0][:prefix_len] = -100
        labels[attention_mask == 0] = -100

        # print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
        
        # tokens = []
        # for tid in labels[0].tolist():
        #     if tid == -100:
        #         tokens.append("█")  # 占位符
        #     else:
        #         tokens.append(tokenizer.decode([tid], skip_special_tokens=False))
        # print("".join(tokens))
        # print("-"*20)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.item()
            losses.append(loss)

    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    print(f"Average loss on {len(losses)} samples {tag}: {avg_loss:.4f}")
    return avg_loss

def evaluate_data_set():
    tokenizer, model = load_model(FULL_PARAMETER, IS_ONLY_QKV, IS_ONLY_QV, tag)
    state_dict = torch.load(MODEL_FILE_PATH, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    evaluate_model_on_data(model, tokenizer, DATA_FILE_PATH, random_seed=9)

    return model, tokenizer

def infer_single_prompt(prompt):
    tokenizer, model = load_model(FULL_PARAMETER, IS_ONLY_QKV, IS_ONLY_QV, tag)
    state_dict = torch.load(MODEL_FILE_PATH, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    prompt = build_gemma_inference_prompt(prompt, tokenizer.bos_token, add_bos=False)

    print(f"src BOS:{tokenizer.bos_token} {tokenizer.bos_token_id} EOS:{tokenizer.eos_token} {tokenizer.eos_token_id}")

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=0.7, top_p=0.9, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=False))

    return model, tokenizer

if __name__ == "__main__":

    prompt = json.dumps({"context":"My wife often telephones me when I'm traveling in another country.","target":"telephones","idx":1})

    model, tokenizer = infer_single_prompt(prompt)
    # model, tokenizer = evaluate_data_set()

    del tokenizer
    del model
    gc.collect()
    torch.mps.empty_cache()