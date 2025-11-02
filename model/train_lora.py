"""
TODO:
1. DoRA
2. data format -> {"class": "preposition", "zh": "像", "phrase": "treats me like","phrase_zh":"像对待我一样"}
"""
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
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

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

MODEL_FILE_PATH = os.path.join(BASE_DIR, f"weights_{tag}.pt")

def gen_tag():
    global tag, MODEL_FILE_PATH
    if FULL_PARAMETER:
        tag = "Full"
    elif IS_ONLY_QKV:
        if IS_ONLY_QV:
            tag = "LoRA_QV"
        else:
            tag = "LoRA_QKV"
    else:
        tag = "LoRA_Linear"

    MODEL_FILE_PATH = os.path.join(BASE_DIR, f"weights_{tag}.pt")
    print(f"tag:{tag} save model:{MODEL_FILE_PATH}")


# Core LoRA class: uses two low-rank matrices A and B to approximate updates to large weight matrices, enabling parameter-efficient fine-tuning
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.base = base_layer
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        device = base_layer.weight.device

        # LoRA matrix A: projects input into a low-rank subspace
        # A is the transform that maps the input into a smaller (rank-r) dimension
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features), device=device))
        # LoRA matrix B: maps low-rank representation back to output space
        # Together with A, forms a low-rank approximation to a full-rank update to the original weight matrix
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank), device=device))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):
        # LoRA forward logic: preserve original inference path and add a trainable low-rank update
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @property
    def weight(self):
        return self.base.weight

# Injects LoRA into a model by replacing selected Linear layers with LoRALinear layers
# This allows fine-tuning only low-rank matrices while keeping the base model frozen

def inject_lora(model, r=16, alpha=32, only_qkv=IS_ONLY_QKV, only_qv=IS_ONLY_QV):
    if only_qv:
        target_modules = ["q_proj", "v_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj"]
    count = 0
    for name, module in model.named_modules():
        indent = "\t" * name.count(".")
        is_replace = isinstance(module, torch.nn.Linear)
        color = ""
        if isinstance(module, torch.nn.Linear):
            if any(n in name for n in target_modules):
                color = colorama.Fore.RED
            else:
                color = colorama.Fore.YELLOW
        # print(f"Lora:{indent}{color}{name},{type(module)}{colorama.Style.RESET_ALL}")

        if only_qkv:
            condition = any(n in name for n in target_modules) and isinstance(module, nn.Linear)
        else:
            condition = isinstance(module, nn.Linear)

        if condition:
            parent = model
            *path, attr = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, attr, LoRALinear(module, rank=r, alpha=alpha))
            count += 1
    print(f"Injected LoRA into {count} layers.")

def save_lora_weights(model, filepath):
    if FULL_PARAMETER:
        torch.save(model.state_dict(), filepath)
        print(f"Full weights saved to {filepath}")
    else:
        lora_state_dict = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
        torch.save(lora_state_dict, filepath)
        print(f"LoRA weights saved to {filepath}")

def load_model(is_full=FULL_PARAMETER, is_lora_only_qkv=IS_ONLY_QKV, is_loar_only_qv=IS_ONLY_QV, p_tag=tag):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", torch_dtype=torch.bfloat16).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters of the model: {total_params:,}")


    if is_full:
        for param in model.parameters():
            param.requires_grad = True
    else:

        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters before injecting LoRA: {total_params:,}")

        inject_lora(model,only_qkv=is_lora_only_qkv,only_qv=is_loar_only_qv)


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,} on {device} tag:{p_tag}")
    return tokenizer, model

class TranslationDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["prompt"] + " " + ex["response"]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

def train(model, dataloader, epoch_count):

    one_epoch_len = len(dataloader)
    time_left_secs = 0

    # lora_params = [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    optimizer = AdamW(model.parameters(), lr=5e-4)
    model.train()
    for epoch in range(epoch_count):
        epoch_start = time.time()
        ten_batch_run_time = 0
        for idx, batch in enumerate(dataloader):
            batch_start = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            if idx == 0:
                ten_batch_run_time = time.time()
                print(f"Model: {next(model.parameters()).dtype}",f"input_ids dtype: {batch['input_ids'].dtype}", f"logits dtype: {outputs.logits.dtype}")

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            one_batch_last_time = time.time()-batch_start

            if (idx+1)%10 == 0:
                ten_batch_last = time.time() - ten_batch_run_time
                ten_batch_run_time = time.time()
                time_left_secs = int(((one_epoch_len - idx - 1) + (epoch_count-epoch-1)*one_epoch_len)/10 * ten_batch_last)
            else:
                if time_left_secs > 0:
                    time_left_secs -= int(one_batch_last_time)

            print(f"Epoch {epoch} {(idx+1)}/{one_epoch_len}, Loss: {loss.item():.4f} {round(one_batch_last_time,2)} secs {str(timedelta(seconds=time_left_secs)) if time_left_secs > 0 else 'unknown time'} left")

        one_epoch_last_time = time.time() - epoch_start
        print(f"Epoch {epoch}: {round(one_epoch_last_time,1)} secs {str(timedelta(seconds=one_epoch_last_time))}")

def infer(prompt):
    tokenizer, model = load_model(is_full=FULL_PARAMETER, is_lora_only_qkv=IS_ONLY_QKV, is_loar_only_qv=IS_ONLY_QV, p_tag=tag)
    state_dict = torch.load(MODEL_FILE_PATH, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    prompt = build_gemma_inference_prompt(prompt, tokenizer.bos_token, add_bos=False)

    print(f"src BOS:{tokenizer.bos_token} {tokenizer.bos_token_id} EOS:{tokenizer.eos_token} {tokenizer.eos_token_id}")

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample = True, temperature=0.7, top_p=0.9, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=False))

    return model, tokenizer

def build_gemma_inference_prompt(prompt: str, bos_token="<s>", add_bos=True):
    """
    Build inference prompt in Gemma format:
    <s>[INST] prompt [/INST]

    Args:
        prompt (str): The user's input prompt.
        bos_token (str): Beginning-of-sequence token (default: "<s>").
        add_bos (bool): Whether to prepend the BOS token.

    Returns:
        str: The formatted prompt for inference.
    """
    prompt = prompt.strip()
    text = f"[INST] {prompt} [/INST]"
    if add_bos and bos_token:
        text = bos_token + text
    return text


def build_gemma_prompt(prompt: str, response: str = "", bos_token="<s>", eos_token="</s>", add_bos=True, add_eos=True):
    """
    Build training prompt-response text in Gemma format:
    <s>[INST] prompt [/INST] response</s>

    Args:
        prompt (str): The user's input.
        response (str): The assistant's response.
        bos_token (str): Beginning-of-sequence token (default: "<s>").
        eos_token (str): End-of-sequence token (default: "</s>").
        add_bos (bool): Whether to prepend the BOS token.
        add_eos (bool): Whether to append the EOS token.

    Returns:
        str: The full training-formatted input string.
    """
    prompt = prompt.strip()
    response = response.strip()
    text = f"[INST] {prompt} [/INST] {response}"
    if add_bos and bos_token:
        text = bos_token + text
    if add_eos and eos_token:
        text = text + eos_token
    return text

class Gemma2ChatDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"src BOS:{tokenizer.bos_token} {tokenizer.bos_token_id} EOS:{tokenizer.eos_token} {tokenizer.eos_token_id}")
        # tokenizer.bos_token is super slow, so use the cache vars: bos and eos
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token

        with open(file_path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                count+=1
                item = json.loads(line)
                prompt = item["prompt"]
                response = item["response"]

                # Build full text including prompt and response
                full_text = build_gemma_prompt(prompt, response, eos_token=eos, bos_token=bos, add_bos=False, add_eos=True)

                # print(full_text)

                # Tokenize full text（prompt + response）
                tokenized = tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length"
                )

                input_ids = tokenized["input_ids"].squeeze(0)
                attention_mask = tokenized["attention_mask"].squeeze(0)
                labels = input_ids.clone()

                # find the start position of response
                prefix_text = build_gemma_prompt(prompt, response="", eos_token=eos, bos_token=bos, add_bos=False, add_eos=False)
                prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                prefix_len = len(prefix_ids)

                # mask prompt
                labels[:prefix_len] = -100
                labels[attention_mask == 0] = -100

                # print(tokenizer.decode(input_ids, skip_special_tokens=False))

                # tokens = []
                # for tid in labels.tolist():
                #     if tid == -100:
                #         tokens.append("█")  # 
                #     else:
                #         tokens.append(tokenizer.decode([tid], skip_special_tokens=False))

                # print("".join(tokens))

                # print("-"*20)

                self.data.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

        if len(self.data) > 4000:
            random.seed(9) # got the same data everytime
            self.data = random.sample(self.data, 4000)


    def _find_response_start(self, full_ids, resp_ids):
        for i in range(len(full_ids) - len(resp_ids)):
            if full_ids[i:i + len(resp_ids)] == resp_ids:
                return i
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_and_train():
    tokenizer, model = load_model(is_full=FULL_PARAMETER, is_lora_only_qkv=IS_ONLY_QKV, is_loar_only_qv=IS_ONLY_QV, p_tag=tag)

    # dataset = TranslationDataset(examples, tokenizer)
    print("loading samples ...")
    dataset = Gemma2ChatDataset(DATA_FILE_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"loading {len(dataset)} samples done.")


    train(model, dataloader, epoch_count=1)
    save_lora_weights(model, MODEL_FILE_PATH)

    del tokenizer
    del model
    gc.collect()
    torch.mps.empty_cache()

def reference():

    prompt = json.dumps({"context":"I need to hit the sack","target":"hit","idx":1})
    model, tokenizer = infer(prompt)

    del tokenizer
    del model
    gc.collect()
    torch.mps.empty_cache()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="fine tune type")

    parser.add_argument('--fttype', type=str, default="LoRA_QV", help='fine tune type')

    args = parser.parse_args()

    if args.fttype == "LoRA_QV":
        FULL_PARAMETER = False
        IS_ONLY_QKV = True
        IS_ONLY_QV = True
    elif args.fttype == "LoRA_QKV":
        FULL_PARAMETER = False
        IS_ONLY_QKV = True
        IS_ONLY_QV = False

    gen_tag()

    load_and_train()

    reference()
