import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model_id = "ModelSpace/GemmaX2-28-9B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

print(type(model))
print(model.__class__.__module__)

for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")

print("The model is created pytorch ?", isinstance(model, nn.Module))

print("Datatype:", model.dtype, next(model.parameters()).dtype)
print("Model on device:", model.device)

print("Parameters on device:", next(model.parameters()).device)

total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total / 1e6:.2f}M")

text = "Translate this from English to Chinese:\nEnglish:My wife often telephones me when I'm traveling in another country.\nChinese:"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
