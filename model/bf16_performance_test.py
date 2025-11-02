import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "ModelSpace/GemmaX2-28-2B-v0.1"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

long_text = (
    "Translate this from English to Chinese:\nEnglish: "
    "Over the past few decades, the field of artificial intelligence has undergone a remarkable transformation, "
    "evolving from symbolic systems and handcrafted rules to powerful data-driven deep learning models. "
    "This evolution has been accelerated by exponential increases in computational power, the availability of massive datasets, "
    "and significant advancements in neural network architectures. "
    "In particular, large language models such as GPT, PaLM, and Gemini have demonstrated unprecedented capabilities in tasks such as machine translation, "
    "summarization, question answering, and even code generation. "
    "These models are capable of understanding complex queries, generating coherent responses, and adapting to diverse linguistic and cultural contexts, "
    "making them suitable for deployment in customer service, education, content creation, and more. "
    "Despite their impressive performance, however, there are growing concerns about the ethical and societal implications of deploying such systems at scale. "
    "Issues such as data privacy, bias in training data, hallucination of facts, and the concentration of AI capabilities within a few corporations "
    "have raised important questions about transparency, accountability, and inclusiveness. "
    "Moreover, the energy consumption and carbon footprint associated with training and deploying large models "
    "have drawn criticism from environmental and sustainability advocates. "
    "As a result, researchers and practitioners are actively exploring more efficient training techniques, model distillation methods, and alignment strategies "
    "to ensure that future models are not only powerful but also responsible. "
    "Furthermore, governments and regulatory bodies around the world are beginning to formulate policies and standards aimed at guiding the development and "
    "deployment of AI technologies in a manner that aligns with democratic values and protects human rights. "
    "Ultimately, the challenge for the next generation of AI researchers is to create systems that are not only intelligent but also aligned with human intentions, "
    "robust against adversarial attacks, and beneficial to society as a whole.\nChinese:"
)

def run_inference(dtype, label=""):
    print(f"\n--- Running with {label} on {device} ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(long_text, return_tensors="pt").to(device)

    torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None

    start = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=512)
    end = time.time()

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Output:\n{result}...\n")
    print(f"Inference time: {end - start:.3f} seconds")

# Run FP32
run_inference(torch.float32, "float32")

# Run BF16
run_inference(torch.bfloat16, "bfloat16")
