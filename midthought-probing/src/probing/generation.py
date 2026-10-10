"""Prompt construction and greedy reasoning generation."""

import re


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem carefully. Think step by step and put the final "
        "numeric answer in \\boxed{...}.\n\nProblem: "
        f"{question}\n\nAnswer:"
    )


def extract_final_answer(response: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", response)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def generate_greedy(model, tokenizer, question: str, max_new_tokens: int = 256):
    import torch

    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "response": response,
        "final_answer": extract_final_answer(response),
        "token_ids": generated.detach().cpu().tolist(),
    }
