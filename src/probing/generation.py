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


def find_reasoning_boundary(response: str) -> int:
    """Return the character index marking the start of the final answer.

    All characters before this index are pre-answer reasoning.
    """
    if not response:
        return 0

    # Look for boxed answer patterns, including introductory phrases
    boxed_pattern = (
        r"(?:(?:therefore|so|thus|hence|finally)[,\s]+)?(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)?\s*\\boxed\{"
    )
    match = re.search(boxed_pattern, response, re.IGNORECASE)
    if match:
        return match.start()

    idx = response.find(r"\boxed{")
    if idx != -1:
        preamble = re.search(
            r"(?:(?:therefore|so|thus|hence|finally)[,\s]+)?(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)?\s*$",
            response[:idx],
            re.IGNORECASE,
        )
        if preamble:
            return preamble.start()
        return idx

    # Look for GSM8K delimiter "####"
    hash_idx = response.find("####")
    if hash_idx != -1:
        return hash_idx

    # Look for "final answer is ..."
    ans_match = re.search(
        r"(?:(?:the\s+)?final\s+answer\s*(?:is|:|=)|####)\s*", response, re.IGNORECASE
    )
    if ans_match:
        return ans_match.start()

    last_newline = response.rfind("\n")
    if last_newline > 0:
        return last_newline

    return len(response)


def split_reasoning_prefix(response: str) -> tuple[str, str]:
    """Split response into (reasoning_prefix, answer_suffix)."""
    boundary = find_reasoning_boundary(response)
    return response[:boundary].rstrip(), response[boundary:].lstrip()


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
    final_answer = extract_final_answer(response)

    # Strictly isolate reasoning prefix prior to the answer
    reasoning_text, answer_text = split_reasoning_prefix(response)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_prompt_reasoning = tokenizer(
        prompt + (" " + reasoning_text if reasoning_text else ""),
        add_special_tokens=False,
    )["input_ids"]
    reasoning_token_ids = full_prompt_reasoning[len(prompt_ids) :]

    return {
        "prompt": prompt,
        "response": response,
        "final_answer": final_answer,
        "token_ids": generated.detach().cpu().tolist(),
        "reasoning_text": reasoning_text,
        "reasoning_token_ids": reasoning_token_ids,
        "answer_text": answer_text,
        "boundary_char": find_reasoning_boundary(response),
    }
