import torch

def jaccard_ngrams(a, b, n=2):
    #jaccard similarity over ngrams to ensure that the two summaries are sufficiently different
    def ngrams(s):
        tokens = [t for t in s.lower().split() if t.strip()]
        if len(tokens) < n:
            return set()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    A, B = ngrams(a), ngrams(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def make_prompt(source_text):
    #generates prompt to tell speaker model to summarize
    return ("Extract the key words from the sentence, exclude punctuation. \n"
            "It is more important to mention key words than to speak in complete sentences. \n\n"
            f"TEXT: {source_text}\n\n"
            "SUMMARY: \n"
    )

@torch.inference_mode()
def generate_summary(model, tokenizer, prompt, top_p, temperature, max_new_tokens, repetition_penalty, no_repeat_ngram_size, seed):
    torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens, #depends on epochs
        min_new_tokens=2,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    return decoded.strip()
