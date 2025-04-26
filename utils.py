import torch

SENTENCE_MAX_LENGTH_DEFAULT = 30

def hf_text_embedding(sentence:str, hf_tokenizer, hf_model, max_length=SENTENCE_MAX_LENGTH_DEFAULT):
    inputs = hf_tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    with torch.no_grad():
        outputs = hf_model(**inputs)
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    return sentence_embedding.squeeze().numpy()