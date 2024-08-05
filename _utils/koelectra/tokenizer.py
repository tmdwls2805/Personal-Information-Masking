from transformers import ElectraTokenizer

def get_tokenizer():
    return ElectraTokenizer.from_pretrained("monologg/koelectra-base-v2-discriminator")