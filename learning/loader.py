from tiktoken import get_encoding as tokenizer

"""Data sampling with a sliding window"""

# testing the tokenizer
with open("data/the-verdict.txt", "r") as file:
    text = file.read()

encoded_text = tokenizer("o200k_base").encode(text)
print(len(encoded_text))

decoded_text = tokenizer("o200k_base").decode(encoded_text)
print(len(decoded_text))

enc_sample = encoded_text[50:]

context_size = 4

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    target = enc_sample[i]
    print("-" * 100)
    print(context)
    print("-->", target)
    print(
        f"{tokenizer('o200k_base').decode(context)} --> {tokenizer('o200k_base').decode([target])}"
    )
    print("-" * 100)
