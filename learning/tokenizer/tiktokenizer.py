import tiktoken

tokenizer = tiktoken.get_encoding("o200k_base")

text = '''The quick brown fox jumps over the lazy dog.'''
tokens = tokenizer.encode(text)
print(tokens)

decoded = tokenizer.decode(tokens)
print(decoded)



text = '''The quick brown fox jumps over the lazy dog. <|endoftext|>'''
tokens = tokenizer.encode(text, allowed_special='all')
print(tokens)

decoded = tokenizer.decode(tokens)
print(decoded)







