import os
import re
from urllib import request

URL = '''https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt'''
file_path = 'data/the-verdict.txt'
request.urlretrieve(URL, file_path)

with open(file_path,'r', encoding='utf-8') as file:
    raw_text = file.read()
print(f'Number of characters in text: {len(raw_text)}')
print(f'First 1000 characters: {raw_text[:1000]}')

