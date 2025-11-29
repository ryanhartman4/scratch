import os
import re

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size

    def train(self, text):
        # convert text to bytes



        self.vocab = vocab
        return vocab