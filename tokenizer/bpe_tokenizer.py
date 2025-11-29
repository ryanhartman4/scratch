import os
import regex as re
from typing import List, Dict, Tuple, Optional, Union

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.special_tokens = {}
        self._inverse_special_tokens = {}
        self.GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.GPT4_PATTERN)

    def train(self, text):
        
        # Pre-tokenize text into chunks
        chunks = self.compiled_pattern.findall(text)

        # Convert chunks to token lists
        token_lists = [list(chunk.encode('utf-8')) for chunk in chunks]

        self.vocab = {i: bytes([i]) for i in range(256)}
        num_merges = self.vocab_size - len(self.vocab)

        initial_tokens = sum(len(tl) for tl in token_lists)

        for i in range(num_merges):
            stats = self._get_stats(token_lists)
            if not stats:
                break

            top_pair = max(stats.items(), key=lambda x: x[1])[0]

            # create new token
            new_id = i + 256
            self.merges[new_id] = top_pair
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            token_lists = self._merge_all(token_lists, top_pair, new_id)

        final_tokens = sum(len(tl) for tl in token_lists)
        compression = 100 * (1 - final_tokens / initial_tokens)
        print()
        print(f"Training complete!")
        print(f"  Vocabulary: {len(self.vocab)} tokens")
        print(f"  Compression: {compression:.1f}%")
        return None

    def _get_stats(self, token_lists, counts: Optional[dict] = None):
        if counts is None:
            counts = {}
        
        for token_list in token_lists:
            for pair in zip(token_list, token_list[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        
        return counts
    
    def _merge_vocab(self, token_list, pair, new_id):
        new_list = []
        i = 0

        while i < len(token_list):
            if i < len(token_list) - 1 and tuple(token_list[i:i+2]) == pair:
                new_list.append(new_id)
                i += 2
            else:
                new_list.append(token_list[i])
                i += 1
        return new_list
    
    def _merge_all(self, token_lists, pair, new_id):
        return [self._merge_vocab(token_list, pair, new_id) for token_list in token_lists]

    def register_special_tokens(self, special_tokens: dict):
        self.special_tokens = special_tokens
        self._inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def add_special_tokens(self, tokens: List[str]) -> Dict[str, int]:
        """
        Add special tokens, auto-assigning IDs after current vocab.
        
        Args:
            tokens: List of special token strings
            
        Returns:
            Dict of added tokens with their IDs
        """
        start_id = max(
            len(self.vocab),
            max(self.special_tokens.values(), default=0) + 1
        )
        
        new_tokens = {}
        for i, token in enumerate(tokens):
            if token not in self.special_tokens:
                token_id = start_id + len(new_tokens)
                self.special_tokens[token] = token_id
                new_tokens[token] = token_id
        
        self._inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        return new_tokens   
    
    
    def encode(self, text):
        special = {k: v for k, v in self.special_tokens.items()}

        parts = self._split_with_special(text, special)

        all_tokens = []
        for part in parts:
            if isinstance(part, tuple):
                all_tokens.append(part[1])
            else:
                all_tokens.extend(self._encode_text(part))
        return all_tokens

    def _encode_text(self, text: str) -> List[int]:
        """Encode regular text (no special tokens)."""
        chunks = self.compiled_pattern.findall(text)
        all_tokens = []
        
        for chunk in chunks:
            tokens = list(chunk.encode('utf-8'))
            for new_id, pair in self.merges.items():
                tokens = self._merge_vocab(tokens, pair, new_id)
            all_tokens.extend(tokens)
        
        return all_tokens

    def _split_with_special(
        self,
        text: str,
        special: Dict[str, int]
    ) -> List[Union[str, Tuple[str, int]]]:
        """Split text on special tokens."""
        if not special:
            return [text] if text else []
        
        # Build pattern matching special tokens
        sorted_tokens = sorted(special.keys(), key=len, reverse=True)
        pattern = '(' + '|'.join(re.escape(t) for t in sorted_tokens) + ')'
        
        parts = re.split(pattern, text)
        
        result = []
        for part in parts:
            if not part:
                continue
            if part in special:
                result.append((part, special[part]))
            else:
                result.append(part)
        
        return result

    def decode(self, tokens: List[int], errors: str = 'replace') -> str:
            """
            Decode token IDs to text.
            
            Args:
                tokens: List of token IDs
                errors: How to handle invalid UTF-8 ('replace', 'ignore', 'strict')
                
            Returns:
                Decoded string
            """
            parts = []
            byte_buffer = b''
            
            for token in tokens:
                if token in self._inverse_special_tokens:
                    # Flush byte buffer
                    if byte_buffer:
                        parts.append(byte_buffer.decode('utf-8', errors=errors))
                        byte_buffer = b''
                    parts.append(self._inverse_special_tokens[token])
                elif token in self.vocab:
                    byte_buffer += self.vocab[token]
                else:
                    # Unknown token - skip or use replacement
                    if byte_buffer:
                        parts.append(byte_buffer.decode('utf-8', errors=errors))
                        byte_buffer = b''
                    parts.append(f"<unk:{token}>")
            
            if byte_buffer:
                parts.append(byte_buffer.decode('utf-8', errors=errors))
            
            return ''.join(parts)
    
    def decode_tokens(self, tokens: List[int]) -> List[str]:
        """Decode each token individually (for visualization)."""
        result = []
        for token in tokens:
            if token in self._inverse_special_tokens:
                result.append(self._inverse_special_tokens[token])
            elif token in self.vocab:
                try:
                    result.append(self.vocab[token].decode('utf-8'))
                except:
                    result.append(f"<0x{self.vocab[token].hex()}>")
            else:
                result.append(f"<unk:{token}>")
        return result

if __name__ == "__main__":
    print("=" * 80)
    print("COMPLETE BPE TOKENIZER CLASS")
    print("=" * 80)
    
    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=300)
    
    training_text = """
    The quick brown fox jumps over the lazy dog.
    The quick brown fox is quicker than the lazy dog.
    The lazy dog sleeps while the quick fox jumps.
    Foxes are quick. Dogs are lazy. The fox jumps quickly.
    Machine learning is transforming how we build software.
    Large language models use tokenization for text processing.
    """
    
    print("\n" + "-" * 40)
    print("TRAINING:")
    print("-" * 40)
    tokenizer.train(training_text)
    
    # Add special tokens
    print("\n" + "-" * 40)
    print("SPECIAL TOKENS:")
    print("-" * 40)
    added = tokenizer.add_special_tokens([
        "<|endoftext|>",
        "<|pad|>",
        "<|startoftext|>",
    ])
    print(f"Added: {added}")
    print(f"Tokenizer: {tokenizer}")
    
    # Test encoding/decoding
    print("\n" + "-" * 40)
    print("ENCODING/DECODING:")
    print("-" * 40)
    
    test_cases = [
        "The quick fox",
        "Hello world!",
        "<|startoftext|>The lazy dog<|endoftext|>",
        "Machine learning models",
    ]
    
    for text in test_cases:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        token_strs = tokenizer.decode_tokens(tokens)
        
        print(f"\n  Text:    {text!r}")
        print(f"  Tokens:  {tokens}")
        print(f"  Strings: {token_strs}")
        print(f"  Decoded: {decoded!r}")
        print(f"  Match:   {'✓' if decoded == text else '✗'}")
