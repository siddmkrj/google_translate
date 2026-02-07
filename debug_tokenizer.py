from tokenizers import Tokenizer
import os

try:
    path = "models/tokenizer/tokenizer.json"
    if not os.path.exists(path):
        print("File not found.")
    else:
        print(f"Loading {path}...")
        t = Tokenizer.from_file(path)
        print("Success! Vocab size:", t.get_vocab_size())
except Exception as e:
    print("Failed to load:")
    print(e)
