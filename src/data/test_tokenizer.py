from tokenizers import Tokenizer
import os
import sys
import tokenizers

def test_tokenizer():
    print("="*40)
    print("      Tokenizer Verification Script      ")
    print("="*40)
    
    print(f"Python Executable: {sys.executable}")
    print(f"Tokenizers Version: {tokenizers.__version__}")
    print("-" * 40)

    tokenizer_path = 'models/tokenizer/tokenizer.json'

    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer NOT found at {tokenizer_path}!")
        return

    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"✅ Tokenizer loaded successfully.")
        print(f"   Vocab Size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        print(f"❌ ERROR Loading Tokenizer: {e}")
        return

    print("-" * 40)

    # Test Cases
    test_cases = [
        ("English", "Hello, world! This is a test."),
        ("Bengali", "বাংলা ভাষা একটি ইন্দো-আর্য ভাষা।"),
        ("Mixed", "I love বাংলা language."),
        ("Rare/Unknown", "🎉")
    ]

    for name, text in test_cases:
        print(f"\n🔹 Testing: {name}")
        print(f"   Input: '{text}'")
        encoded = tokenizer.encode(text)
        print(f"   Tokens: {encoded.tokens}")
        print(f"   IDs:    {encoded.ids}")
        decoded = tokenizer.decode(encoded.ids)
        print(f"   Decoded: '{decoded}'")

    print("\n" + "="*40)
    
    # Check for UNK
    unk_id = tokenizer.token_to_id("[UNK]")
    if unk_id is not None:
        print(f"ℹ️  [UNK] Token ID: {unk_id}")
    else:
        print("⚠️  [UNK] token not found in vocab map.")

if __name__ == "__main__":
    test_tokenizer()
