from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
from datasets import Dataset
import os
import pickle
from model import Config
import sys
sys.stdout.reconfigure(encoding='utf-8')

# --------- Train Tokenizer ----------
def train_tokenizer(texts, vocab_file):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"])
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(vocab_file)
    return tokenizer

# --------- Load Tokenizer ----------
def load_tokenizer(vocab_file):
    return Tokenizer.from_file(vocab_file)

# --------- Test Round Trip ----------
def test_round_trip(tokenizer, text):
    ids = tokenizer.encode(text).ids
    decoded = tokenizer.decode(ids)
    return ids, decoded

# ===== Step 1: Read CSV =====
df = pd.read_csv("english_to_bangla.csv")  # Your CSV with columns 'en' and 'bn'
df = df.dropna(subset=["en", "bn"])
dataset = Dataset.from_pandas(df)
# dataset = Dataset.from_pandas(df)

# Train English and Bangla tokenizers
tokenizer_en = train_tokenizer(df["en"].tolist(), "tokenizer_en.json")
tokenizer_bn = train_tokenizer(df["bn"].tolist(), "tokenizer_bn.json")

# Load them later
tokenizer_en = load_tokenizer("tokenizer_en.json")
tokenizer_bn = load_tokenizer("tokenizer_bn.json")

# Test for English
ids_en, decoded_en = test_round_trip(tokenizer_en, df['en'][5])
print("EN ids:", ids_en)
print("EN decoded:", decoded_en)

# Test for Bangla
ids_bn, decoded_bn = test_round_trip(tokenizer_bn, df['bn'][5])
print("BN ids:", ids_bn)
print("BN decoded:", decoded_bn)

# ===== Step 3: Prepare Encoder/Decoder Inputs =====
def preprocess(examples, max_len=Config().block_size):
    encoder_input = []
    decoder_input = []
    decoder_output = []
    
    for en_sentence, bn_sentence in zip(examples["en"], examples["bn"]):
        if not en_sentence or not bn_sentence:
            continue  # skip empty
        # Encode English
        en_tokens = tokenizer_en.encode(en_sentence).ids
        
        # Encode Bangla with SOS/EOS
        bn_tokens = tokenizer_bn.encode(bn_sentence).ids
        bn_input_tokens = [tokenizer_bn.token_to_id("[SOS]")] + bn_tokens
        bn_output_tokens = bn_tokens + [tokenizer_bn.token_to_id("[EOS]")]

        if len(en_tokens) > max_len: 
            en_tokens = en_tokens[:max_len]
        else:
            en_tokens += [tokenizer_en.token_to_id("[PAD]")] * (max_len - len(en_tokens))
        if len(bn_input_tokens) > max_len:
            bn_input_tokens = bn_input_tokens[:max_len]
        else:
            bn_input_tokens += [tokenizer_bn.token_to_id("[PAD]")] * (max_len - len(bn_input_tokens))
        if len(bn_output_tokens) > max_len:
            bn_output_tokens = bn_output_tokens[:max_len]
        else:
            bn_output_tokens += [-100] * (max_len - len(bn_output_tokens))
        
        encoder_input.append(en_tokens)
        decoder_input.append(bn_input_tokens)
        decoder_output.append(bn_output_tokens)
    
    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "decoder_output": decoder_output
    }

tokenized_dataset = dataset.map(preprocess, batched=True)

# ===== Step 4: Save dataset =====
os.makedirs("processed", exist_ok=True)
tokenized_dataset.save_to_disk("processed/translation_dataset")

# Save tokenizers for later decoding
with open("processed/tokenizer_en.pkl", "wb") as f:
    pickle.dump(tokenizer_en, f)
with open("processed/tokenizer_bn.pkl", "wb") as f:
    pickle.dump(tokenizer_bn, f)

print("✅ Dataset prepared and saved.")

# ===== Step 5: Load & Verify =====
def load_and_verify():
    ds = Dataset.load_from_disk("processed/translation_dataset")
    with open("processed/tokenizer_en.pkl", "rb") as f:
        tok_en = pickle.load(f)
    with open("processed/tokenizer_bn.pkl", "rb") as f:
        tok_bn = pickle.load(f)
    
    sample = ds[43]

    print("\n--- Verification ---")
    print("Encoder input tokens:", sample["encoder_input"])
    print("Decoder input tokens:", sample["decoder_input"])
    print("Decoder output tokens:", sample["decoder_output"])
    
    # Decode back
    print("Decoded English:", tok_en.decode([x for x in sample["encoder_input"] if x != tokenizer_en.token_to_id("[PAD]")]))
    print("Decoded Bangla (input):", tok_bn.decode([x for x in sample["decoder_input"][1:] if x != tokenizer_bn.token_to_id("[PAD]")]))  # remove SOS
    print("Decoded Bangla (output):", tok_bn.decode([x for x in sample["decoder_output"] if x != -100 and x != 3]))  # remove EOS


# Run verification
load_and_verify()
