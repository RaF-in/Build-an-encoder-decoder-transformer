import re
import pickle
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
from model import Config

config = Config()

class TranslationTokenizer:
    def __init__(self, vocab_size: int = config.vocab_size):
        self.vocab_size = vocab_size
        self.en_vocab = {}
        self.bn_vocab = {}
        self.en_word2idx = {}
        self.en_idx2word = {}
        self.bn_word2idx = {}
        self.bn_idx2word = {}
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'  # Start of sequence
        self.EOS_TOKEN = '<EOS>'  # End of sequence
        
    def load_data_from_txt(self, file_path: str) -> List[Tuple[str, str]]:
        """Load processed data from txt file"""
        pairs = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Split by --- separator
        entries = content.split('---')
        
        for entry in entries:
            if not entry.strip():
                continue
                
            lines = entry.strip().split('\n')
            en_tokens = ""
            bn_tokens = ""
            
            for line in lines:
                if line.startswith('EN_TOKENS:'):
                    en_tokens = line.replace('EN_TOKENS:', '').strip().lower()  # <-- lowercase here
                elif line.startswith('BN_TOKENS:'):
                    bn_tokens = line.replace('BN_TOKENS:', '').strip()
            
            if en_tokens and bn_tokens:
                pairs.append((en_tokens, bn_tokens))
        
        return pairs

    
    def build_vocabulary(self, pairs: List[Tuple[str, str]]):
        """Build vocabulary from training pairs"""
        en_counter = Counter()
        bn_counter = Counter()
        
        # Count word frequencies
        for en_tokens, bn_tokens in pairs:
            en_words = en_tokens.split()
            bn_words = bn_tokens.split()
            
            en_counter.update(en_words)
            bn_counter.update(bn_words)
        
        # Build English vocabulary
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        
        # English vocabulary
        self.en_word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        most_common_en = en_counter.most_common(self.vocab_size - len(special_tokens))
        
        for word, _ in most_common_en:
            if word not in self.en_word2idx:
                self.en_word2idx[word] = len(self.en_word2idx)
        
        self.en_idx2word = {idx: word for word, idx in self.en_word2idx.items()}
        
        # Bengali vocabulary
        self.bn_word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        most_common_bn = bn_counter.most_common(self.vocab_size - len(special_tokens))
        
        for word, _ in most_common_bn:
            if word not in self.bn_word2idx:
                self.bn_word2idx[word] = len(self.bn_word2idx)
        
        self.bn_idx2word = {idx: word for word, idx in self.bn_word2idx.items()}
        
        print(f"English vocabulary size: {len(self.en_word2idx)}")
        print(f"Bengali vocabulary size: {len(self.bn_word2idx)}")
    
    def encode_sentence(self, sentence: str, vocab: Dict[str, int], add_special_tokens: bool = True) -> List[int]:
        """Convert sentence to sequence of token IDs"""
        words = sentence.split()
        
        # Convert words to indices
        indices = []
        if add_special_tokens:
            indices.append(vocab[self.SOS_TOKEN])
        
        for word in words:
            if word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab[self.UNK_TOKEN])
        
        if add_special_tokens:
            indices.append(vocab[self.EOS_TOKEN])
        
        return indices
    
    def decode_sentence(self, indices: List[int], idx2word: Dict[int, str]) -> str:
        """Convert sequence of token IDs back to sentence"""
        words = []
        for idx in indices:
            if idx in idx2word:
                word = idx2word[idx]
                if word not in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                    words.append(word)
        return ' '.join(words)
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int = None) -> np.ndarray:
        """Pad sequences to same length"""
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded = np.full((len(sequences), max_length), self.en_word2idx[self.PAD_TOKEN])
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded[i, :length] = seq[:length]
        
        return padded
    
    def prepare_training_data(self, pairs: List[Tuple[str, str]], max_length: int = config.block_size):
        """Prepare data for transformer training"""
        print("Encoding sentences...")
        
        encoder_inputs = []  # English sentences (source)
        decoder_inputs = []  # Bengali sentences with SOS (target input)
        decoder_targets = []  # Bengali sentences with EOS (target output)
        
        for en_tokens, bn_tokens in pairs:
            # Encode English (source) - encoder input
            en_encoded = self.encode_sentence(en_tokens, self.en_word2idx, add_special_tokens=False)
            en_encoded = [self.en_word2idx[self.SOS_TOKEN]] + en_encoded + [self.en_word2idx[self.EOS_TOKEN]]
            
            # Encode Bengali (target)
            bn_encoded = self.encode_sentence(bn_tokens, self.bn_word2idx, add_special_tokens=False)
            
            # Decoder input: SOS + target sentence
            decoder_input = [self.bn_word2idx[self.SOS_TOKEN]] + bn_encoded
            
            # Decoder target: target sentence + EOS
            decoder_target = bn_encoded + [self.bn_word2idx[self.EOS_TOKEN]]
            
            # Skip sequences that are too long
            if len(en_encoded) <= max_length and len(decoder_input) <= max_length:
                encoder_inputs.append(en_encoded)
                decoder_inputs.append(decoder_input)
                decoder_targets.append(decoder_target)
        
        print(f"Prepared {len(encoder_inputs)} training examples")
        
        # Pad sequences
        encoder_inputs = self.pad_sequences(encoder_inputs, max_length)
        decoder_inputs = self.pad_sequences(decoder_inputs, max_length)
        decoder_targets = self.pad_sequences(decoder_targets, max_length)
        
        return encoder_inputs, decoder_inputs, decoder_targets
    
    def save_tokenizer(self, file_path: str):
        """Save tokenizer to file"""
        tokenizer_data = {
            'en_word2idx': self.en_word2idx,
            'en_idx2word': self.en_idx2word,
            'bn_word2idx': self.bn_word2idx,
            'bn_idx2word': self.bn_idx2word,
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'PAD_TOKEN': self.PAD_TOKEN,
                'UNK_TOKEN': self.UNK_TOKEN,
                'SOS_TOKEN': self.SOS_TOKEN,
                'EOS_TOKEN': self.EOS_TOKEN
            }
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        print(f"Tokenizer saved to {file_path}")
    
    def load_tokenizer(self, file_path: str):
        """Load tokenizer from file"""
        with open(file_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.en_word2idx = tokenizer_data['en_word2idx']
        self.en_idx2word = tokenizer_data['en_idx2word']
        self.bn_word2idx = tokenizer_data['bn_word2idx']
        self.bn_idx2word = tokenizer_data['bn_idx2word']
        self.vocab_size = tokenizer_data['vocab_size']
        
        special_tokens = tokenizer_data['special_tokens']
        self.PAD_TOKEN = special_tokens['PAD_TOKEN']
        self.UNK_TOKEN = special_tokens['UNK_TOKEN']
        self.SOS_TOKEN = special_tokens['SOS_TOKEN']
        self.EOS_TOKEN = special_tokens['EOS_TOKEN']
        
        print(f"Tokenizer loaded from {file_path}")

def create_transformer_dataset(txt_file_path: str, output_dir: str = "./transformer_data/"):
    """Main function to create transformer-ready dataset"""
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = TranslationTokenizer(vocab_size=config.vocab_size)  # Adjust vocab size as needed
    
    # Load data
    print("Loading data from txt file...")
    pairs = tokenizer.load_data_from_txt(txt_file_path)
    print(f"Loaded {len(pairs)} translation pairs")
    
    # Show sample data
    print("\nSample data:")
    for i in range(min(3, len(pairs))):
        en, bn = pairs[i]
        print(f"EN: {en}")
        print(f"BN: {bn}")
        print("---")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    tokenizer.build_vocabulary(pairs)
    
    # Prepare training data
    print("\nPreparing training data...")
    encoder_inputs, decoder_inputs, decoder_targets = tokenizer.prepare_training_data(pairs, max_length=config.block_size)
    
    # Save data
    np.save(f"{output_dir}/encoder_inputs.npy", encoder_inputs)
    np.save(f"{output_dir}/decoder_inputs.npy", decoder_inputs)
    np.save(f"{output_dir}/decoder_targets.npy", decoder_targets)
    
    # Save tokenizer
    tokenizer.save_tokenizer(f"{output_dir}/tokenizer.pkl")
    
    print(f"\nData shapes:")
    print(f"Encoder inputs: {encoder_inputs.shape}")
    print(f"Decoder inputs: {decoder_inputs.shape}")
    print(f"Decoder targets: {decoder_targets.shape}")
    
    # Save data info
    with open(f"{output_dir}/data_info.txt", 'w', encoding='utf-8') as f:
        f.write(f"Dataset Information\n")
        f.write(f"==================\n")
        f.write(f"Total samples: {len(pairs)}\n")
        f.write(f"English vocab size: {len(tokenizer.en_word2idx)}\n")
        f.write(f"Bengali vocab size: {len(tokenizer.bn_word2idx)}\n")
        f.write(f"Max sequence length: config.block_size\n")
        f.write(f"Encoder input shape: {encoder_inputs.shape}\n")
        f.write(f"Decoder input shape: {decoder_inputs.shape}\n")
        f.write(f"Decoder target shape: {decoder_targets.shape}\n")
        f.write(f"\nSpecial tokens:\n")
        f.write(f"PAD: {tokenizer.PAD_TOKEN} (ID: {tokenizer.en_word2idx[tokenizer.PAD_TOKEN]})\n")
        f.write(f"UNK: {tokenizer.UNK_TOKEN} (ID: {tokenizer.en_word2idx[tokenizer.UNK_TOKEN]})\n")
        f.write(f"SOS: {tokenizer.SOS_TOKEN} (ID: {tokenizer.en_word2idx[tokenizer.SOS_TOKEN]})\n")
        f.write(f"EOS: {tokenizer.EOS_TOKEN} (ID: {tokenizer.en_word2idx[tokenizer.EOS_TOKEN]})\n")
    
    print(f"\nAll files saved to {output_dir}")
    print("Files created:")
    print("- encoder_inputs.npy (source sequences)")
    print("- decoder_inputs.npy (target input sequences)")
    print("- decoder_targets.npy (target output sequences)")
    print("- tokenizer.pkl (vocabulary and tokenizer)")
    print("- data_info.txt (dataset information)")
    
    return tokenizer, encoder_inputs, decoder_inputs, decoder_targets

# Example usage and testing
if __name__ == "__main__":
    # Replace with your actual txt file path
    txt_file = "kde4_processed_data.txt"
    
    try:
        # Create transformer dataset
        tokenizer, enc_inputs, dec_inputs, dec_targets = create_transformer_dataset(txt_file)
        
        # Test the tokenizer
        print("\n" + "="*config.block_size)
        print("TESTING TOKENIZER")
        print("="*config.block_size)
        
        # Test encoding and decoding
        test_en = "russian"
        test_bn = "রুশ"
        
        en_encoded = tokenizer.encode_sentence(test_en, tokenizer.en_word2idx)
        bn_encoded = tokenizer.encode_sentence(test_bn, tokenizer.bn_word2idx)
        
        print(f"Original English: {test_en}")
        print(f"Encoded: {en_encoded}")
        print(f"Decoded: {tokenizer.decode_sentence(en_encoded, tokenizer.en_idx2word)}")
        
        print(f"\nOriginal Bengali: {test_bn}")
        print(f"Encoded: {bn_encoded}")
        print(f"Decoded: {tokenizer.decode_sentence(bn_encoded, tokenizer.bn_idx2word)}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {txt_file}")
        print("Please make sure the file exists and update the file path")