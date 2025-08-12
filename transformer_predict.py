import torch
import numpy as np
from model import Model  # Make sure to import your model definition
from prepare_dataset import TranslationTokenizer  # Adjust this import as needed
from tokenizers import Tokenizer

torch.set_float32_matmul_precision('high')

def load_tokenizer(vocab_file):
    return Tokenizer.from_file(vocab_file)

class TranslatorPredictor:
    def __init__(self, model_path: str):
        # Load tokenizer
        self.tokenizer_en = load_tokenizer("tokenizer_en.json")
        self.tokenizer_bn = load_tokenizer("tokenizer_bn.json")

        # Load model and config
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.config = checkpoint['config']  # You saved it in training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Recreate model
        self.model = Model(self.config)  # Your Transformer model class
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, src_sentence: str, max_length: int = 30) -> str:
        # Lowercase and encode English sentence
        src_sentence = src_sentence.lower().strip()
        src_encoded = self.tokenizer_en.encode(src_sentence).ids
        src_tensor = torch.tensor(src_encoded, dtype=torch.long).unsqueeze(0).to(self.device)

        # Decoder starts with SOS token
        ys = torch.tensor([[self.tokenizer_bn.token_to_id("[SOS]")]], dtype=torch.long).to(self.device)

        kv_cache = None
        encoder_output_previous = None

        nxt = ys.clone()

        # with kv cache

        for _ in range(max_length):
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                out, _, encoder_output_previous, kv_cache = self.model(src_tensor, nxt, targets=None, targets_single=None, kv_cache=kv_cache, inference=True, encoder_output_previous=encoder_output_previous)
            
            logits = out[:, -1, :]
            next_token = logits.argmax(dim=-1).item()

            ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long).to(self.device)], dim=1)

            nxt = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

            if next_token == self.tokenizer_bn.token_to_id("[EOS]"):
                break

        # without kv cache
        # for _ in range(max_length):
        #     with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
        #         out, _, encoder_output_previous, _= self.model(src_tensor, ys, kv_cache=None, inference=True, encoder_output_previous=encoder_output_previous)
        #     logits = out[:, -1, :]
        #     next_token = logits.argmax(dim=-1).item()

        #     ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long).to(self.device)], dim=1)

        #     if next_token == self.tokenizer_bn.token_to_id("[EOS]"):
        #         break

        decoded_bn = self.tokenizer_bn.decode(ys.squeeze().tolist())
        
        return decoded_bn

# Example usage
if __name__ == "__main__":
    model_path = "trained_model.pth"
    
    predictor = TranslatorPredictor(model_path)
    
    input_sentence = "a little girl is sitting in front of a large painted rainbow"
    prediction = predictor.predict(input_sentence)
    print(f"Translated: {prediction}")

# Main kv cache development bugs I fixed
# 1. During applying positional embeddings I was applying positions from token 0
# 2. During applying causal mask I was applying from postiions 0
# 3. During inference max numbers of tokens to generate can't greater than block size
