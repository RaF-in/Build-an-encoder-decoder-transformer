import torch
import numpy as np
from model import Model  # Make sure to import your model definition
from prepare_dataset import TranslationTokenizer  # Adjust this import as needed

torch.set_float32_matmul_precision('high')

class TranslatorPredictor:
    def __init__(self, model_path: str, tokenizer_path: str):
        # Load tokenizer
        self.tokenizer = TranslationTokenizer()
        self.tokenizer.load_tokenizer(tokenizer_path)

        # Load model and config
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.config = checkpoint['config']  # You saved it in training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Recreate model
        self.model = Model(self.config)  # Your Transformer model class
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, src_sentence: str, max_length: int = 50) -> str:
        # Lowercase and encode English sentence
        src_sentence = src_sentence.lower().strip()
        src_encoded = self.tokenizer.encode_sentence(
            src_sentence, self.tokenizer.en_word2idx, add_special_tokens=True
        )
        src_tensor = torch.tensor(src_encoded, dtype=torch.long).unsqueeze(0).to(self.device)

        # Decoder starts with SOS token
        ys = torch.tensor([[self.tokenizer.bn_word2idx[self.tokenizer.SOS_TOKEN]]], dtype=torch.long).to(self.device)

        kv_cache = None
        encoder_output_previous = None

        nxt = ys.clone()

        for _ in range(max_length):
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                out, _, encoder_output_previous, kv_cache = self.model(src_tensor, nxt, kv_cache=kv_cache, inference=True, encoder_output_previous=encoder_output_previous)
            logits = out[:, -1, :]
            next_token = logits.argmax(dim=-1).item()

            ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long).to(self.device)], dim=1)

            nxt = ys[:, -1:]  # Slice last token

            if next_token == self.tokenizer.bn_word2idx[self.tokenizer.EOS_TOKEN]:
                break

        decoded_bn = self.tokenizer.decode_sentence(ys.squeeze().tolist(), self.tokenizer.bn_idx2word)
        return decoded_bn

# Example usage
if __name__ == "__main__":
    model_path = "trained_model2.pth"
    tokenizer_path = "transformer_data/tokenizer.pkl"
    
    predictor = TranslatorPredictor(model_path, tokenizer_path)
    
    input_sentence = "russian boy is here"
    prediction = predictor.predict(input_sentence)
    print(f"Translated: {prediction}")
