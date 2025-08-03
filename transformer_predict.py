import torch
import numpy as np
from model import Model  # Make sure to import your model definition
from prepare_dataset import TranslationTokenizer  # Adjust this import as needed

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

        # # Create attention mask for encoder input
        # src_mask = (src_tensor != self.tokenizer.en_word2idx[self.tokenizer.PAD_TOKEN]).unsqueeze(1).to(self.device)

        # # Encode input sentence
        # memory = self.model.encode(src_tensor, src_mask)

        # Decoder starts with SOS token
        ys = torch.tensor([[self.tokenizer.bn_word2idx[self.tokenizer.SOS_TOKEN]]], dtype=torch.long).to(self.device)

        for _ in range(max_length):
            # tgt_mask = self.model.generate_square_subsequent_mask(ys.size(1)).to(self.device)
            out, _ = self.model(src_tensor, ys)
            logits = out[:, -1, :]
            next_token = logits.argmax(dim=-1).item()

            ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long).to(self.device)], dim=1)

            if next_token == self.tokenizer.bn_word2idx[self.tokenizer.EOS_TOKEN]:
                break

        decoded_bn = self.tokenizer.decode_sentence(ys.squeeze().tolist(), self.tokenizer.bn_idx2word)
        return decoded_bn

# Example usage
if __name__ == "__main__":
    model_path = "trained_model2.pth"
    tokenizer_path = "transformer_data/tokenizer.pkl"
    
    predictor = TranslatorPredictor(model_path, tokenizer_path)
    
    input_sentence = "hello world"
    prediction = predictor.predict(input_sentence)
    print(f"Translated: {prediction}")
