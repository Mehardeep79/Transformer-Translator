import torch
import torch.nn as nn
from model import build_transformer
from dataset import causal_mask
from tokenizers import Tokenizer
from pathlib import Path
import sys


def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=5):
    """Beam search for better translation quality"""    
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Encode the source sentence
    encoder_output = model.encode(source, source_mask)

    # Initialize the beam with start token
    sequences = [(torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device), 0.0)]

    # Beam search
    for _ in range(max_len):
        new_sequences = []

        # Expand each current sequence
        for seq, score in sequences:
            # If sequence ended with EOS, keep it unchanged
            if seq.size(1) > 1 and seq[0, -1].item() == eos_idx:
                new_sequences.append((seq, score))
                continue

            # Create decoder mask for this sequence
            decoder_mask = causal_mask(seq.size(1)).type_as(source_mask).to(device)

            # Get next token probabilities
            out = model.decode(encoder_output, source_mask, seq, decoder_mask)
            prob = model.project(out[:, -1])
            log_prob = torch.log_softmax(prob, dim=-1)

            # Get top-k token candidates
            topk_probs, topk_indices = torch.topk(log_prob, beam_size, dim=1)

            # Add new candidates to the list
            for i in range(beam_size):
                token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, token], dim=1)
                new_score = score + topk_probs[0, i].item()
                new_sequences.append((new_seq, new_score))

        # Select top-k sequences
        new_sequences.sort(key=lambda x: x[1], reverse=True)
        sequences = new_sequences[:beam_size]

        # Check if all sequences have ended or reached max length
        if all((seq.size(1) > 1 and seq[0, -1].item() == eos_idx) or seq.size(1) >= max_len
               for seq, _ in sequences):
            break

    # Return the best sequence
    return sequences[0][0].squeeze(0)


def translate(model, tokenizer_src, tokenizer_tgt, src_text, device, max_length=100):
    """Translate the given source text to target language"""
    # Tokenize the source text
    model.eval()
    
    # Add SOS and EOS tokens to source
    sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    
    # Tokenize the sentence
    src_tokens = tokenizer_src.encode(src_text).ids
    
    # Create the source tensor
    src = torch.cat([
        sos_token,
        torch.tensor(src_tokens, dtype=torch.int64),
        eos_token
    ]).unsqueeze(0).to(device)
    
    # Create the source mask
    src_mask = (src != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
    
    # Perform translation using beam search
    translated_tokens = beam_search_decode(
        model, src, src_mask, tokenizer_src, tokenizer_tgt, max_length, device
    )
    
    # Decode the translation
    translated_text = tokenizer_tgt.decode(translated_tokens.detach().cpu().numpy())
    
    # Clean up the translation by removing special tokens
    translated_text = translated_text.replace('[SOS]', '').replace('[EOS]', '').strip()
    
    return translated_text


def load_model(model_path):
    """Load the trained model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizers
    tokenizer_src = Tokenizer.from_file('tokenizer_en.json')
    tokenizer_tgt = Tokenizer.from_file('tokenizer_it.json')
    
    # Define model parameters from config
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    
    # Build the transformer model
    model = build_transformer(
        src_vocab_size, 
        tgt_vocab_size, 
        src_seq_len=350, 
        tgt_seq_len=350, 
        d_model=512
    ).to(device)
    
    # Load the model weights
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state['model_state_dict'])
    
    return model, tokenizer_src, tokenizer_tgt, device


def main():
    """Main function for interactive translation"""
    # Load the model, tokenizers, and set device
    model_path = "opus_books_weights/tmodel_30.pt"
    model, tokenizer_src, tokenizer_tgt, device = load_model(model_path)
    
    print("English to Italian Translation System")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    # Interactive loop
    while True:
        # Get input from user
        src_text = input("Enter English text: ")
        
        # Exit condition
        if src_text.lower() == 'exit':
            break
        
        # Skip empty inputs
        if not src_text.strip():
            continue
        
        try:
            # Translate the text
            translated_text = translate(model, tokenizer_src, tokenizer_tgt, src_text, device)
            
            # Display translation
            print(f"Italian translation: {translated_text}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error during translation: {e}")
            print("Please try with a shorter or simpler sentence.")
            print("-" * 50)


if __name__ == "__main__":
    main() 