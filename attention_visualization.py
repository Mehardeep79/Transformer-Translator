import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from model import build_transformer
from dataset import causal_mask
from tokenizers import Tokenizer
from config import get_config, get_weights_file_path
import math

# Create output directory for visualizations
os.makedirs("attention_maps", exist_ok=True)

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


def translate_with_attention(model, tokenizer_src, tokenizer_tgt, src_text, device, max_length=100):
    """Translate the given source text and capture attention weights"""
    # Set model to evaluation mode
    model.eval()
    
    # Add SOS and EOS tokens to source
    sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    pad_token_id = tokenizer_src.token_to_id('[PAD]')
    
    # Tokenize the sentence
    src_tokens = tokenizer_src.encode(src_text).ids
    
    # Create the source tensor with SOS, tokens, and EOS
    src = torch.cat([
        sos_token,
        torch.tensor(src_tokens, dtype=torch.int64),
        eos_token
    ]).unsqueeze(0).to(device)
    
    # Create the source mask
    src_mask = (src != pad_token_id).unsqueeze(0).unsqueeze(0).int().to(device)
    
    # Store the tokens for visualization (convert indices to tokens)
    src_token_list = [tokenizer_src.id_to_token(idx) for idx in src[0].cpu().numpy()]
    
    # Perform translation using beam search
    translated_tokens = beam_search_decode(
        model, src, src_mask, tokenizer_src, tokenizer_tgt, max_length, device
    )
    
    # Get target tokens for visualization
    tgt_token_list = [tokenizer_tgt.id_to_token(idx) for idx in translated_tokens.cpu().numpy()]
    
    # Decode the translation
    translated_text = tokenizer_tgt.decode(translated_tokens.detach().cpu().numpy())
    
    # Clean up the translation by removing special tokens
    translated_text = translated_text.replace('[SOS]', '').replace('[EOS]', '').strip()
    
    return translated_text, src_token_list, tgt_token_list, src, src_mask, translated_tokens


def add_attention_hooks(model):
    """Add hooks to the model to capture attention weights"""
    # Initialize dictionaries to store attention weights
    model.enc_self_attentions = {}
    model.dec_self_attentions = {}
    model.cross_attentions = {}
    
    # Add hooks to encoder self-attention
    def get_enc_self_attn_hook(layer_idx, head_idx):
        def hook(module, input, output):
            # Access the stored attention_scores directly from the module
            if hasattr(module, 'attention_scores'):
                if layer_idx not in model.enc_self_attentions:
                    model.enc_self_attentions[layer_idx] = {}
                model.enc_self_attentions[layer_idx][head_idx] = module.attention_scores
        return hook
    
    # Add hooks to decoder self-attention
    def get_dec_self_attn_hook(layer_idx, head_idx):
        def hook(module, input, output):
            # Access the stored attention_scores directly from the module
            if hasattr(module, 'attention_scores'):
                if layer_idx not in model.dec_self_attentions:
                    model.dec_self_attentions[layer_idx] = {}
                model.dec_self_attentions[layer_idx][head_idx] = module.attention_scores
        return hook
    
    # Add hooks to cross-attention
    def get_cross_attn_hook(layer_idx, head_idx):
        def hook(module, input, output):
            # Access the stored attention_scores directly from the module
            if hasattr(module, 'attention_scores'):
                if layer_idx not in model.cross_attentions:
                    model.cross_attentions[layer_idx] = {}
                model.cross_attentions[layer_idx][head_idx] = module.attention_scores
        return hook
    
    # Register hooks for each layer and head
    for i, layer in enumerate(model.encoder.layers):
        for h in range(layer.self_attention_block.h):
            # We need to modify the forward method of MultiHeadAttentionBlock to return attention weights
            layer.self_attention_block.register_forward_hook(get_enc_self_attn_hook(i, h))
    
    for i, layer in enumerate(model.decoder.layers):
        for h in range(layer.self_attention_block.h):
            layer.self_attention_block.register_forward_hook(get_dec_self_attn_hook(i, h))
        for h in range(layer.cross_attention_block.h):
            layer.cross_attention_block.register_forward_hook(get_cross_attn_hook(i, h))
            
    print("Registered attention hooks for visualization")


def modify_attention_method(model):
    """Modify the attention method to return attention weights"""
    # Define the new method that will store attention weights
    def new_attention(self, query, key, value, mask, dropout):
        d_k = query.shape[-1]
        # Calculate attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_probs = dropout(attention_probs)
        
        # Store attention probabilities for visualization
        self.attention_scores = attention_probs.detach()
        
        # Apply attention to value
        output = attention_probs @ value
        
        # Return output (without attention probabilities as in the original method)
        return output
    
    # Replace the method in all attention blocks
    import types
    
    # Replace attention method in all encoder and decoder layers
    for layer in model.encoder.layers:
        layer.self_attention_block.attention = types.MethodType(new_attention, layer.self_attention_block)
    
    for layer in model.decoder.layers:
        layer.self_attention_block.attention = types.MethodType(new_attention, layer.self_attention_block)
        layer.cross_attention_block.attention = types.MethodType(new_attention, layer.cross_attention_block)
    
    return model


def visualize_attention(model, src_tokens, tgt_tokens, output_dir="attention_maps"):
    """Create and save attention visualizations"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename prefix from source text (clean non-alphanumeric characters)
    prefix = ''.join([c if c.isalnum() else '_' for c in ' '.join(src_tokens)[:30]])
    
    # Define max visualization length (to avoid too large visualizations)
    max_len = min(20, len(src_tokens), len(tgt_tokens))
    
    # Truncate tokens for better visualization
    src_tokens = src_tokens[:max_len]
    tgt_tokens = tgt_tokens[:max_len]
    
    # Print information to help debug
    print(f"Saving attention visualizations for {max_len} tokens...")
    
    try:
        # Visualize encoder self-attention (if available)
        if hasattr(model, 'enc_self_attentions'):
            for layer_idx, layer_attns in model.enc_self_attentions.items():
                for head_idx, attn_weights in layer_attns.items():
                    if attn_weights is None:
                        print(f"Warning: No encoder attention weights for layer {layer_idx}, head {head_idx}")
                        continue
                        
                    plt.figure(figsize=(10, 8))
                    # Extract the attention matrix for the first batch
                    attn_matrix = attn_weights[0, head_idx, :max_len, :max_len].cpu().detach().numpy()
                    
                    # Create heatmap
                    sns.heatmap(attn_matrix, annot=False, cmap='viridis',
                                xticklabels=src_tokens, yticklabels=src_tokens)
                    
                    plt.title(f"Encoder Self-Attention, Layer {layer_idx}, Head {head_idx}")
                    plt.tight_layout()
                    output_file = f"{output_dir}/{prefix}_enc_layer{layer_idx}_head{head_idx}.png"
                    plt.savefig(output_file)
                    plt.close()
                    print(f"Saved encoder attention to {output_file}")
        else:
            print("Warning: Model doesn't have encoder self-attention weights")
            
        # Print sizes of attention dictionaries to help debug
        if hasattr(model, 'enc_self_attentions'):
            print(f"Encoder attention layers: {len(model.enc_self_attentions)}")
        if hasattr(model, 'dec_self_attentions'):
            print(f"Decoder attention layers: {len(model.dec_self_attentions)}")
        if hasattr(model, 'cross_attentions'):
            print(f"Cross attention layers: {len(model.cross_attentions)}")
            
        # Visualize decoder self-attention
        if hasattr(model, 'dec_self_attentions'):
            for layer_idx, layer_attns in model.dec_self_attentions.items():
                for head_idx, attn_weights in layer_attns.items():
                    if attn_weights is None:
                        continue
                        
                    plt.figure(figsize=(10, 8))
                    # Extract the attention matrix for the first batch
                    attn_matrix = attn_weights[0, head_idx, :max_len, :max_len].cpu().detach().numpy()
                    
                    # Create heatmap
                    sns.heatmap(attn_matrix, annot=False, cmap='viridis',
                                xticklabels=tgt_tokens, yticklabels=tgt_tokens)
                    
                    plt.title(f"Decoder Self-Attention, Layer {layer_idx}, Head {head_idx}")
                    plt.tight_layout()
                    output_file = f"{output_dir}/{prefix}_dec_layer{layer_idx}_head{head_idx}.png"
                    plt.savefig(output_file)
                    plt.close()
                    print(f"Saved decoder attention to {output_file}")
        
        # Visualize cross-attention
        if hasattr(model, 'cross_attentions'):
            for layer_idx, layer_attns in model.cross_attentions.items():
                for head_idx, attn_weights in layer_attns.items():
                    if attn_weights is None:
                        continue
                        
                    plt.figure(figsize=(10, 8))
                    # Extract the attention matrix for the first batch
                    attn_matrix = attn_weights[0, head_idx, :max_len, :max_len].cpu().detach().numpy()
                    
                    # Create heatmap
                    sns.heatmap(attn_matrix, annot=False, cmap='viridis',
                                xticklabels=src_tokens, yticklabels=tgt_tokens)
                    
                    plt.title(f"Cross-Attention, Layer {layer_idx}, Head {head_idx}")
                    plt.tight_layout()
                    output_file = f"{output_dir}/{prefix}_cross_layer{layer_idx}_head{head_idx}.png"
                    plt.savefig(output_file)
                    plt.close()
                    print(f"Saved cross attention to {output_file}")
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    return output_dir


def load_model():
    """Load the trained model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get configuration
    config = get_config()
    
    # Load tokenizers
    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
    
    # Build the transformer model
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(), 
        config['seq_len'], 
        config['seq_len'], 
        d_model=config['d_model']
    ).to(device)
    
    # Load model weights (30th epoch)
    model_path = get_weights_file_path(config, "30")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    print(f"Successfully loaded model from {model_path}")
    
    # Modify the attention method to capture attention weights
    model = modify_attention_method(model)
    
    # Register hooks to capture attention weights
    add_attention_hooks(model)
    
    return model, tokenizer_src, tokenizer_tgt, device, config


def main():
    """Main function for interactive translation with attention visualization"""
    # Load the model, tokenizers, and set device
    model, tokenizer_src, tokenizer_tgt, device, config = load_model()
    
    print("\nEnglish to Italian Translation with Attention Visualization")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    # Create a simple sample first to check if attention mechanism works
    test_text = "Hello, how are you?"
    print(f"Testing attention capture with: '{test_text}'")
    try:
        translated_text, src_tokens, tgt_tokens, _, _, _ = translate_with_attention(
            model, tokenizer_src, tokenizer_tgt, test_text, device, max_length=config['seq_len']
        )
        
        # Check if attention weights are captured
        has_enc_attn = hasattr(model, 'enc_self_attentions') and len(model.enc_self_attentions) > 0
        has_dec_attn = hasattr(model, 'dec_self_attentions') and len(model.dec_self_attentions) > 0
        has_cross_attn = hasattr(model, 'cross_attentions') and len(model.cross_attentions) > 0
        
        print(f"Translation: {translated_text}")
        print(f"Attention weights captured: Encoder={has_enc_attn}, Decoder={has_dec_attn}, Cross={has_cross_attn}")
        
        if not (has_enc_attn or has_dec_attn or has_cross_attn):
            print("""
WARNING: No attention weights were captured. This might be due to:
1. Your model implementation doesn't store attention scores
2. The attention modification wasn't applied correctly
3. You may need to modify your model's MultiHeadAttentionBlock class
""")
    except Exception as e:
        print(f"Error during test: {e}")
    
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
            # Clear previous attention data
            if hasattr(model, 'enc_self_attentions'):
                model.enc_self_attentions.clear()
            if hasattr(model, 'dec_self_attentions'):
                model.dec_self_attentions.clear()
            if hasattr(model, 'cross_attentions'):
                model.cross_attentions.clear()
            
            # Translate the text and get tokens for visualization
            translated_text, src_tokens, tgt_tokens, _, _, _ = translate_with_attention(
                model, tokenizer_src, tokenizer_tgt, src_text, device, max_length=config['seq_len']
            )
            
            # Display translation
            print(f"Italian translation: {translated_text}")
            
            # Visualize attention and save to files
            output_dir = visualize_attention(model, src_tokens, tgt_tokens)
            print(f"Attention visualizations saved to '{output_dir}' directory")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error during translation or visualization: {e}")
            print("Please try with a shorter or simpler sentence.")
            print("-" * 50)


if __name__ == "__main__":
    main() 