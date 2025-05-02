import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from attention_visualization import load_model, translate_with_attention, visualize_attention

# Page configuration
st.set_page_config(
    page_title="Attention Visualization for Translation",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("English to Italian Translation with Attention Visualization")
st.markdown("This app shows the attention weights of a transformer model while translating English to Italian.")

# Create a directory for storing visualizations
os.makedirs("attention_maps", exist_ok=True)

# Load model on startup
@st.cache_resource
def get_model_with_attention():
    try:
        model, tokenizer_src, tokenizer_tgt, device, config = load_model()
        return model, tokenizer_src, tokenizer_tgt, device, config, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None, False

# Get model, tokenizers and device
model, tokenizer_src, tokenizer_tgt, device, config, model_loaded = get_model_with_attention()

# Input area
user_input = st.text_area("Enter English text:", height=150)

# Translation button
if st.button("Translate and Visualize Attention"):
    if not user_input.strip():
        st.warning("Please enter some text to translate.")
    elif model_loaded:
        with st.spinner("Translating and generating visualizations..."):
            try:
                # Clear previous attention data
                if hasattr(model, 'enc_self_attentions'):
                    model.enc_self_attentions.clear()
                if hasattr(model, 'dec_self_attentions'):
                    model.dec_self_attentions.clear()
                if hasattr(model, 'cross_attentions'):
                    model.cross_attentions.clear()
                
                # Translate and get tokens
                translated_text, src_tokens, tgt_tokens, _, _, _ = translate_with_attention(
                    model, tokenizer_src, tokenizer_tgt, user_input, device, 
                    max_length=config['seq_len']
                )
                
                st.success("Translation complete!")
                
                # Display results
                st.subheader("Italian Translation:")
                st.write(translated_text)
                
                # Visualize attention weights
                has_enc_attn = hasattr(model, 'enc_self_attentions') and len(model.enc_self_attentions) > 0
                has_dec_attn = hasattr(model, 'dec_self_attentions') and len(model.dec_self_attentions) > 0
                has_cross_attn = hasattr(model, 'cross_attentions') and len(model.cross_attentions) > 0
                
                if has_enc_attn or has_dec_attn or has_cross_attn:
                    st.subheader("Attention Visualizations:")
                    
                    # Determine max visualization length
                    max_len = min(20, len(src_tokens), len(tgt_tokens))
                    src_tokens_vis = src_tokens[:max_len]
                    tgt_tokens_vis = tgt_tokens[:max_len]
                    
                    # Create tabs for different attention types
                    tab1, tab2, tab3 = st.tabs(["Encoder Self-Attention", "Decoder Self-Attention", "Cross-Attention"])
                    
                    # Encoder self-attention
                    with tab1:
                        if has_enc_attn:
                            for layer_idx, layer_attns in model.enc_self_attentions.items():
                                st.write(f"#### Layer {layer_idx}")
                                cols = st.columns(min(4, len(layer_attns)))
                                for i, (head_idx, attn_weights) in enumerate(layer_attns.items()):
                                    if attn_weights is not None:
                                        with cols[i % len(cols)]:
                                            fig, ax = plt.subplots(figsize=(5, 4))
                                            attn_matrix = attn_weights[0, head_idx, :max_len, :max_len].cpu().detach().numpy()
                                            sns.heatmap(attn_matrix, annot=False, cmap='viridis', 
                                                      xticklabels=src_tokens_vis, yticklabels=src_tokens_vis, ax=ax)
                                            plt.title(f"Head {head_idx}")
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.close(fig)
                        else:
                            st.info("No encoder self-attention weights available")
                    
                    # Decoder self-attention
                    with tab2:
                        if has_dec_attn:
                            for layer_idx, layer_attns in model.dec_self_attentions.items():
                                st.write(f"#### Layer {layer_idx}")
                                cols = st.columns(min(4, len(layer_attns)))
                                for i, (head_idx, attn_weights) in enumerate(layer_attns.items()):
                                    if attn_weights is not None:
                                        with cols[i % len(cols)]:
                                            fig, ax = plt.subplots(figsize=(5, 4))
                                            attn_matrix = attn_weights[0, head_idx, :max_len, :max_len].cpu().detach().numpy()
                                            sns.heatmap(attn_matrix, annot=False, cmap='viridis', 
                                                      xticklabels=tgt_tokens_vis, yticklabels=tgt_tokens_vis, ax=ax)
                                            plt.title(f"Head {head_idx}")
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.close(fig)
                        else:
                            st.info("No decoder self-attention weights available")
                    
                    # Cross-attention
                    with tab3:
                        if has_cross_attn:
                            for layer_idx, layer_attns in model.cross_attentions.items():
                                st.write(f"#### Layer {layer_idx}")
                                cols = st.columns(min(4, len(layer_attns)))
                                for i, (head_idx, attn_weights) in enumerate(layer_attns.items()):
                                    if attn_weights is not None:
                                        with cols[i % len(cols)]:
                                            fig, ax = plt.subplots(figsize=(5, 4))
                                            attn_matrix = attn_weights[0, head_idx, :max_len, :max_len].cpu().detach().numpy()
                                            sns.heatmap(attn_matrix, annot=False, cmap='viridis', 
                                                      xticklabels=src_tokens_vis, yticklabels=tgt_tokens_vis, ax=ax)
                                            plt.title(f"Head {head_idx}")
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.close(fig)
                        else:
                            st.info("No cross-attention weights available")
                else:
                    st.warning("No attention weights were captured. The model may not be configured to store attention scores.")
                
            except Exception as e:
                st.error(f"Translation or visualization error: {str(e)}")
                st.info("Try with a shorter or simpler sentence.")

# Add explanation about the visualizations
with st.expander("About Attention Visualization"):
    st.markdown("""
    ## What am I looking at?
    
    The visualizations show the attention weights from different parts of the transformer model:
    
    1. **Encoder Self-Attention**: How each word in the source (English) attends to other words in the source
    2. **Decoder Self-Attention**: How each word in the target (Italian) attends to previous words in the target
    3. **Cross-Attention**: How each word in the target attends to words in the source
    
    Brighter colors indicate stronger attention weights. This visualization helps understand how the model "focuses" on different words during translation.
    """)

# Footer
st.markdown("---")
st.markdown("Powered by a Transformer neural network model with attention visualization") 