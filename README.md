# Project: Ophelia
## Who is Ophelia?
She is a tiny custom character-level **autoregressive transformer** built from **scratch** using TensorFlow/Keras.
The main goal was to learn about **attention mechanism**, **transformer**, and **autoregression system**, alongside personal desire of creating a *"life"*.

## Architecture Overview
- **Type**: Autoregressive Decoder-only Transformer
- **Embedding Dim (d_model)**: 64
- **Positional Embedding**: Learned 1D embedding (X = E + P) from GPT-2
- **Transformer Block**:
  - Pre-Layer Normalization (Pre-LN) for training stability
  - Multi-Head Causal Self-Attention to prevent future token leakage
  - Feed-Forward Network (FFN) with GELU activation
  - Residual streaming across sub-layers (Attention and FFN blocks)

## Project Structures (unfinished)
1) `ophelia.txt` - personal checklist and roadmap
2) `convos.py`   - dataset holding small conversations
3) `prepare.py`  - tokenizer and dataset serialization (`vocab.json`, `encoded.npy`)
4) `dataset.py`  - vectorized sliding-window data loader, generating (B, T) inputs and right-shifted targets
5) `model.py`    - resolving input vector (X) from token and positional embedding (E + P)

## References
- *Attention Is All You Need* by Vaswani et al. (2017), Section 3.2 and further
- Google Gemini as personal co-pilot and trainer
