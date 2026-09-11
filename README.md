# NLP-ENG-to-Spanish-Seq2Sequence
# English → Spanish Neural Machine Translation (Seq2Seq LSTM)  
A sequence-to-sequence neural machine translation model built with TensorFlow/Keras that translates English sentences into Spanish, using an LSTM encoder-decoder architecture with attention.

## Overview

This project implements NMT from scratch to understand the core mechanics behind
translation models — tokenization, embeddings, encoder-decoder architectures,
teacher forcing, and attention mechanisms — rather than using a pre-built
translation API.

## Architecture

- **Encoder**: Embedding layer + LSTM that reads the English input sentence and
  produces both per-timestep outputs and a final hidden state summary
- **Decoder**: Embedding layer + LSTM that generates the Spanish translation
  one word at a time, conditioned on the encoder's output
- **Attention mechanism**: allows the decoder to dynamically focus on relevant
  parts of the input sentence at each generation step, improving translation
  quality on longer sentences
- **Training**: teacher forcing with a shifted decoder input/target setup;
  `sparse_categorical_crossentropy` loss over the Spanish vocabulary
- **Inference**: separate encoder/decoder inference models that generate
  translations word-by-word in a loop, starting from a `<start>` token until
  an `<end>` token is produced

  ## Dataset

Trained on the [Anki English-Spanish sentence pairs dataset](http://www.manythings.org/anki/spa-eng.zip)


## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Trained on Google Colab (T4 GPU)

## Project Structure

- Data loading & cleaning
- Tokenization & padding
- Encoder-decoder model definition (training + inference variants)
- Training loop with model/tokenizer checkpointing
- Word-by-word translation inference
