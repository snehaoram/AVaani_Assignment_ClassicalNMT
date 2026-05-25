# Hindi ↔ Marathi Neural Machine Translation

This repository contains an implementation of a classical Neural Machine Translation (NMT) system for Hindi ↔ Marathi translation using an **LSTM Encoder–Decoder with Attention** architecture.

Two embedding strategies are explored:

1. **BERT-based embeddings**
2. **Randomly initialized embeddings**

## Model Architecture

* Encoder: LSTM
* Decoder: LSTM with Attention
* Beam Search Decoding (beam width = 5)
* Teacher Forcing (ratio = 0.7)
* Decoder Weight Tying
* Optimizer: Adam
* Gradient Clipping: 0.1

## Dataset Setup

* 30% of the provided training data is used
* Training set: first 72k sentence pairs
* Validation set: 7k sentence pairs immediately following the training split

## Method 1: BERT-based Embeddings

* Uses Hindi-BERT and Marathi-BERT tokenizers
* Vocabulary size reduced to 10k for efficient training
* Embedding dimension: 256

### Observation

* Lower training and validation loss compared to random embeddings
* Better BLEU and ChRF++ scores
* Marathi → Hindi translation performs comparatively better

## Method 2: Random Embeddings

* Uses randomly initialized embeddings with the same architecture and training setup
* Frequent tokenizer IDs are remapped to a reduced 10k vocabulary

### Observation

* Training loss decreases, but validation improvement is limited
* Performance is lower than BERT-based embeddings

## Comparative Analysis

BERT-based embeddings outperform random embeddings due to the availability of pretrained semantic representations before training.

## Compute Resources

* 1 × NVIDIA A100 (80GB)

## Evaluation Metrics

* BLEU Score
* ChRF++ Score


## Future Improvements

* Incorporate greedy decoding 
* Using larger training corpus
* Hyperparameter optimization
