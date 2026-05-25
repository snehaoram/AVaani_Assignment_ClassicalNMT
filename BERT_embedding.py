import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("path_to_model--HBert") #l3cube-pune/hindi-bert-v2
model = AutoModelForMaskedLM.from_pretrained("path_to_model--HBert") #l3cube-pune/hindi-bert-v2

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pickle

import math
import random

with open('train.hi', 'r', encoding='utf-8') as f:
    hindi_tr = [line.strip() for line in f if line.strip()]


# sample_size = math.ceil(len(hindi_tr) * 0.3)

hindi_tr_trun = hindi_tr[:72555] 
# hindi_val_trun = hindi_tr[72555:int(72555+0.1*(72555))]

def extract_and_save(sentences, filename, max_len=50):
    all_embeddings = []
    # Ensure the model is on the correct device
    model.to(device)

    for text in tqdm(sentences, desc=f"Extracting {filename}"):
        # Tokenize and encode
        inputs = tokenizer(text,
                           return_tensors='pt',
                           max_length=max_len,
                           padding='max_length',
                           truncation=True)

        # Move input tensors to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get hidden states
        # Pass inputs as keyword arguments using **inputs
        # Request output_hidden_states to get embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # last_hidden_state is outputs.hidden_states[-1]
        # Move to CPU before converting to numpy
        embeddings = outputs.hidden_states[-1].cpu().numpy()
        all_embeddings.append(embeddings[0])  # [0] to remove batch dimension

    # Save to disk as a single numpy array
    np.save(filename, np.array(all_embeddings))
    print(f"Saved to {filename}. Total shape: {np.array(all_embeddings).shape}")


source_sentences = hindi_tr_trun
extract_and_save(source_sentences, "encoder_train_embeddings.npy") #encoder_train_embeddings_V.npy