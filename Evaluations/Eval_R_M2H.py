import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import sacrebleu

tokenizer = AutoTokenizer.from_pretrained("/home/cfiltlab/23m2159/HBert") 

tokenizerM = AutoTokenizer.from_pretrained("/home/cfiltlab/23m2159/MaBert")

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Random_M2H import decode_tokens, bleu_chrf, compute_metrics, build_limited_vocab, tokenize_and_remap

from Random_M2H import Seq2SeqLSTMAttention


with open('test.hi', 'r', encoding='utf-8') as f:
    hindi_te = [line.strip() for line in f if line.strip()]

with open('test.mr', 'r', encoding='utf-8') as f:
    marathi_te = [line.strip() for line in f if line.strip()]


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
    # np.save(filename, np.array(all_embeddings))
    arr = np.array(all_embeddings)
    print(f"Saved to {filename}. Total shape: {np.array(all_embeddings).shape}")
    return arr


source_sentences = marathi_te
target_sentence = hindi_te
# print(source_sentences)
x_test = extract_and_save(source_sentences, "encoder_train_embeddingst.npy")
y_test = extract_and_save(target_sentence, "encoder_train_embeddingst.npy")

ENCODER_INPUT_DIM = 768
HIDDEN_DIM = 256
NUM_HEADS = 8
EMBEDDING_DIM = 256

BATCH_SIZE = 32
# EPOCHS = 10
LR = 1e-3
# TEACHER_FORCING_RATIO = 0.7


VOCAB_SIZE = 10000  # or 5000
PAD_TOKEN_ID = 0
SOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3


model = Seq2SeqLSTMAttention().to(device)

# 2. Load the checkpoint dictionary
checkpoint_path = "best_seq2seq_lstm_attention_RANMH.pt" #best_seq2seq_lstm_attention
checkpoint = torch.load(checkpoint_path, map_location=device)

# 3. Load the weights into the model
model.load_state_dict(checkpoint["model_state_dict"])

# Extract metadata if needed (e.g., for logging or resume)
best_val_loss = checkpoint["best_val_loss"]
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with Val Loss: {best_val_loss:.4f}")

# mr_old_to_new, mr_new_to_old = build_limited_vocab(
#     marathi_te,
#     tokenizerM,
#     max_vocab_size=VOCAB_SIZE,
# )
criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_TOKEN_ID,
)


test_loader = DataLoader(
    TensorDataset(x_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

y_test_ref = marathi_te

@torch.no_grad()
def validate():
    model.eval()
    hypotheses = []
    total_loss = 0.0
    print("Inside valid")

    for encoder_batch, target_batch in test_loader:
        # for j in range(encoder_batch.shape[0]):
        encoder_batch = encoder_batch.to(device)
        target_batch = target_batch.to(device)

        logits = model(
            encoder_batch,
            target_batch,
            teacher_forcing_ratio=0.0,
        )

        target_output = target_batch[:, 1:]

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            target_output.reshape(-1),
        )
        total_loss += loss.item()

    
    return total_loss / len(test_loader)

        # enc_input = encoder_batch[j].to(device)
        # token_ids = model.beam_search(enc_input, beam_width=5, max_len=50)
        # hypotheses.append(decode_tokens(token_ids))

        # train_bleu_score, train_chrf_score = compute_metrics(
        #     test_loader,
        #     y_test_ref
        # )

test_loss = validate()

print("Test loss: ", test_loss)

train_bleu_score, train_chrf_score = compute_metrics(test_loader, y_test_ref)

print("Test bleu: ", train_bleu_score, "; Test chrf: ", train_chrf_score)