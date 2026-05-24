import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm
# import os

import random
# from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from transformers import AutoTokenizer
import sacrebleu

tokenizerH = AutoTokenizer.from_pretrained("/home/cfiltlab/23m2159/HBert") 
tokenizerM = AutoTokenizer.from_pretrained("/home/cfiltlab/23m2159/MaBert")


print("Hindi Tokenizer vocab size: " ,tokenizerH.vocab_size)
print("Marathi Tokenizer: " ,tokenizerM.vocab_size)

# ---------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

with open('train.hi', 'r', encoding='utf-8') as f:
    hindi_tr = [line.strip() for line in f if line.strip()]

hindi_tr_trun = hindi_tr[:72555]
hindi_val_trun = hindi_tr[72555:int(72555+0.1*(72555))]



with open('train.mr', 'r', encoding='utf-8') as f:
    marathi_tr = [line.strip() for line in f if line.strip()]

marathi_tr_trun = marathi_tr[:72555]
marathi_val_trun = marathi_tr[72555:int(72555+0.1*(72555))]

# ------------------------------------------------------------------------
# Build the 10k vocab from the most frequent tokens in training data.
# ------------------------------------------------------------------------

from collections import Counter

MAX_VOCAB_SIZE = 10000

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
SOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

# ------------------------------------------------------------------------
# Build Hindi and Marathi vocab mappings separately:
# ------------------------------------------------------------------------


def build_limited_vocab(sentences, tokenizer, max_vocab_size=10000):
    counter = Counter()

    for sent in sentences:
        ids = tokenizer(
            sent,
            add_special_tokens=False,
            truncation=True,
            max_length=50
        )["input_ids"]

        counter.update(ids)

    special_old_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    }

    old_to_new = {}

    old_to_new[tokenizer.pad_token_id] = PAD_TOKEN_ID
    old_to_new[tokenizer.unk_token_id] = UNK_TOKEN_ID
    old_to_new[tokenizer.cls_token_id] = SOS_TOKEN_ID
    old_to_new[tokenizer.sep_token_id] = EOS_TOKEN_ID

    next_id = 4

    for old_id, freq in counter.most_common():
        if old_id in special_old_ids:
            continue

        if next_id >= max_vocab_size:
            break

        old_to_new[old_id] = next_id
        next_id += 1

    new_to_old = {v: k for k, v in old_to_new.items()}

    return old_to_new, new_to_old


#Then create mappings

hi_old_to_new, hi_new_to_old = build_limited_vocab(
    hindi_tr_trun,
    tokenizerH,
    max_vocab_size=10000
)

mr_old_to_new, mr_new_to_old = build_limited_vocab(
    marathi_tr_trun,
    tokenizerM,
    max_vocab_size=10000
)

# ------------------------------------------------------------------------
# Tokenize and remap:
# ------------------------------------------------------------------------

def tokenize_and_remap(sentences, tokenizer, old_to_new, max_len=50):
    all_ids = []

    for sent in sentences:
        ids = tokenizer(
            sent,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len - 2
        )["input_ids"]

        remapped = [SOS_TOKEN_ID]

        for old_id in ids:
            remapped.append(old_to_new.get(old_id, UNK_TOKEN_ID))

        remapped.append(EOS_TOKEN_ID)

        if len(remapped) < max_len:
            remapped += [PAD_TOKEN_ID] * (max_len - len(remapped))
        else:
            remapped = remapped[:max_len]
            remapped[-1] = EOS_TOKEN_ID

        all_ids.append(remapped)

    return torch.tensor(all_ids, dtype=torch.long)

inputsH_tr_ids = tokenize_and_remap(
    hindi_tr_trun,
    tokenizerH,
    hi_old_to_new,
    max_len=50
)

inputsH_vl_ids = tokenize_and_remap(
    hindi_val_trun,
    tokenizerH,
    hi_old_to_new,
    max_len=50
)

inputsM_tr_ids = tokenize_and_remap(
    marathi_tr_trun,
    tokenizerM,
    mr_old_to_new,
    max_len=50
)

inputsM_vl_ids = tokenize_and_remap(
    marathi_val_trun,
    tokenizerM,
    mr_old_to_new,
    max_len=50
)

train_dataset = TensorDataset(
    inputsH_tr_ids,
    inputsM_tr_ids
)

val_dataset = TensorDataset(
    inputsH_vl_ids,
    inputsM_vl_ids
)


# -----------------------------
# Config
# -----------------------------

# SRC_VOCAB_SIZE = tokenizerH.vocab_size
# TGT_VOCAB_SIZE = tokenizerM.vocab_size

VOCAB_SIZE = 10000
ENCODER_INPUT_DIM = 768
HIDDEN_DIM = 256
NUM_HEADS = 8
EMBEDDING_DIM = 256

BATCH_SIZE = 32
EPOCHS = 20 #10
LR = 1e-3
TEACHER_FORCING_RATIO = 0.7


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    # shuffle=False
)


# --------------------------------------------------------------------
# Model
# --------------------------------------------------------------------

class Seq2SeqLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=PAD_TOKEN_ID)

        self.encoder_lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
        )

        self.decoder_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=PAD_TOKEN_ID)

        self.decoder_lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            batch_first=True,
        )

        # self.fc = nn.Linear(HIDDEN_DIM * 2, VOCAB_SIZE)
        self.fc = nn.Linear(HIDDEN_DIM * 2, VOCAB_SIZE)

    def encode(self, encoder_input_ids):
        # encoder_input_ids:
        # [batch_size, seq_len]

        embedded = self.encoder_embedding(encoder_input_ids) # embedded: [batch_size, seq_len, EMBEDDING_DIM]      

        encoder_outputs, hidden = self.encoder_lstm(embedded) # encoder_outputs: [batch_size, seq_len, HIDDEN_DIM]; # hidden: (h_n, c_n)

        return encoder_outputs, hidden
    

    def decode_step(self, input_token, hidden, encoder_outputs):
        # input_token:
        # [batch_size]
        # OR
        # [batch_size, 1]

        # Ensure proper shape
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1) # input_token: [batch_size, 1]

        embedded = self.decoder_embedding(input_token) # embedded: [batch_size, 1, EMBEDDING_DIM]

        decoder_output, hidden = self.decoder_lstm(
            embedded,
            hidden,
        ) # decoder_output: [batch_size, 1, HIDDEN_DIM]

        attn_out, attn_weights = self.attention(
            query=decoder_output,
            key=encoder_outputs,
            value=encoder_outputs,
            # key_padding_mask=key_padding_mask
        ) # attn_out: [batch_size, 1, HIDDEN_DIM]

        combined = torch.cat(
            [decoder_output, attn_out],
            dim=-1
        ) # combined: [batch_size, 1, HIDDEN_DIM * 2]

        logits = self.fc(combined) # logits: [batch_size, 1, VOCAB_SIZE]

        return logits, hidden

    def forward(self, encoder_input_ids, target_tokens, teacher_forcing_ratio=0.7): 

        # encoder_input_ids:
        # [batch_size, src_seq_len]

        # target_tokens:
        # [batch_size, tgt_seq_len]

        batch_size, target_len = target_tokens.shape

        # Encode source sentence
        encoder_outputs, hidden = self.encode(
            encoder_input_ids
        )

        # Store decoder outputs
        outputs = torch.zeros(
            batch_size,
            target_len - 1,
            VOCAB_SIZE, #TGT_VOCAB_SIZE, VOCAB_SIZE
            device=encoder_input_ids.device,
        )

        # First decoder input = <SOS>
        input_token = target_tokens[:, 0] # shape: [batch_size]

        # Decoder loop
        for t in range(1, target_len):

            logits, hidden = self.decode_step(
                input_token,
                hidden,
                encoder_outputs,
            ) # logits: [batch_size, 1, VOCAB_SIZE]

            outputs[:, t - 1, :] = logits.squeeze(1)

            # Predicted token
            predicted_token = logits.argmax(dim=-1).squeeze(1) # predicted_token: [batch_size]

            # Teacher forcing
            use_teacher_forcing = (
                random.random() < teacher_forcing_ratio
            )

            if use_teacher_forcing:

                input_token = target_tokens[:, t]

            else:

                input_token = predicted_token

        return outputs


    @torch.no_grad()
    def beam_search(self, encoder_input_ids, beam_width=5, max_len=50):

        self.eval()

        # Add batch dimension
        encoder_input_ids = (
            encoder_input_ids
            .unsqueeze(0)
            .to(device)
        ) # shape: [1, seq_len]

        # Encode source sentence
        encoder_outputs, hidden = self.encode(encoder_input_ids)

        # Initialize beam
        beams = [
            {
                "tokens": [SOS_TOKEN_ID],
                "score": 0.0,
                "hidden": hidden,
            }
        ]

        completed = []

        # Decoding loop
        for _ in range(max_len):

            new_beams = []

            for beam in beams:

                last_token = beam["tokens"][-1]
                # Stop if EOS reached
                if last_token == EOS_TOKEN_ID:
                    completed.append(beam)
                    continue

                # Prepare decoder input
                input_token = torch.tensor([last_token], dtype=torch.long, device=device) # shape: [1]

                # Decode one step
                logits, new_hidden = self.decode_step(input_token, beam["hidden"], encoder_outputs)  # logits: [1, 1, VOCAB_SIZE]

                # Convert to log probabilities
                log_probs = torch.log_softmax(logits.squeeze(1), dim=-1) # shape: [1, VOCAB_SIZE]

                # Top beam_width candidates
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

                # Expand beams
                for i in range(beam_width):

                    token_id = top_indices[0, i].item()

                    token_score = (top_log_probs[0, i].item())

                    new_beams.append(
                        {
                            "tokens":
                                beam["tokens"] + [token_id],

                            "score":
                                beam["score"] + token_score,

                            "hidden":
                                (
                                    new_hidden[0].clone(),
                                    new_hidden[1].clone(),
                                ),
                        }
                    )

            # Stop if no candidates
            if not new_beams:
                break

            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x["score"] / len(x["tokens"]), reverse=True)[:beam_width]

        # Add unfinished beams
        completed.extend(beams)

        # Select best beam
        best = sorted(completed, key=lambda x: x["score"] / len(x["tokens"]), reverse=True)[0]

        return best["tokens"]




model = Seq2SeqLSTMAttention().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_TOKEN_ID,
)

# -----------------------------
# BLEU and CHRF++
# -----------------------------


with open('train.mr', 'r', encoding='utf-8') as f:
    marathi_tr = [line.strip() for line in f if line.strip()]


y_train_ref = marathi_tr[:72555]
y_val_ref = marathi_tr[72555:int(72555+0.1*(72555))]

def bleu_chrf(hypotheses, references):
    """
    Example hypotheses = ["The cat sat on the mat.", "There is a dog outside."]
    references = [["The cat sat on the mat.", "A dog is outside."]]
    """
    bleu_scorer = sacrebleu.corpus_bleu(hypotheses, references)
    bleu_score = bleu_scorer.score

    # 2. Calculate chrF++ score (0 to 100 scale)
    chrf_scorer = sacrebleu.corpus_chrf(hypotheses, references, word_order=2)
    chrf_score = chrf_scorer.score

    # Print results
    # print(f"BLEU Score:   {bleu_score:.2f}")
    # print(f"chrF++ Score: {chrf_score:.2f}")

    # return f"BLEU Score:   {bleu_score:.2f}", f"chrF++ Score: {chrf_score:.2f}"
    return f"{bleu_score:.2f}", f"{chrf_score:.2f}"


# -----------------------------------------------------------------
# decoding generated Marathi tokens back to text, map new IDs back to old tokenizer IDs:
# -----------------------------------------------------------------

def decode_remapped(token_ids, tokenizer, new_to_old):
    old_ids = []

    for new_id in token_ids:
        if new_id == EOS_TOKEN_ID:
            break

        if new_id in [PAD_TOKEN_ID, SOS_TOKEN_ID]:
            continue

        old_id = new_to_old.get(new_id, tokenizer.unk_token_id)
        old_ids.append(old_id)

    return tokenizer.decode(old_ids, skip_special_tokens=True)

# -----------------------------------------------------------------
# Compute metrics
# -----------------------------------------------------------------

@torch.no_grad()
def compute_metrics(data_loader, references):
    model.eval()
    hypotheses = []

    for i, (encoder_batch, _) in enumerate(data_loader):
        for j in range(encoder_batch.shape[0]):
            global_idx = i * BATCH_SIZE + j

            if global_idx >= len(references):
                break

            enc_input = encoder_batch[j].to(device)

            predicted_ids = model.beam_search(
                enc_input,
                beam_width=5,
                max_len=50
            )

            pred_text = decode_remapped(
                predicted_ids,
                tokenizerM,
                mr_new_to_old
            )

            hypotheses.append(pred_text)

    references = references[:len(hypotheses)]

    bleu_score, chrf_score = bleu_chrf(
        hypotheses,
        [references]
    )

    return bleu_score, chrf_score

# -----------------------------------------------------------------
# Training
# -----------------------------------------------------------------

def train_one_epoch():
    model.train()
    total_loss = 0.0
        
    for encoder_input_ids, target_input_ids in train_loader:
        encoder_input_ids = encoder_input_ids.to(device)
        target_input_ids = target_input_ids.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(
            encoder_input_ids,
            target_input_ids,
            teacher_forcing_ratio=
                TEACHER_FORCING_RATIO,
        ) # logits: [batch, tgt_len-1, VOCAB_SIZE]

        # Shift target
        target_output = target_input_ids[:, 1:] # shape:[batch, tgt_len-1]

        # Compute loss

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE), #VOCAB_SIZE
            target_output.reshape(-1),
        )
        # Backpropagation
        loss.backward()

        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0
    
    for encoder_input_ids, target_input_ids in val_loader:
        encoder_input_ids = encoder_input_ids.to(device)
        target_input_ids = target_input_ids.to(device)

        # Forward pass
        logits = model(
            encoder_input_ids,
            target_input_ids,
            teacher_forcing_ratio=0.0,
        )

        # Shift target
        target_output = target_input_ids[:, 1:]

        # Compute loss
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE), #VOCAB_SIZE
            target_output.reshape(-1),
        )

        total_loss += loss.item()

    return total_loss / len(val_loader)



train_losses = []
val_losses = []
# bleu_scores = []      # rename from bleu to avoid shadowing the function
# chrf_scores = [] 

best_val_loss = float("inf")
best_model_path = "best_seq2seq_lstm_attention_RAN.pt"
final_model_path = "final_seq2seq_lstm_attention_RAN.pt"


# EVAL_EVERY = 2  # compute BLEU/chrF++ every 2 epochs to save time

# for epoch in range(EPOCHS):
#     train_loss = train_one_epoch()
#     val_loss = validate()

#     train_losses.append(train_loss)
#     val_losses.append(val_loss)

#     # --- BLEU / chrF++ on val set ---
#     if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
#         bleu_score, chrf_score = compute_metrics(val_loader, y_val_ref)
#         bleu_scores.append((epoch + 1, bleu_score))
#         chrf_scores.append((epoch + 1, chrf_score))
#         print(
#             f"Epoch {epoch + 1}/{EPOCHS} | "
#             f"Train Loss: {train_loss:.4f} | "
#             f"Val Loss: {val_loss:.4f} | "
#             f"{bleu_score} | {chrf_score}"
#         )
#     else:
#         print(
#             f"Epoch {epoch + 1}/{EPOCHS} | "
#             f"Train Loss: {train_loss:.4f} | "
#             f"Val Loss: {val_loss:.4f}"
#         )
EVAL_EVERY = 2  # compute BLEU/chrF++ every 2 epochs to save time

train_bleu_scores = []
train_chrf_scores = []
val_bleu_scores = []
val_chrf_scores = []

for epoch in range(EPOCHS):
    train_loss = train_one_epoch()
    val_loss = validate()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # --- BLEU / chrF++ on train and val sets ---
    if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
        train_bleu_score, train_chrf_score = compute_metrics(
            train_loader,
            y_train_ref
        )

        val_bleu_score, val_chrf_score = compute_metrics(
            val_loader,
            y_val_ref
        )

        train_bleu_scores.append((epoch + 1, train_bleu_score))
        train_chrf_scores.append((epoch + 1, train_chrf_score))

        val_bleu_scores.append((epoch + 1, val_bleu_score))
        val_chrf_scores.append((epoch + 1, val_chrf_score))

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train BLEU: {train_bleu_score} | "
            f"Train chrF++: {train_chrf_score} | "
            f"Val BLEU: {val_bleu_score} | "
            f"Val chrF++: {val_chrf_score}"
        )

    else:
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                # "bleu_scores": bleu_scores,    # ← store with checkpoint
                # "chrf_scores": chrf_scores,
                "train_bleu_scores": train_bleu_scores,
                "train_chrf_scores": train_chrf_scores,
                "val_bleu_scores": val_bleu_scores,
                "val_chrf_scores": val_chrf_scores,
            },
            best_model_path,
        )


# Save final model after all epochs
torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        # "bleu_scores": bleu_scores,    # ← store with checkpoint
        # "chrf_scores": chrf_scores,
        "train_bleu_scores": train_bleu_scores,
        "train_chrf_scores": train_chrf_scores,
        "val_bleu_scores": val_bleu_scores,
        "val_chrf_scores": val_chrf_scores,
    },
    final_model_path,
)

print(f"Best model saved to: {best_model_path}")
print(f"Final model saved to: {final_model_path}")





# -----------------------------
# Beam search example
# -----------------------------

# sample_encoder_input = x_val[0]
# predicted_token_ids = model.beam_search(
#     sample_encoder_input,
#     beam_width=5,
#     max_len=50,
# )

# print(predicted_token_ids)




# x_train_enc = np.load("encoder_train_embeddings.npy")
# y_train = np.load("encoder_train_embeddings_MR.npy")

# x_val = np.load("encoder_train_embeddings_V.npy")
# y_val = np.load("encoder_train_embeddings_MR_V.npy")

# # If y_train is one-hot encoded, convert it to token ids
# if y_train.ndim == 3:
#     y_train = np.argmax(y_train, axis=-1)

# # x_train, x_val, y_train, y_val = train_test_split(
# #     x_train_enc,
# #     y_train,
# #     test_size=0.10,
# #     random_state=42,
# #     shuffle=True,
# # )

# x_train = x_train_enc

# def __init__(self):
#         super().__init__()

#         self.encoder_lstm = nn.LSTM(
#             input_size=ENCODER_INPUT_DIM,
#             hidden_size=HIDDEN_DIM,
#             batch_first=True,
#         )

#         self.decoder_embedding = nn.Embedding(
#             num_embeddings=VOCAB_SIZE,
#             embedding_dim=EMBEDDING_DIM,  #VOCAB_SIZE
#             padding_idx=PAD_TOKEN_ID,
#         )

#         self.decoder_lstm = nn.LSTM(
#             input_size=EMBEDDING_DIM,
#             hidden_size=HIDDEN_DIM,  #VOCAB_SIZE
#             batch_first=True,
#         )

#         self.attention = nn.MultiheadAttention(
#             embed_dim=HIDDEN_DIM,
#             num_heads=NUM_HEADS,
#             batch_first=True,
#         )

#         self.fc = nn.Linear(HIDDEN_DIM * 2, VOCAB_SIZE)


# print("inputsH_tr: ", inputsH_tr.min(), inputsH_tr.max())
# print("inputsH_vl: ", inputsH_vl.min(), inputsH_vl.max())
# print("inputsM_tr: ", inputsM_tr.min(), inputsM_tr.max())
# print("inputsM_vl: ", inputsM_vl.min(), inputsM_vl.max())

# inputsH_tr = extract_and_saveH(hindi_tr_trun, "x_trainH.json") 
# inputsH_vl = extract_and_saveH(hindi_val_trun, "x_valH.json") 

# inputsM_tr = extract_and_saveM(marathi_tr_trun, "y_trainM.json")
# inputsM_vl = extract_and_saveM(marathi_val_trun, "y_valM.json")


# def encode(self, encoder_inputs):
#         encoder_outputs, hidden = self.encoder_lstm(encoder_inputs)
#         return encoder_outputs, hidden

#     def decode_step(self, input_token, hidden, encoder_outputs):
#         embedded = self.decoder_embedding(input_token)

#         decoder_output, hidden = self.decoder_lstm(
#             embedded,
#             hidden,
#         )

#         attn_out, _ = self.attention(
#             query=decoder_output,
#             key=encoder_outputs,
#             value=encoder_outputs,
#         )

#         combined = torch.cat([decoder_output, attn_out], dim=-1)
#         logits = self.fc(combined)

#         return logits, hidden

# def compute_metrics(data_loader, references):
#     model.eval()
#     hypotheses = []

#     for i, (encoder_batch, _) in enumerate(data_loader):
#         for j in range(encoder_batch.shape[0]):
#             global_idx = i * BATCH_SIZE + j
#             if global_idx >= len(references):
#                 break
#             enc_input = encoder_batch[j].to(device)
#             token_ids = model.beam_search(enc_input, beam_width=5, max_len=50)
#             # hypotheses.append(decode_tokens(token_ids))
#             hypotheses.append(decode_remapped(token_ids, tokenizerM, mr_new_to_old))

#     bleu_score, chrf_score = bleu_chrf(hypotheses, [references[:len(hypotheses)]])
#     return bleu_score, chrf_score

# x_train = torch.tensor(x_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)

# for epoch in range(EPOCHS):
#     train_loss = train_one_epoch()
#     val_loss = validate()

#     print(
#         f"Epoch {epoch + 1}/{EPOCHS} | "
#         f"Train Loss: {train_loss:.4f} | "
#         f"Val Loss: {val_loss:.4f}"
#     )

# x_val = torch.tensor(x_val, dtype=torch.float32)
# y_val = torch.tensor(y_val, dtype=torch.long)

# train_loader = DataLoader(
#     TensorDataset(x_train, y_train),
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )

# val_loader = DataLoader(
#     TensorDataset(x_val, y_val),
#     batch_size=BATCH_SIZE,
#     shuffle=False,
# )


# def decode_remapped(token_ids, tokenizer, new_to_old):
#     old_ids = []

#     for new_id in token_ids:
#         if new_id in [PAD_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID]:
#             continue

#         old_id = new_to_old.get(new_id, tokenizer.unk_token_id)
#         old_ids.append(old_id)

#     return tokenizer.decode(old_ids, skip_special_tokens=True)

# def extract_and_saveH(sentences, filename, max_len=50):
#     all_embeddings = []
#     # Ensure the model is on the correct device
#     model.to(device)

#     for text in tqdm(sentences, desc=f"Extracting {filename}"):
#         # Tokenize and encode
#         inputs = tokenizerH(text,
#                            return_tensors='pt',
#                            max_length=max_len,
#                            padding='max_length',
#                            truncation=True)

#         # Move input tensors to the same device as the model
#         inputsH = {k: v.to(device) for k, v in inputs.items()}

#     return inputsH


# word2id = {}

# with open("vocab_mr.txt", "r", encoding="utf-8") as f:
#     for idx, word in enumerate(f):
#         word = word.strip()
#         word2id[word] = idx

# id2word = {v: k for k, v in word2id.items()}
# id2word = {idx: word for word, idx in word2id.items()}

# def decode_tokens(token_ids):
#     words = []
#     for tid in token_ids:
#         if tid == EOS_TOKEN_ID:
#             break
#         if tid in (SOS_TOKEN_ID, PAD_TOKEN_ID):
#             continue
#         words.append(id2word.get(tid, "<unk>"))
#     return " ".join(words)

# def extract_and_saveM(sentences, filename, max_len=50):
#     all_embeddings = []
#     # Ensure the model is on the correct device
#     model.to(device)

#     for text in tqdm(sentences, desc=f"Extracting {filename}"):
#         # Tokenize and encode
#         inputs = tokenizerM(text,
#                            return_tensors='pt',
#                            max_length=max_len,
#                            padding='max_length',
#                            truncation=True)

#         # Move input tensors to the same device as the model
#         inputsM = {k: v.to(device) for k, v in inputs.items()}
    
#     return inputsM

# class Seq2SeqLSTMAttention(nn.Module):
#     def __init__(
#         self,
#         encoder_input_dim=768,
#         decoder_input_dim=5000,
#         hidden_dim=256,
#         target_vocab_size=5000,
#         num_heads=8,
#     ):
#         super().__init__()

#         self.encoder_lstm = nn.LSTM(
#             input_size=encoder_input_dim,
#             hidden_size=hidden_dim,
#             batch_first=True,
#         )

#         self.decoder_lstm = nn.LSTM(
#             input_size=decoder_input_dim,
#             hidden_size=hidden_dim,
#             batch_first=True,
#         )

#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             batch_first=True,
#         )

#         self.fc = nn.Linear(hidden_dim * 2, target_vocab_size)

#     def forward(self, encoder_inputs, decoder_inputs):
#         encoder_outputs, (state_h, state_c) = self.encoder_lstm(encoder_inputs)

#         decoder_outputs, _ = self.decoder_lstm(
#             decoder_inputs,
#             (state_h, state_c),
#         )


# def tokenize_hindi(sentences, max_len=50):

#     encodings = tokenizerH(
#         sentences,
#         padding='max_length',
#         truncation=True,
#         max_length=max_len,
#         return_tensors='pt'
#     )

#     return encodings


# def tokenize_marathi(sentences, max_len=50):

#     encodings = tokenizerM(
#         sentences,
#         padding='max_length',
#         truncation=True,
#         max_length=max_len,
#         return_tensors='pt'
#     )

#     return encodings


# inputsH_tr = tokenize_hindi(hindi_tr_trun)
# inputsH_vl = tokenize_hindi(hindi_val_trun)

# inputsM_tr = tokenize_marathi(marathi_tr_trun)
# inputsM_vl = tokenize_marathi(marathi_val_trun)


# train_dataset = TensorDataset(
#     inputsH_tr['input_ids'],
#     inputsH_tr['attention_mask'],
#     inputsM_tr['input_ids'],
#     inputsM_tr['attention_mask']
# )

# val_dataset = TensorDataset(
#     inputsH_vl['input_ids'],
#     inputsH_vl['attention_mask'],
#     inputsM_vl['input_ids'],
#     inputsM_vl['attention_mask']
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )
#         attn_out, _ = self.attention(
#             query=decoder_outputs,
#             key=encoder_outputs,
#             value=encoder_outputs,
#         )

#         concat = torch.cat([decoder_outputs, attn_out], dim=-1)

#         logits = self.fc(concat)
#         return logits


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Seq2SeqLSTMAttention().to(device)

# optimizer = optim.Adam(model.parameters())

# # PyTorch CrossEntropyLoss expects raw logits, not softmax output.
# criterion = nn.CrossEntropyLoss()

# x_train_enc = torch.tensor(x_train_enc, dtype=torch.float32).to(device)
# y_train_dec_input = torch.tensor(y_train_dec_input, dtype=torch.float32).to(device)

# print(model)



# for epoch in range(EPOCHS):
#     train_loss = train_one_epoch()
#     val_loss = validate()

#     train_losses.append(train_loss)
#     val_losses.append(val_loss)

#     print(
#         f"Epoch {epoch + 1}/{EPOCHS} | "
#         f"Train Loss: {train_loss:.4f} | "
#         f"Val Loss: {val_loss:.4f}"
#     )

#     # Save best model based on validation loss
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss

#         torch.save(
#             {
#                 "epoch": epoch + 1,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "train_loss": train_loss,
#                 "val_loss": val_loss,
#                 "train_losses": train_losses,
#                 "val_losses": val_losses,
#             },
#             best_model_path,
#         )
        
#     break

# # Save final model after all epochs
# torch.save(
#     {
#         "epoch": EPOCHS,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "train_losses": train_losses,
#         "val_losses": val_losses,
#         "best_val_loss": best_val_loss,
#     },
#     final_model_path,
# )

# print(f"Best model saved to: {best_model_path}")
# print(f"Final model saved to: {final_model_path}")




# # import tensorflow as tf
# # from tensorflow.keras.layers import Input, LSTM, Dense, MultiHeadAttention, LayerNormalization
# # from tensorflow.keras.models import Model
# # import numpy as np

# # # Load the data we saved in Step 1
# # x_train_enc = np.load("encoder_train_embeddings.npy")
# # # (Assume y_train_dec is your target language data, one-hot encoded)

# # y_train_dec_input = np.load("encoder_train_embeddings_MR.npy")

# # # --- ENCODER (Using Static BERT Embeddings) ---
# # # Shape: (max_seq_len, 768)
# # encoder_inputs = Input(shape=(95, 768))  #50
# # encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
# # encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# # # --- DECODER + ATTENTION ---
# # decoder_inputs = Input(shape=(None, 5000)) # 5000 = target vocab size
# # decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
# # decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])

# # # Cross-Attention: Decoder queries the Encoder outputs
# # attention = MultiHeadAttention(num_heads=8, key_dim=256)
# # attn_out = attention(query=decoder_outputs, value=encoder_outputs)

# # # Combine and Predict
# # concat = tf.keras.layers.Concatenate()([decoder_outputs, attn_out])
# # dense_out = Dense(5000, activation='softmax')(concat)

# # model = Model([encoder_inputs, decoder_inputs], dense_out)
# # model.compile(optimizer='adam', loss='categorical_crossentropy')

# # print(model.compile(optimizer='adam', loss='categorical_crossentropy'))

# # Train with the pre-saved BERT vectors!
# # model.fit([x_train_enc, y_train_dec_input], y_train_dec_target, ...)