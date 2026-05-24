import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import random
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import sacrebleu
from transformers import AutoTokenizer
from collections import Counter

# tokenizer = AutoTokenizer.from_pretrained("/home/cfiltlab/23m2159/HBert")
tokenizerH = AutoTokenizer.from_pretrained("/home/cfiltlab/23m2159/HBert") 
tokenizerM = AutoTokenizer.from_pretrained("/home/cfiltlab/23m2159/MaBert")

# src_pad_mask

# -----------------------------
# Config
# -----------------------------

#5000tokenizer.vocab_size
ENCODER_INPUT_DIM = 768
HIDDEN_DIM = 256
NUM_HEADS = 8
EMBEDDING_DIM = 256

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
TEACHER_FORCING_RATIO = 0.7


VOCAB_SIZE = 10000  # or 5000
PAD_TOKEN_ID = 0
SOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load data
# -----------------------------

MAX_TARGET_LEN = 50
TRAIN_SIZE = 72555
VAL_SIZE = int(0.1 * TRAIN_SIZE)

with open("train.mr", "r", encoding="utf-8") as f:
    marathi_all = [line.strip() for line in f if line.strip()]

marathi_tr_trun = marathi_all[:TRAIN_SIZE]
marathi_val_trun = marathi_all[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]


def build_limited_vocab(sentences, tokenizer, max_vocab_size=10000):
    counter = Counter()

    for sent in sentences:
        ids = tokenizer(
            sent,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_TARGET_LEN - 2,
        )["input_ids"]

        counter.update(ids)

    old_to_new = {}

    special_pairs = [
        (tokenizer.pad_token_id, PAD_TOKEN_ID),
        (tokenizer.cls_token_id, SOS_TOKEN_ID),
        (tokenizer.sep_token_id, EOS_TOKEN_ID),
        (tokenizer.unk_token_id, UNK_TOKEN_ID),
    ]

    for old_id, new_id in special_pairs:
        if old_id is not None:
            old_to_new[old_id] = new_id

    next_id = 4

    for old_id, _ in counter.most_common():
        if old_id in old_to_new:
            continue

        if next_id >= max_vocab_size:
            break

        old_to_new[old_id] = next_id
        next_id += 1

    new_to_old = {new_id: old_id for old_id, new_id in old_to_new.items()}

    print(f"Reduced Marathi vocab size: {len(old_to_new)}")

    return old_to_new, new_to_old


def tokenize_and_remap(sentences, tokenizer, old_to_new, max_len=50):
    all_ids = []

    for sent in sentences:
        old_ids = tokenizer(
            sent,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len - 2,
        )["input_ids"]

        remapped = [SOS_TOKEN_ID]

        for old_id in old_ids:
            remapped.append(old_to_new.get(old_id, UNK_TOKEN_ID))

        remapped.append(EOS_TOKEN_ID)

        if len(remapped) < max_len:
            remapped += [PAD_TOKEN_ID] * (max_len - len(remapped))
        else:
            remapped = remapped[:max_len]
            remapped[-1] = EOS_TOKEN_ID

        all_ids.append(remapped)

    return np.array(all_ids, dtype=np.int64)


mr_old_to_new, mr_new_to_old = build_limited_vocab(
    marathi_tr_trun,
    tokenizerM,
    max_vocab_size=VOCAB_SIZE,
)

ORIG_TO_SMALL = mr_old_to_new
SMALL_TO_ORIG = mr_new_to_old

y_train = tokenize_and_remap(
    marathi_tr_trun,
    tokenizerM,
    ORIG_TO_SMALL,
    max_len=MAX_TARGET_LEN,
)

y_val = tokenize_and_remap(
    marathi_val_trun,
    tokenizerM,
    ORIG_TO_SMALL,
    max_len=MAX_TARGET_LEN,
)

x_train = np.load("encoder_train_embeddings.npy")
x_val = np.load("encoder_train_embeddings_V.npy")

assert len(x_train) == len(y_train), f"x_train={len(x_train)}, y_train={len(y_train)}"
assert len(x_val) == len(y_val), f"x_val={len(x_val)}, y_val={len(y_val)}"

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(x_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# -----------------------------
# Model
# -----------------------------

class Seq2SeqLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=ENCODER_INPUT_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
        )

        self.decoder_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=PAD_TOKEN_ID,
        )

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

        self.fc = nn.Linear(HIDDEN_DIM * 2, VOCAB_SIZE)
        

    def encode(self, encoder_inputs):
        encoder_outputs, hidden = self.encoder_lstm(encoder_inputs)
        return encoder_outputs, hidden

    def decode_step(self, input_token, hidden, encoder_outputs):
        embedded = self.decoder_embedding(input_token)

        decoder_output, hidden = self.decoder_lstm(
            embedded,
            hidden,
        )

        
        attn_out, _ = self.attention(
            query=decoder_output,
            key=encoder_outputs,
            value=encoder_outputs,
            # key_padding_mask=src_pad_mask,
        )

        combined = torch.cat([decoder_output, attn_out], dim=-1)
        logits = self.fc(combined)

        return logits, hidden

    def forward(self, encoder_inputs, target_tokens, teacher_forcing_ratio=0.7):
        batch_size, target_len = target_tokens.shape

        encoder_outputs, hidden = self.encode(encoder_inputs)

        outputs = torch.zeros(
            batch_size,
            target_len - 1,
            VOCAB_SIZE,
            device=encoder_inputs.device,
        )

        input_token = target_tokens[:, 0].unsqueeze(1)

        for t in range(1, target_len):
            logits, hidden = self.decode_step(
                input_token,
                hidden,
                encoder_outputs,
            )

            outputs[:, t - 1, :] = logits.squeeze(1)

            predicted_token = logits.argmax(dim=-1)

            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing:
                input_token = target_tokens[:, t].unsqueeze(1)
            else:
                input_token = predicted_token

        return outputs

    @torch.no_grad()
    def beam_search(self, encoder_input, beam_width=5, max_len=50):
        self.eval()

        encoder_input = encoder_input.unsqueeze(0).to(device)
        encoder_outputs, hidden = self.encode(encoder_input)

        beams = [
            {
                "tokens": [SOS_TOKEN_ID],
                "score": 0.0,
                "hidden": hidden,
            }
        ]

        completed = []

        for _ in range(max_len):
            new_beams = []

            for beam in beams:
                last_token = beam["tokens"][-1]

                if last_token == EOS_TOKEN_ID:
                    completed.append(beam)
                    continue

                input_token = torch.tensor(
                    [[last_token]],
                    dtype=torch.long,
                    device=device,
                )

                logits, new_hidden = self.decode_step(
                    input_token,
                    beam["hidden"],
                    encoder_outputs,
                )

                log_probs = torch.log_softmax(logits.squeeze(1), dim=-1)
                top_log_probs, top_indices = torch.topk(
                    log_probs,
                    beam_width,
                    dim=-1,
                )

                for i in range(beam_width):
                    token_id = top_indices[0, i].item()
                    token_score = top_log_probs[0, i].item()

                    new_beams.append(
                        {
                            "tokens": beam["tokens"] + [token_id],
                            "score": beam["score"] + token_score,
                            # "hidden": new_hidden,
                            "hidden": (new_hidden[0].clone(), new_hidden[1].clone())
                        }
                    )

            if not new_beams:
                break

            beams = sorted(
                new_beams,
                key=lambda x: x["score"] / len(x["tokens"]),
                reverse=True,
            )[:beam_width]

        completed.extend(beams)

        best = sorted(
            completed,
            key=lambda x: x["score"] / len(x["tokens"]),
            reverse=True,
        )[0]

        return best["tokens"]


# -----------------------------
# BLEU and CHRF++
# -----------------------------

def decode_tokens(token_ids):
    original_ids = []

    for small_id in token_ids:
        if small_id == EOS_TOKEN_ID:
            break
        if small_id in (SOS_TOKEN_ID, PAD_TOKEN_ID):
            continue

        orig_id = SMALL_TO_ORIG.get(small_id, tokenizerM.unk_token_id)
        original_ids.append(orig_id)

    return tokenizerM.decode(original_ids, skip_special_tokens=True)

# with open('train.mr', 'r', encoding='utf-8') as f:
#     marathi_tr = [line.strip() for line in f if line.strip()]


# y_train_ref = marathi_tr[:72555]
# y_val_ref = marathi_tr[72555:int(72555+0.1*(72555))]

y_train_ref = marathi_tr_trun
y_val_ref = marathi_val_trun

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

    return f"{bleu_score:.2f}", f"{chrf_score:.2f}"

def compute_metrics(data_loader, references):
    model.eval()
    hypotheses = []

    for i, (encoder_batch, _) in enumerate(data_loader):
        for j in range(encoder_batch.shape[0]):
            global_idx = i * BATCH_SIZE + j
            if global_idx >= len(references):
                break
            enc_input = encoder_batch[j].to(device)
            token_ids = model.beam_search(enc_input, beam_width=5, max_len=50)
            hypotheses.append(decode_tokens(token_ids))

    bleu_score, chrf_score = bleu_chrf(hypotheses, [references[:len(hypotheses)]])
    return bleu_score, chrf_score

# -----------------------------
# Training
# -----------------------------

model = Seq2SeqLSTMAttention().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_TOKEN_ID,
)


def train_one_epoch():
    model.train()
    total_loss = 0.0

    for encoder_batch, target_batch in train_loader:
        encoder_batch = encoder_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()

        logits = model(
            encoder_batch,
            target_batch,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO,
        )

        target_output = target_batch[:, 1:]

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            target_output.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #new line
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0.0

    for encoder_batch, target_batch in val_loader:
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

    return total_loss / len(val_loader)



train_losses = []
val_losses = []
# bleu_scores = []      # rename from bleu to avoid shadowing the function
# chrf_scores = []      # rename from chrf

best_val_loss = float("inf")
best_model_path = "best_seq2seq_lstm_attention.pt"
final_model_path = "final_seq2seq_lstm_attention.pt"


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
#                 "bleu_scores": bleu_scores,    # ← store with checkpoint
#                 "chrf_scores": chrf_scores,
#             },
#             best_model_path,
#         )


# # Save final model after all epochs
# torch.save(
#     {
#         "epoch": EPOCHS,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "train_losses": train_losses,
#         "val_losses": val_losses,
#         "best_val_loss": best_val_loss,
#         "bleu_scores": bleu_scores,    # ← store with checkpoint
#         "chrf_scores": chrf_scores,
#     },
#     final_model_path,
# )

# print(f"Best model saved to: {best_model_path}")
# print(f"Final model saved to: {final_model_path}")


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


# hi_old_to_new, hi_new_to_old = build_limited_vocab(
#     hindi_tr_trun,
#     tokenizerH,
#     max_vocab_size=10000
# )

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

#         attn_out, _ = self.attention(
#             query=decoder_outputs,
#             key=encoder_outputs,
#             value=encoder_outputs,
#         )

#         concat = torch.cat([decoder_outputs, attn_out], dim=-1)

#         logits = self.fc(concat)
#         return logits

# self.decoder_embedding = nn.Embedding(
#             num_embeddings=VOCAB_SIZE,
#             embedding_dim=EMBEDDING_DIM,  #VOCAB_SIZE
#             padding_idx=PAD_TOKEN_ID,
#         )

# inputsM_tr_ids = tokenize_and_remap(
#     marathi_tr_trun,
#     tokenizerM,
#     mr_old_to_new,
#     max_len=50
# )

# ORIG_TO_SMALL = mr_old_to_new # {original_bert_id: small_id} #
# SMALL_TO_ORIG = mr_new_to_old # {small_id: original_bert_id}

# small_ids = [SOS_TOKEN_ID]
# small_ids = [SOS_TOKEN_ID]
# for tid in inputsM_tr_ids: #, original_token_ids
#     small_ids.append(ORIG_TO_SMALL.get(tid, UNK_TOKEN_ID))
# small_ids.append(EOS_TOKEN_ID)

# with open('train.hi', 'r', encoding='utf-8') as f:
#     hindi_tr = [line.strip() for line in f if line.strip()]

# hindi_tr_trun = hindi_tr[:72555]
# hindi_val_trun = hindi_tr[72555:int(72555+0.1*(72555))]



# with open('train.mr', 'r', encoding='utf-8') as f:
#     marathi_tr = [line.strip() for line in f if line.strip()]

# marathi_tr_trun = marathi_tr[:72555]
# marathi_val_trun = marathi_tr[72555:int(72555+0.1*(72555))]

# def build_limited_vocab(sentences, tokenizer, max_vocab_size=10000):
#     counter = Counter()

#     for sent in sentences:
#         ids = tokenizer(
#             sent,
#             add_special_tokens=False,
#             truncation=True,
#             max_length=50
#         )["input_ids"]

#         counter.update(ids)

#     special_old_ids = {
#         tokenizer.pad_token_id,
#         tokenizer.unk_token_id,
#         tokenizer.cls_token_id,
#         tokenizer.sep_token_id,
#     }

#     old_to_new = {}

#     old_to_new[tokenizer.pad_token_id] = PAD_TOKEN_ID
#     old_to_new[tokenizer.unk_token_id] = UNK_TOKEN_ID
#     old_to_new[tokenizer.cls_token_id] = SOS_TOKEN_ID
#     old_to_new[tokenizer.sep_token_id] = EOS_TOKEN_ID

#     next_id = 4

#     for old_id, freq in counter.most_common():
#         if old_id in special_old_ids:
#             continue

#         if next_id >= max_vocab_size:
#             break

#         old_to_new[old_id] = next_id
#         next_id += 1

#     new_to_old = {v: k for k, v in old_to_new.items()}

#     return old_to_new, new_to_old


# #Then create mappings

# mr_old_to_new, mr_new_to_old = build_limited_vocab(
#     marathi_tr_trun,
#     tokenizerM,
#     max_vocab_size=10000
# )

# def tokenize_and_remap(sentences, tokenizer, old_to_new, max_len=50):
#     all_ids = []

#     for sent in sentences:
#         ids = tokenizer(
#             sent,
#             add_special_tokens=False,
#             truncation=True,
#             max_length=max_len - 2
#         )["input_ids"]

#         remapped = [SOS_TOKEN_ID]

#         for old_id in ids:
#             remapped.append(old_to_new.get(old_id, UNK_TOKEN_ID))

#         remapped.append(EOS_TOKEN_ID)

#         if len(remapped) < max_len:
#             remapped += [PAD_TOKEN_ID] * (max_len - len(remapped))
#         else:
#             remapped = remapped[:max_len]
#             remapped[-1] = EOS_TOKEN_ID

#         all_ids.append(remapped)



# SOS_TOKEN_ID = 1
# EOS_TOKEN_ID = 2
# PAD_TOKEN_ID = 0

# PAD_TOKEN_ID = mr_tokenizer.pad_token_id
# SOS_TOKEN_ID = mr_tokenizer.cls_token_id
# EOS_TOKEN_ID = mr_tokenizer.sep_token_id
# UNK_TOKEN_ID = mr_tokenizer.unk_token_id
# for epoch in range(EPOCHS):
#     train_loss = train_one_epoch()
#     val_loss = validate()

#     print(
#         f"Epoch {epoch + 1}/{EPOCHS} | "
#         f"Train Loss: {train_loss:.4f} | "
#         f"Val Loss: {val_loss:.4f}"
#     )


# x_train_enc = np.load("encoder_train_embeddings.npy")
# y_train = np.load("encoder_train_embeddings_MR.npy")

# x_val = np.load("encoder_train_embeddings_V.npy")
# y_val = np.load("encoder_train_embeddings_MR_V.npy")


# for arr_name, arr in [("y_train", y_train), ("y_val", y_val)]:
#     if arr.ndim == 3:
#         if arr_name == "y_train":
#             y_train = np.argmax(arr, axis=-1)
#         else:
#             y_val = np.argmax(arr, axis=-1)


# x_train = x_train_enc
# def decode_tokens(token_ids):
#     cleaned = []
#     for tid in token_ids:
#         if tid == EOS_TOKEN_ID:
#             break
#         if tid in (SOS_TOKEN_ID, PAD_TOKEN_ID):
#             continue
#         cleaned.append(tid)
#     return mr_tokenizer.decode(cleaned, skip_special_tokens=True)


# x_train = torch.tensor(x_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)

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
#     # shuffle=False,
# )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Seq2SeqLSTMAttention().to(device)

# optimizer = optim.Adam(model.parameters())

# # PyTorch CrossEntropyLoss expects raw logits, not softmax output.
# criterion = nn.CrossEntropyLoss()

# x_train_enc = torch.tensor(x_train_enc, dtype=torch.float32).to(device)
# y_train_dec_input = torch.tensor(y_train_dec_input, dtype=torch.float32).to(device)

# print(model)

# x_train, x_val, y_train, y_val = train_test_split(
#     x_train_enc,
#     y_train,
#     test_size=0.10,
#     random_state=42,
#     shuffle=True,
# )

# attn_out, _ = self.attention(
        #     query=decoder_output,
        #     key=encoder_outputs,
        #     value=encoder_outputs,
        # )







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


# train_losses = []
# val_losses = []
# bleu_scores = []
# chrf_scores = [] 

# best_val_loss = float("inf")
# best_model_path = "best_seq2seq_lstm_attention.pt"
# final_model_path = "final_seq2seq_lstm_attention.pt"

# EVAL_EVERY = 2

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
        
    # break

    # word2id = {}

# with open("vocab_mr.txt", "r", encoding="utf-8") as f:
#     for idx, word in enumerate(f):
#         word = word.strip()
#         word2id[word] = idx

# # id2word = {v: k for k, v in word2id.items()}
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