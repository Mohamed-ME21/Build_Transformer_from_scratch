# Transformer from Scratch (PyTorch)

A from-scratch implementation of the Transformer in PyTorch, written to understand every piece of the architecture instead of relying on `nn.Transformer`. The notebook builds two models on the same components:

- **Encoder-Decoder Transformer** for English → Arabic translation (the architecture in the diagram below).
- **GPT-style decoder-only model** for next-token prediction and text generation.

Everything runs on CPU in a few minutes. The datasets are deliberately tiny (50 sentence pairs), so this project is about **understanding the architecture and the training loop**, not about translation quality. See [Results](#results) for what that means in practice.

## Table of contents

1. [Architecture](#architecture)
2. [Repository structure](#repository-structure)
3. [Setup and quick start](#setup-and-quick-start)
4. [Code walkthrough](#code-walkthrough)
5. [Data](#data)
6. [Training](#training)
7. [Inference](#inference)
8. [Results](#results)
9. [Limitations and next steps](#limitations-and-next-steps)
10. [Saving and loading checkpoints](#saving-and-loading-checkpoints)
11. [References](#references)

---

## Architecture

The implementation follows this diagram:

![Transformer architecture](assets/Transformer_Arch.jpg)

Two things are worth noticing in the diagram, because they differ from the original paper:

- **Pre-LN.** `LayerNorm` is applied *before* each sub-layer, and the residual connection adds the un-normalized input back: `x = x + SubLayer(LayerNorm(x))`. The original 2017 paper used Post-LN. Pre-LN is generally more stable to train, which is why a plain constant learning rate (no warm-up) is enough for this small setup.
- **Final LayerNorm.** Both the encoder stack and the decoder stack end with an extra `LayerNorm` (needed with Pre-LN because the residual stream is never normalized inside the blocks).

### Diagram block → code

| Block in the diagram | Where it lives in the code |
|---|---|
| Embeddings | `nn.Embedding` (`src_emb`, `tgt_emb`), multiplied by `sqrt(d_model)` |
| Positional Encoding | `PositionalEncoding` (fixed sin/cos table) |
| Multi-Head Attention (encoder) | `MultiHeadAttention` inside `EncoderBlock`, with the padding mask |
| Masked Multi-Head Attention | `self_attn` inside `DecoderBlock`, with causal + padding mask |
| Cross Multi-Head Attention | `cross_attn` inside `DecoderBlock`: Q from the decoder, K and V from the encoder output |
| Feed-Forward Net | `FeedForward` (Linear → GELU → Dropout → Linear) |
| LayerNorm before each sub-layer | `norm1`, `norm2`, `norm_cross` |
| ⊕ residual connections | `x = x + self.drop(sublayer(...))` |
| Final LayerNorm | `enc_norm`, `dec_norm` |
| Linear | `head` (`nn.Linear(d_model, vocab)`) |
| Softmax | applied inside `F.cross_entropy` during training; replaced by `argmax` in greedy decoding |
| Output sequence (shifted right) | `tgt_in` (target tokens starting with `<BOS>`) |
| Nx | `num_layers` |

The GPT model reuses the same `DecoderBlock` with `use_cross_attn=False`, which removes the cross-attention branch and leaves the masked self-attention and feed-forward layers.

---

## Repository structure

```
.
├── Transformer_from_scratch.ipynb   # all the code: model, tokenizer, training, inference
├── toy_translation_pairs.csv        # 50 English → Arabic pairs
├── gpt_toy_texts.jsonl              # 56 text samples for the GPT model
├── requirements.txt
├── README.md
└── assets/
    ├── Transformer_Arch.jpg         # architecture diagram
    └── loss_curve.png               # translation model training curve
```

The notebook loads the two data files with relative paths, so keep them in the same folder as the notebook (or edit the paths in the notebook).

---

## Setup and quick start

**Requirements:** Python 3.9+, PyTorch, Matplotlib, Jupyter.

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook Transformer_from_scratch.ipynb
```

Run the cells from top to bottom. The notebook is order-dependent: later cells use variables defined earlier (the tokenizer, `PAD_ID`, `device`, and so on). A GPU is used automatically if available, otherwise the code falls back to CPU.

Output files created by the notebook:

- `gpt_toy_checkpoint.pt`: trained GPT weights.
- `transformer_toy_checkpoint.pt`: trained Encoder-Decoder weights.

---

## Code walkthrough

The notebook is organized in the order the pieces depend on each other.

### 1. Masks

Masks are boolean tensors where **`True` means "this position may be attended to"**. Positions marked `False` receive a score of `-inf` before the softmax, so their attention weight becomes exactly zero.

- `causal_mask(T)` returns a lower-triangular `[1, 1, T, T]` matrix so token `t` only sees tokens `0..t`. This is what stops the decoder from looking at future words.
- `padding_mask(ids)` returns a `[B, 1, 1, T]` mask that hides `<PAD>` tokens so batches of different lengths can be processed together.
- In the decoder the two are combined with `&`, giving a `[B, 1, T, T]` mask.

### 2. Positional encoding

Attention has no built-in sense of word order, so a fixed sinusoidal signal is added to the embeddings:

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

The table is computed once in `__init__` (in log space for numerical stability) and stored with `register_buffer`, so it moves with the model to the GPU and is saved in the checkpoint, but it is not a trainable parameter. `d_model` must be even because sine and cosine fill alternating dimensions.

### 3. Multi-head attention

`MultiHeadAttention(d_model, num_heads, dropout)` implements scaled dot-product attention:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- Four linear layers: `wq`, `wk`, `wv` project the inputs, `wo` projects the concatenated heads.
- `_split` reshapes `[B, T, d_model]` into `[B, h, T, d_k]` where `d_k = d_model / num_heads`, so each head works on its own slice.
- `forward(q_in, kv_in, mask)` takes the queries and the keys/values from **separate inputs**. For self-attention both are the same tensor. For cross-attention `kv_in` is the encoder output. One class covers both cases.
- Dividing by `sqrt(d_k)` keeps the scores from growing with the head size, which would push the softmax into saturated regions.
- The attention weights of the last forward pass are saved in `self.last_attn` so they can be plotted as heatmaps.

### 4. Feed-forward network

Two linear layers with a GELU in between, applied to each token independently: `d_model → d_ff → d_model`.

### 5. Encoder and decoder blocks

```python
# EncoderBlock (Pre-LN)
h = norm1(x)
x = x + dropout(self_attention(h, h, src_mask))
x = x + dropout(ffn(norm2(x)))

# DecoderBlock
h = norm1(x)
x = x + dropout(masked_self_attention(h, h, tgt_mask))
if use_cross_attn:
    h = norm_cross(x)
    x = x + dropout(cross_attention(h, enc_out, src_mask))   # Q from decoder, K/V from encoder
x = x + dropout(ffn(norm2(x)))
```

`DecoderBlock` takes a `use_cross_attn` flag: `True` for the translation model, `False` for GPT.

### 6. Weight initialization

`_init_weights` is applied to every sub-module with `model.apply(...)`:

- `nn.Linear`: weights from `N(0, 0.02)`, biases set to zero.
- `nn.Embedding`: weights from `N(0, d_model^-0.5)`.
- In the translation model the `<PAD>` embedding row is then reset to zero.

### 7. `Transformer` (Encoder-Decoder)

| Method | What it does |
|---|---|
| `encode(src)` | embeds the source, scales by `sqrt(d_model)`, adds positions, runs the encoder blocks, applies `enc_norm`. Returns the encoder output and the source padding mask. |
| `decode(tgt, enc_out, src_mask)` | builds the causal + padding mask, runs the decoder blocks with cross-attention, applies `dec_norm`, and projects to vocabulary logits `[B, T, vocab]`. |
| `forward(src, tgt_in)` | `encode` followed by `decode`. This is what training uses. |
| `greedy_decode(src, bos_id, eos_id, max_len)` | generates one token at a time, always taking the `argmax`. Encodes the source once, then loops. Sequences that already produced `<EOS>` are filled with `<PAD>`, and the loop stops early when every sequence is finished. |

### 8. `GPT` (decoder-only)

- Same blocks, no encoder and no cross-attention, one causal mask over the whole sequence.
- **Weight tying:** the output layer shares its weight matrix with the token embedding (`self.head.weight = self.tok.weight`). This is done *after* initialization and reduces the parameter count.
- `forward(ids, targets)` returns `(logits, loss)`. The loss is cross-entropy over all positions with `ignore_index=PAD_ID`.
- `generate(ids, max_new_tokens, temperature, top_k)` samples one token at a time:
  - `temperature` divides the logits. Below 1 makes the output more deterministic, above 1 more random.
  - `top_k` keeps only the `k` most likely tokens and sets the rest to `-inf` before sampling.
  - The context is cropped to the last `max_len` tokens at each step.

### 9. Tokenizer

`SimpleTokenizer` is a word-level tokenizer:

- `tokenize` lowercases the text and splits it with the regex `\w+|[^\w\s]` into words and single punctuation marks. It works for both English and Arabic.
- `build_vocab` counts tokens and assigns IDs by frequency. The four special tokens always come first:

| Token | ID | Purpose |
|---|---|---|
| `<PAD>` | 0 | fills short sequences up to the batch length |
| `<BOS>` | 1 | marks the start of a sequence |
| `<EOS>` | 2 | marks the end of a sequence |
| `<UNK>` | 3 | replaces words that are not in the vocabulary |

- `encode(text, add_bos, add_eos)` returns token IDs, `decode(ids)` turns them back into text and re-attaches punctuation.
- One **shared vocabulary** is built from the English side, the Arabic side and the GPT texts, which gives 199 tokens in total.

### 10. Batching helpers

- `pad_sequences(sequences, pad_id, max_len)` pads a list of variable-length ID lists into a single rectangular tensor, truncating if `max_len` is given.
- `get_batch(...)` (GPT) samples random windows of `block_size` tokens from one long token stream. The target `y` is the same window shifted by one token.
- `make_transformer_batch(...)` (translation) collects the chosen examples and pads the source, decoder input and expected output separately.

---

## Data

Both files are tiny toy datasets, chosen so the whole pipeline can be inspected and debugged by hand.

### `toy_translation_pairs.csv`

50 English → Arabic pairs with the columns `source_en` and `target_ar`. Sentences are short (about 3.5 tokens on the English side on average) and cover greetings, simple statements and a few machine-learning phrases.

```
source_en,target_ar
hello,مرحبا
good morning,صباح الخير
how are you,كيف حالك
i am learning python,أنا أتعلم بايثون
```

The file is read with `encoding="utf-8-sig"` because it starts with a byte-order mark.

### `gpt_toy_texts.jsonl`

56 samples, one JSON object per line with a `text` field. 50 of them are the same translation pairs written in a prompt-like format, and 6 are short free-text sentences in English and Arabic.

```json
{"text": "English: hello\nArabic: مرحبا\n"}
{"text": "مرحبا. أنا أتعلم بايثون. أحب التعلم.\n"}
```

### Splits

Both datasets are shuffled with a fixed seed (`42`) and split **90% train / 10% validation**:

| Model | Train | Validation |
|---|---|---|
| Translation | 45 pairs | 5 pairs |
| GPT | 608 tokens | 67 tokens |

---

## Training

### How a translation example is built

Training uses **teacher forcing**: the decoder receives the correct previous tokens as input and is trained to predict the next one at every position. For "i am learning python" → "أنا أتعلم بايثون":

Encoder input (`src`): `i am learning python <EOS>` → IDs `[9, 18, 12, 20, 2]`

| | pos 0 | pos 1 | pos 2 | pos 3 |
|---|---|---|---|---|
| Decoder input (`tgt_in`) | `<BOS>` | `أنا` | `أتعلم` | `بايثون` |
| Expected output (`tgt_out`) | `أنا` | `أتعلم` | `بايثون` | `<EOS>` |

At every position the decoder only sees the tokens to its left (causal mask) and is scored on predicting the token in the row below it.

This is the "output sequence (shifted right)" in the diagram. In code the target is encoded as `<BOS> … <EOS>`, then `tgt_ids[:-1]` becomes the decoder input and `tgt_ids[1:]` the expected output.

### Loss

Cross-entropy between the logits and the expected tokens, with `ignore_index=PAD_ID` so padding positions do not contribute. Logits are flattened to `[B*T, vocab]` before the call.

### Hyperparameters

| | Translation (Encoder-Decoder) | GPT |
|---|---|---|
| `d_model` | 128 | 128 |
| `num_heads` | 4 | 4 |
| `d_ff` | 512 | 512 |
| `num_layers` | 2 encoder + 2 decoder | 2 |
| `dropout` | 0.1 | 0.1 |
| `max_len` | 256 | 256 |
| Vocabulary | 199 | 199 |
| Approx. parameters | ~1.0 M | ~0.42 M |
| Optimizer | AdamW | AdamW |
| Learning rate | 3e-4 | 3e-4 |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping | max norm 1.0 | max norm 1.0 |
| Batch size | 8 pairs | 16 sequences |
| Length | 80 epochs | 400 steps, `block_size=32` |
| Evaluation | every epoch on the validation pairs | at step 1 and every 50 steps, on 10 random batches |

Parameter counts are calculated from the architecture. You can confirm them with `count_params(model)`, which is defined in the notebook.

### Training loop (both models)

```python
optimizer.zero_grad(set_to_none=True)
logits = model(...)
loss = F.cross_entropy(..., ignore_index=PAD_ID)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Gradient clipping caps the gradient norm so one bad batch cannot blow up the weights. The translation model reshuffles the training pairs at the start of every epoch.

---

## Inference

**Translation (greedy decoding).** The source sentence is encoded once, then the decoder starts from `<BOS>` and repeatedly appends its most likely next token until it produces `<EOS>` or reaches `max_len`.

```python
src = torch.tensor([tokenizer.encode("hello", add_eos=True)], device=device)
out = transformer.greedy_decode(src, bos_id=BOS_ID, eos_id=EOS_ID, max_len=20)
print(tokenizer.decode(out[0].tolist()))   # مرحبا
```

**Text generation (GPT).** The prompt is encoded with `<BOS>` and extended with sampled tokens.

```python
prompt = torch.tensor([tokenizer.encode("English:", add_bos=True)], device=device)
out = gpt.generate(prompt, max_new_tokens=30, temperature=0.8, top_k=10)
print(tokenizer.decode(out[0].tolist()))
```

---

## Results

![Translation training curve](assets/loss_curve.png)

### Translation model (45 train / 5 validation pairs)

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 1.2028 | 5.3999 |
| 20 | 0.5165 | 5.9146 |
| 40 | 0.1850 | 6.2374 |
| 60 | 0.0792 | 6.5030 |
| 80 | 0.0493 | 6.7438 |

Test: `hello` → `مرحبا` (generated IDs `[<BOS>, 22, <EOS>]`).

### GPT model (608 train / 67 validation tokens)

| Step | Train loss | Validation loss |
|---|---|---|
| 1 | 1.0240 | 3.5492 |
| 100 | 0.7706 | 3.4489 |
| 200 | 0.5068 | 3.5275 |
| 300 | 0.3184 | 3.6528 |
| 400 | 0.2058 | 3.9647 |

Sample generation from the prompt `English:` reproduces lines from the training data (for example `english: i work with data arabic: أنا أعمل مع البيانات`). The tokenizer lowercases everything, so the output is lowercase.

### How to read these numbers

The training loss goes down steadily, so the model, the masks and the training loop are doing their job: gradients flow and the network can fit its data. But the validation loss goes **up** in both experiments. With only 45 training pairs and 5 validation pairs, the models memorize the training set instead of learning to translate, and that is exactly what a model of this size should do on this little data. The test above works because `hello` appears in the training data.

Also keep in mind that validation sets of 5 pairs and 67 tokens are far too small to give reliable numbers. Treat the validation curve as a sanity check of the pipeline, not as a quality estimate.

---

## Limitations and next steps

- **Dataset size.** The main limitation. Meaningful translation needs thousands of sentence pairs at minimum. Public parallel corpora such as Tatoeba or OPUS are a natural next step.
- **Tokenizer.** The word-level tokenizer has no subword handling, so any unseen word becomes `<UNK>`, and it lowercases everything. A BPE or SentencePiece tokenizer would fix both, and would suit Arabic morphology much better.
- **Regularization.** To reduce overfitting: more data, higher dropout, early stopping on validation loss, label smoothing.
- **Decoding.** Greedy decoding only. Beam search would improve translation quality.
- **Learning-rate schedule.** A constant learning rate is used. Warm-up plus cosine decay is the usual choice for larger runs.
- **Evaluation.** There is no BLEU score yet, only the loss.

### Notes about the notebook

- `model` is assigned twice in the "instantiate models" cell, so only the GPT instance remains afterwards. The models actually trained are `gpt` and `transformer`, created in later cells.
- The translation training cell sets `batch_size = 8`, which overwrites the `batch_size = 16` used for GPT. Re-running the GPT training cells after that would use the smaller value.

---

## Saving and loading checkpoints

Each checkpoint stores the weights, the vocabulary and the model configuration. The classes (`Transformer`, `GPT`, `SimpleTokenizer`) must be defined first, so run the class cells of the notebook before loading.

```python
import torch

ckpt = torch.load("transformer_toy_checkpoint.pt", map_location="cpu")

# Rebuild the tokenizer from the saved vocabulary
tokenizer = SimpleTokenizer()
tokenizer.token_to_id = ckpt["vocab"]
tokenizer.id_to_token = {i: t for t, i in ckpt["vocab"].items()}

# Rebuild the model and load the weights
model = Transformer(len(tokenizer), len(tokenizer), **ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

For the GPT checkpoint use `GPT(vocab_size=len(tokenizer), **ckpt["config"])` instead. The `dropout` value is not stored in the config, so it falls back to the default of `0.1`, which has no effect in eval mode.

---

## References

- Vaswani et al., *Attention Is All You Need*, 2017.
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, 2020 (Pre-LN vs Post-LN).
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.
- Press and Wolf, *Using the Output Embedding to Improve Language Models*, 2017 (weight tying).
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization*, 2019 (AdamW).
