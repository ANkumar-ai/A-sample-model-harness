# ERA V5 — Session 9: Loss Functions & Output Heads

A single self-contained Colab notebook that takes the three-line loss snippet from the session and makes it **correct and observable**, then adds a second output head that predicts `t+2`.

**Notebook:** [`era_v5_s9_loss_harness.ipynb`](./era_v5_s9_loss_harness.ipynb) — Runtime → Run all. One `pip install tiktoken`, everything else is PyTorch.

## Setup

| | |
|---|---|
| Data | Tiny Shakespeare — 1,115,394 chars → 338,025 tokens (304,222 train / 33,803 val) |
| Tokenizer | GPT-2 BPE via `tiktoken`, **V = 50,257** |
| Model | 4 layers, d_model 256, 4 heads, ctx 128 — pre-norm, RMSNorm, SwiGLU FFN, untied head. 28,781,824 params |
| Training | 300 steps, AdamW, lr 3e-4, batch 16, on a Colab T4 |

## Part 1 — the seven numbers

| # | Check | Result |
|---|---|---|
| 1 | Tensor shapes | tokens `[16,128]` → hidden `[16,128,256]` → logits `[16,128,50257]`; the logits tensor is **196×** the hidden state that produced it |
| 2 | Shift verified on **strings** | `" whence 'tis derived.\nThere is another friar"` → `" 'tis derived.\nThere is another friar that"` — shifted left by one |
| 3 | Padding masked | loss **5.7769 → 5.4542**, contributing tokens **60 → 42** |
| 4 | Document boundary masked | loss **9.3517 → 9.1732**, contributing tokens **28 → 27** |
| 5 | Untrained perplexity | loss **10.9808** vs `ln(V) = 10.8249`; perplexity **58,736** vs V = 50,257 |
| 6 | Tied vs untied head | untied **28,781,824** → tied **15,916,032** (tying saves 12,865,792, **44.7%**) |
| 7 | Naive vs chunked CE | peak logits **389.6 MiB → 49.1 MiB** (**7.9×**) at chunk 256; loss 10.991126 vs 10.991125, max gradient difference **0.00e+00** |

Rows 3 and 4 are the numbers from the trained model — see below.

### What the masking checks actually show
- **Padding.** Counting PAD makes the loss look *better* than it is, because PAD is trivially predictable. On the untrained model the two numbers are indistinguishable (10.9079 vs 10.9335 — noise). After 300 steps the gap is real and in the expected direction: **5.7769 with PAD counted, 5.4542 without**. The model had already learned to emit PAD after PAD, and that was quietly propping up the loss.
- **Boundary.** The crossing pair prints as `'.'` → `'Crypt'` — the end of a sentence about Delhi's weather predicting the start of a cryptography document. Training that pair teaches the model that unrelated things follow each other.
- **The mean.** Both checks report the contributing-token count, because the denominator has to be the positions that actually counted, not `B×T`.

### Chunked cross-entropy
Written by hand: process the 2,032 flattened positions in blocks of 256, accumulate `reduction="sum"`, divide once by the contributing count. Backward recomputes each chunk's logits via `torch.utils.checkpoint` — a little extra arithmetic for a large memory saving.

The gradient into the hidden states is **bit-identical**; the loss differs by 1e-6, which is float accumulation order, not a different objective.

Two memory numbers are reported and they don't agree, on purpose:
- **389.6 MiB → 49.1 MiB (7.9×)** — the logits tensor itself, `N × V × 4 bytes`. This is the thing chunking actually removes.
- **2350.4 MiB → 1437.9 MiB (1.6×)** — CUDA peak allocated, which also contains the head weights, their gradients, and softmax intermediates that chunking doesn't touch. The real saving is diluted by everything else living on the card.

At the session's config (V=131,072, D=4,096, 256K context) the first number is what turns 64 GiB into something that fits.

## Part 2 — a second head at t+2

One trunk, two `[V, D]` heads, losses added:

```
h  = trunk(tokens)
L1 = CE(head1(h), tokens shifted by 1)
L2 = CE(head2(h), tokens shifted by 2)
loss = L1 + L2
```

| Step | head1 (t+1) | head2 (t+2) | sum | gap |
|---|---|---|---|---|
| 1 | 10.9812 | 11.0004 | 21.9816 | 0.019 |
| 50 | 6.5976 | 6.7347 | 13.3322 | 0.137 |
| 150 | 5.9832 | 6.2052 | 12.1884 | 0.222 |
| 300 | 5.4826 | 5.9489 | 11.4316 | 0.466 |

**What happens:** both heads fall together, but head2 sits above head1 the whole way and **the gap widens** — 0.02 nats at initialisation, 0.47 by step 300. At step 1 both heads are near `ln(V)` and equally ignorant, so there is nothing to separate them. As the trunk learns, head1 collects the easy wins — finishing a BPE-split word, the space after a token, the second half of a name — and none of those are available to head2, which has to skip the token it was never shown. What's left for head2 is the genuinely harder question, so its loss plateaus higher.

The two heads share every parameter below the head, so head2's gradient forces the hidden state to carry information beyond the immediate next token. That is the training-side argument for MTP, and it's separate from the inference-side draft-and-verify story. The cost is honest: each extra head is another full `[V, D]` matrix — here +12.9M params, at the session's real config +536.9M — which is the argument for a factored head.

## Notes
- Untrained perplexity lands at 58,736 against a vocabulary of 50,257 — close enough to `ln(V)` to trust the target alignment. Anything near 4 or 5 at step 0 would have meant the model was being handed its own input.
