# Dropout + Autoregressive Rollout — Design

## Context

The existing ETTh1 (input=672, predict=96) reproduction trained on Kaggle T4×2
(`kaggle/experiments/03_full_train_dual_gpu`) and evaluated
(`04_test_eval`) scores MSE=0.4911/MAE=0.4729 against the paper's
0.342/0.380 (Table A.12). Two known gaps from that comparison are in scope
for this session:

1. **Dropout (Table 2: 0.2 for ETTh1)** is specified in the paper's
   hyperparameters and shown in Fig. 2's block diagrams, but is entirely
   absent from the current architecture code.
2. **Autoregressive rollout** for horizons 192/336/720 doesn't exist yet —
   the current pipeline only ever predicts a single 96-step patch. The
   paper's own text (Section 3.2.4/3.3) confirms these longer horizons come
   from feeding the model's own 96-step predictions back in as input,
   recursively, exactly the way AutoTimes does it — this is not an
   enhancement, it's the only way to reach those benchmark numbers at all.

Gradient checkpointing (a third gap noted in the original report) is
explicitly deprioritized for this session — it isn't paper-mandated (the
paper never mentions how it reached its reported ~6GB training memory), and
the existing T4×2 `DataParallel` workaround already lets the paper's true
batch=256 run to completion.

## Goals

- Add dropout to the trainable architecture, matching the paper's Table 2
  value (0.2) and Fig. 2's block structure as closely as the figure's
  extracted text supports.
- Retrain ETTh1-96 from scratch with dropout active throughout, producing a
  new checkpoint comparable to the existing one (same everything else held
  constant, so any change in test MSE/MAE is attributable to dropout).
- Build an autoregressive rollout evaluator for horizons {192, 336, 720},
  matching the paper's Table A.12 comparison points.
- Run that rollout evaluator against **both** the existing (no-dropout) and
  new (dropout) checkpoints, producing one combined comparison table. This
  also happens to give us a free look at whether dropout changes
  error-accumulation behavior over the rollout — something the paper's own
  ablation (Table 8) never isolates.
- Add a small CPU-only pytest that exercises the new architecture code
  (embedding module + FAN-MoE decoder, dropout included) end-to-end on
  random tensors with a dummy stand-in LLM, to catch shape/wiring bugs
  before spending Kaggle GPU quota.

## Non-goals

- Gradient checkpointing (deprioritized, see above).
- Any dataset beyond ETTh1, or any horizon beyond 720.
- Editing `01`–`04` in place — they stay as frozen historical records of
  what was actually run and logged.
- Hyperparameter tuning beyond what Table 2 already specifies for ETTh1.

## Design

### 1. Dropout placement

Fig. 2's block diagrams show `Dropout` once in the Encoder (embedding
module) stack and twice in the Fourier Expert (shared expert) stack. Exact
*position* within each stack isn't reliably recoverable from the figure's
extracted text (PDF box-label extraction doesn't preserve layout order
faithfully), so placement follows the paper's Eq. 9–11 / Eq. 14 layer order
with dropout inserted at the most natural regularization points — right
before the FAN transform, and (for the Fourier expert) once more before the
final projection:

```
FourierEmbeddingModule:
  Linear -> SiLU -> Dropout(0.2) -> FAN -> Tanh -> Linear

FANExpert (shared/Fourier expert):
  Linear -> Tanh -> Dropout(0.2) -> FAN -> SiLU -> Dropout(0.2) -> Linear

RoutedExpert:
  unchanged — no dropout (paper: "standard feed-forward networks... composed
  of two linear layers", no dropout mentioned)
```

`nn.Dropout` carries no learnable parameters, so this doesn't change any
`state_dict` key or shape — the same class definitions can load *either*
the old (no-dropout-trained) or new (dropout-trained) checkpoint, and
`nn.Dropout` no-ops automatically under `.eval()`. This means the rollout
evaluator (component 3) needs only one model definition, not two.

This dropout-added architecture is written once, in the new `05_*`
experiment folder (component 2) — `01`–`04` are not modified.

### 2. Retrain — `kaggle/experiments/05_dropout_retrain/`

Direct copy of `03_full_train_dual_gpu/kaggle_train_full.py` with only the
dropout layers from Section 1 added. Everything else held identical:
batch=256, T4×2 `DataParallel` (128+128 split), lr=2e-4, `NUM_EPOCHS=30`
ceiling with `PATIENCE=3` early stopping, same 12/4/4-month chronological
split and per-channel normalization, same hybrid + balance loss. Holding
every other variable constant isolates dropout as the only explanatory
change versus the existing checkpoint.

Output checkpoint: `etth1_96_dropout_best.pt`, saved to `/kaggle/working/`
during the run exactly like `03` does.

**Checkpoint distribution**: after pulling the new checkpoint down locally,
add it *alongside* the existing `etth1_96_best.pt` in
`kaggle/datasets/checkpoint/` and push a new version of that same private
Kaggle dataset (`kaggle datasets version -p kaggle/datasets/checkpoint -m "add dropout checkpoint"`).
One dataset mount then gives the eval kernel (component 3) access to both
checkpoints simultaneously, rather than needing two separate dataset
attachments.

### 3. Rollout eval — `kaggle/experiments/06_rollout_eval/`

- `ForecastWindowDataset` generalized to build windows with `history=672`
  and `future=720` (the max horizon needed) — one window's future slice
  serves all three horizons.
- Rollout procedure per window, sliding-window style (context always
  exactly 7 tokens = 672 steps):
  1. Start from the 7 history tokens.
  2. Predict the next 96-step patch (one forward pass through the full
     pipeline, same as training/eval today).
  3. Drop the oldest token, append the newly predicted patch as the newest
     token — context stays fixed at 7 tokens.
  4. Repeat until 8 patches have been generated (768 steps ≥ the 720-step
     max horizon; 8 steps because 336 and 720 aren't multiples of 96, so
     rollout must over-generate and truncate).
  5. From that one 768-step generated trajectory, slice the first
     192/336/720 steps to score each horizon separately — one rollout run
     serves all three horizons, rather than three independent rollouts.
- Loops over **both** checkpoints within a single kernel run (Llama backbone
  loaded once; embedding module + decoder weights swapped between the two
  checkpoints), producing one combined table: MSE/MAE for
  {192, 336, 720} × {no-dropout, dropout}, reported alongside the existing
  96-step numbers (already known from `04_test_eval`) for context.
- Metrics computed on normalized data, matching the convention already
  established in `04_test_eval`.
- Inference only — no gradients, single GPU (no `DataParallel` needed, same
  as `04`).

### 4. Local smoke test (new)

A CPU-only pytest added under `tests/` that:

- Builds `FourierEmbeddingModule`, `FANExpert`, `RoutedExpert`, `MoEGate`,
  and `FANMoEDecoder` (the dropout-added versions from Section 1) with small
  dimensions.
- Feeds random tensors through the full embedding → (dummy stand-in for the
  frozen Llama backbone, e.g. an `nn.Identity`-shaped stub preserving the
  `(b, s, n, h)` shape contract) → decoder pipeline, and asserts output
  shapes match expectations.
- Asserts dropout actually behaves differently between `.train()` and
  `.eval()` mode (e.g., two forward passes in train mode on the same input
  produce different outputs; eval mode is deterministic) — a cheap way to
  catch a dropout layer that got wired in but never actually applied.
- Does **not** load the real Llama-3.2-1B (no GPU, no HF token needed
  locally) — purpose is to catch shape/wiring bugs before spending Kaggle
  GPU quota, not to validate the real backbone.

## Data flow summary

```
ETTh1 CSV
  -> chronological 12/4/4-month split, per-channel normalization (train stats)
  -> ForecastWindowDataset(history=672, future=720)

Retrain (05):
  window -> patch tokens -> FourierEmbeddingModule -> frozen Llama -> FAN-MoE decoder
  -> single 96-step patch prediction -> hybrid loss + balance loss -> backward
  -> early-stopped best checkpoint (etth1_96_dropout_best.pt)

Rollout eval (06), per checkpoint:
  window -> 7 history tokens -> [predict patch -> slide context -> repeat] x8
  -> 768-step trajectory -> slice to {192,336,720} -> MSE/MAE per horizon
```

## Error handling

Follows existing repo conventions — no new patterns needed:

- `assert torch.cuda.is_available()` / GPU count assertions as in `01`–`04`.
- HF token and checkpoint dataset paths use the existing two-candidate
  fallback (`/kaggle/input/<slug>/...` vs
  `/kaggle/input/datasets/mackeykumi/<slug>/...`), since Kaggle's mount path
  has been observed to vary between kernels.

## Testing

- New local pytest (Section 4) — fast, CPU-only, runs in the existing `uv
  run pytest` flow.
- Kaggle-side validation is inherently manual: push each kernel, poll for
  `COMPLETE`, inspect the printed metrics — same workflow as `01`–`04`,
  documented in `kaggle/README.md`. `05` and `06` will be added to that
  experiments table once they've actually run, following the existing
  convention.
