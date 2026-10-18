# anvil

Fine-tune language models and decision models from one screen, on your own GPUs.

Everything runs in Docker, on AMD's ROCm 10.1 + PyTorch 2.14 image, so the host only needs Docker and the GPU driver.

## Run it

1. Copy `.env.example` to `.env` and fill it in. `ANVIL_PASSWORD` becomes the first password of the `admin` account.
2. Create the four folders `.env` names, so they belong to you rather than to root:

   ```bash
   mkdir -p /srv/anvil/data /srv/anvil/hf-cache /srv/anvil/models /srv/anvil/datasets
   ```

3. Development, with live reload:

   ```bash
   docker compose up dev
   ```

   Open `http://<this machine>:5173` and log in as `admin`.

4. Or run the finished app in the background:

   ```bash
   docker compose up -d app
   ```

   Open `http://<this machine>:8420`.

## Folders

Inside the container everything lives at fixed paths. `.env` says which host folders they are.

| Inside | `.env` | What |
|---|---|---|
| `/data` | `ANVIL_DATA` | Accounts, runs, uploaded datasets |
| `/hf-cache` | `ANVIL_HF_CACHE` | The Hugging Face cache: downloaded models and datasets |
| `/models` | `ANVIL_MODELS` | Local models you can pick from |
| `/datasets` | `ANVIL_DATASETS` | Dataset files you can pick from (uploads go into `/data`) |

To reuse a Hugging Face cache you already have, or show models from another folder, copy
`compose.override.example.yaml` to `compose.override.yaml` and fill in the paths.

## Data

A dataset is a .parquet, .jsonl, .json or .csv file on this machine, a file uploaded from your browser, or a split
of a Hugging Face dataset. Its format is recognised from its columns:

| Format | Columns |
|---|---|
| Decision | `state`, `questions`, `gold` (System One style, checked with Studio's rules) |
| Chat | `messages` (a list of `{role, content}`) |
| Prompt and completion | `prompt`, `completion` |
| Plain text | `text` |

**Prepare data** checks every row, removes exact and near-copies (MinHash over word trigrams, 70% shared), drops
validation or test rows that share an input with train, and cuts whatever split you didn't give from train. Rows that
share an input stay on one side of the split. The result is cached: preparing the same data with the same settings
again is instant.

## Training

**Start training** checks the whole form, then starts a separate trainer process on one GPU (one that doesn't drive a
screen, if you picked several). The run gets its own page (Runs, top left) with progress, speed, time left, GPU
memory, the loss and validation charts, a table of every validation check, the final test, the checkpoints and the
log.

- **Decision models** train the letters way: each question lists its options as A, B, C… and the model's scores for
  the letters are its answer. Option order is shuffled per row so no letter is favoured; validation asks every question
  in both orders and averages. Validation reports loss, accuracy, AUROC per yes/no question, the catch rate at 1%
  false alarms, the false alarms at 50%, and the calibration error.
- **Language models** learn chat (only the assistant's replies), prompt and completion (only the completion), or plain
  text.
- Checkpoints keep the trainable weights and the optimizer, so **Stop** loses at most one step and **Resume** continues
  exactly where it stopped, after a crash too.
- The finished model is in the run's folder: `model/adapter` and `model/merged` (for LoRA), or `model/merged`. Its
  `anvil.json` says which checkpoint it is and what it scored on the test set.

### The judge: picking the best checkpoint

- Every checkpoint gets a validation check, and so does the model before training (step 0, shown as the starting
  point, never picked). **Pick best by** names the number that ranks them; the run page highlights it.
- If there's a free GPU besides the training one, a second process, the judge, does the checks there, so training
  never pauses for them. Without one, training pauses briefly at each checkpoint. If the judge crashes, the trainer
  takes over the checks.
- Only the best **Keep best** checkpoints are kept, plus the newest (Resume starts there) and any not checked yet.
- **Stop after no gain for** N checks ends training early once the score hasn't improved for N checks in a row
  (0 turns this off).
- When training ends, the best checkpoint becomes the model and is scored once on the test set (up to 5,000 rows).
  The test set plays no part in picking, so that's the fair number to quote.

The image compiles `causal_conv1d` for your GPU (`ANVIL_GPU_ARCH` in `.env`) and includes flash-linear-attention, the
fast kernels for linear-attention models such as Qwen3.5. On an R9700 that makes Qwen3.5 training about 22% faster.
`ANVIL_SAFE_KERNELS=1` switches both off if a model misbehaves.

## Accounts and Hugging Face

- anvil starts with one account, `admin`. Change its password in Settings (the gear, top right).
- Paste your Hugging Face token in Settings. anvil checks it with Hugging Face, keeps it on the server, and never
  shows it again. Public models work without a token; gated ones (like Llama) need it.
- Forgot the password? Put a new one in `ANVIL_PASSWORD`, add `ANVIL_RESET_PASSWORD=1`, restart, then remove that
  line again.

## Code

| Path | What |
|---|---|
| `docker/` | The image (`Dockerfile`), plus the scripts the containers run |
| `server/anvil/` | The Python server: accounts and login, GPUs, the Hugging Face search, model checks, datasets, runs |
| `server/anvil/train.py` | The trainer process; `tasks.py` holds what each kind of model learns and how it's scored |
| `server/anvil/judge.py` | The judge process: checks checkpoints on a spare GPU; `picking.py` ranks them and decides when to stop |
| `server/tests/` | `docker compose exec -w /workspace/server dev python -m pytest` |
| `web/src/` | The React screen: `Finetune.tsx`, `Runs.tsx`, `RunPage.tsx`, `Settings.tsx`, sections in `sections/` |

## Status

Built in steps:

1. **The screen** (done): your real GPUs, Hugging Face search, local model folders and their checks, the full form,
   accounts and settings.
2. **Data** (done): choosing datasets (this machine, upload, Hugging Face), checking rows, duplicate removal, splits.
3. **Training** (done): one GPU, decision models (letters) and language models, checkpoints, Stop and Resume, a live
   run page.
4. **The judge** (done): every checkpoint scored (on a spare GPU when there is one), the best kept and used as the
   model, early stopping, and a final test on the test set.
5. Training on several GPUs at once.
6. The Clef decision head.
