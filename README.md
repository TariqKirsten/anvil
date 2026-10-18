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
| `server/anvil/` | The Python server: accounts and login, GPUs, the Hugging Face search, model checks, datasets |
| `server/tests/` | `docker compose exec -w /workspace/server dev python -m pytest` |
| `web/src/` | The React screen: `Finetune.tsx`, `Settings.tsx`, and one file per section in `sections/` |

## Status

Built in steps:

1. **The screen** (done): your real GPUs, Hugging Face search, local model folders and their checks, the full form,
   accounts and settings.
2. **Data** (done): choosing datasets (this machine, upload, Hugging Face), checking rows, duplicate removal, splits.
3. Training on one GPU, with checkpoints and live progress.
4. The judge: score checkpoints, keep the best, stop early.
5. Training on several GPUs at once.
6. The Clef decision head.
