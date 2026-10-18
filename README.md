# anvil

Fine-tune language models and decision models from one screen, on your own GPUs.

Everything runs in Docker, on AMD's ROCm 10.1 + PyTorch 2.14 image, so the host only needs Docker and the GPU driver.

## Run it

1. Copy `.env.example` to `.env` and fill it in. `ANVIL_PASSWORD` becomes the first password of the `admin` account.
2. Create the three folders `.env` names, so they belong to you rather than to root:

   ```bash
   mkdir -p /srv/anvil/data /srv/anvil/hf-cache /srv/anvil/models
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

To reuse a Hugging Face cache you already have, or show models from another folder, copy
`compose.override.example.yaml` to `compose.override.yaml` and fill in the paths.

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
| `server/anvil/` | The Python server: accounts and login, GPUs, the Hugging Face search, model checks |
| `server/tests/` | `docker compose exec -w /workspace/server dev python -m pytest` |
| `web/src/` | The React screen: `Finetune.tsx`, `Settings.tsx`, and one file per section in `sections/` |

## Status

Built in steps:

1. **The screen** (done): your real GPUs, Hugging Face search, local model folders and their checks, the full form,
   accounts and settings.
2. Data: choosing datasets, duplicate removal and splits.
3. Training on one GPU, with checkpoints and live progress.
4. The judge: score checkpoints, keep the best, stop early.
5. Training on several GPUs at once.
6. The Clef decision head.
