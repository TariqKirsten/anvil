# Anvil

**Make training data with a model, fine-tune on your own GPUs, and export a model that runs
anywhere, all from one web app.**

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![React 19](https://img.shields.io/badge/react-19-61dafb)
![ROCm 10.1](https://img.shields.io/badge/ROCm-10.1-red)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

Anvil fine-tunes two kinds of model: language models, and decision models, which answer typed
questions about an input (yes or no, a choice, or a score) with probabilities. It runs in Docker on
AMD GPUs.
It covers the whole path: synthetic data from a model, data checks, training, picking the best
checkpoint, and export to GGUF or Hugging Face.

<!-- SCREENSHOT: add a screenshot or GIF of Anvil here (a run's page with its charts, or the pipeline
     canvas). Save it as docs/images/anvil.png (or .gif) and uncomment the line below. -->
<!-- ![Anvil: a fine-tuning run's page, with live charts and scores per question](docs/images/anvil.png) -->

## Why I built it

I would have used an existing tool such as Unsloth. When I started, it did not support ROCm 10.1,
which my two Radeon AI PRO R9700 GPUs need for training. It also could not train on both GPUs, so
every run took twice as long. I needed multi-GPU training, so I built my own platform for
fine-tuning and data generation.

I have used it to fine-tune Qwen3.5 models, among them decision-0.8B, a safety classifier
([docs/DECISION-MODELS.md](docs/DECISION-MODELS.md)).

## Key features

- **Two kinds of model.** Language models learn from chat, prompt and completion, or plain text.
  Decision models learn to answer questions about an input. Both train with LoRA or full
  fine-tuning.
- **Two ways to train a decision model.** *Letters* reads the model's scores for the option letters.
  *Clef* adds Cloudflare's joint schema head, a small network that reads the model's hidden states
  and answers every question about an input in one pass.
- **Synthetic data pipelines.** You draw the steps on a canvas: Model, Seed data, List, Generate,
  Question, Filter, Dedupe and Save. A Model step can use any OpenAI-compatible endpoint (vLLM,
  llama.cpp, Ollama, OpenAI). A preview shows each row's way through the steps. A run continues in
  its own process, and you can stop and resume it. Six templates are included.
- **Pipeline assistant.** You describe the data you want, and a model that can call tools proposes
  the pipeline and builds it with you. It previews, but only you start a full run. You can undo each
  of its turns.
- **Data preparation.** Datasets come from local files (.parquet, .jsonl, .json, .csv), uploads or
  Hugging Face. Each row is checked. Exact and near-duplicates are removed (MinHash). Validation and
  test rows that share an input with a training row are dropped.
- **A judge that picks the best checkpoint.** Each checkpoint is scored on the validation set, on a
  spare GPU if there is one. Each score has a bootstrap standard error. Early stopping counts only
  gains larger than a minimum that you set. The test set is scored once, with the picked checkpoint.
- **Scores for each question.** Yes/no questions get AUROC and the catch rate at a fixed false-alarm
  rate. Choices get macro-F1, and scores get the mean absolute error. A check on rows the model
  trained on shows which questions it memorised.
- **Several GPUs.** Training is data-parallel through torchrun. A run can resume on fewer cards.
- **Runs that survive failures.** A checkpoint keeps the optimizer state, and it is complete on disk
  before it gets its name, so a power cut cannot break the newest one. A supervisor restarts a
  trainer after a crash in native code, or after 15 minutes without progress. A batch whose loss or
  gradients are NaN or infinite is skipped.
- **Full fine-tuning in bf16.** Anvil's own AdamW does the maths in fp32 and rounds the results into
  bf16 stochastically, so small updates are not lost.
- **Export and share.** Export the finished model or any kept checkpoint as GGUF (BF16 to Q4_K_M)
  with llama.cpp. A Clef model keeps its head, and llama-server answers decision requests with it.
  Upload weights and GGUF files to Hugging Face, with a model card that Anvil writes.
- **Login and GPU sharing.** A password login protects the app, because it listens on the network.
  Before a run starts, Anvil can ask a model server on the same GPUs to unload its model.

## Architecture

```mermaid
flowchart LR
    browser(["Browser<br/>React · React Flow"])

    subgraph box["Docker container: ROCm 10.1 · PyTorch 2.14"]
        api["FastAPI server<br/>accounts · datasets · runs · pipelines"]
        subgraph procs["Processes that the server starts"]
            sup["Supervisor<br/>restarts a crashed or frozen trainer"]
            trainer["Trainer<br/>one GPU, or several through torchrun"]
            judge["Judge<br/>scores checkpoints on a spare GPU"]
            gen["Pipeline run<br/>makes rows with a model"]
            asst["Pipeline assistant<br/>builds a pipeline with you"]
            exp["GGUF export<br/>llama.cpp, on the CPU"]
            up["Hugging Face upload"]
        end
    end

    gpus[["AMD GPUs"]]
    data[("/data<br/>accounts · runs · checkpoints<br/>datasets · pipelines")]
    hub[("Hugging Face Hub")]
    llms[("OpenAI-compatible servers<br/>vLLM · llama.cpp · Ollama · OpenAI")]

    browser -->|"/api, login cookie"| api
    api --> sup
    sup --> trainer
    api --> judge
    api --> gen
    api --> asst
    api --> exp
    api --> up
    trainer --> gpus
    judge --> gpus
    procs -->|"status, metrics, checkpoints"| data
    api -->|"reads"| data
    gen --> llms
    asst --> llms
    up --> hub
    api -->|"search, model checks"| hub
```

The server does not train. It starts each long job as a process of its own: the trainer (through
a supervisor), the judge, a pipeline run, an assistant turn, an export or an upload. These
processes report through files in the run's or pipeline's folder, and the server reads those files
and sends signals. So a restart of the server never stops a run.

| Path | What it holds |
|---|---|
| `server/anvil/main.py` | The API, and the built screen at every other path |
| `server/anvil/train.py`, `tasks.py` | The trainer process, and what each kind of model learns and how it is scored |
| `server/anvil/judge.py`, `picking.py` | Checking checkpoints on a spare GPU, ranking them and early stopping |
| `server/anvil/supervisor.py`, `parallel.py` | Restarts after crashes and freezes, and training on several GPUs |
| `server/anvil/datasets.py` | Reading datasets, checking rows, removing duplicates and making the splits |
| `server/anvil/pipelines.py`, `generate.py`, `assistant.py` | Pipelines, their runs and the assistant |
| `server/anvil/export.py`, `upload.py` | GGUF export with llama.cpp, and uploads to Hugging Face |
| `server/anvil/optim.py` | AdamW for bf16 weights, with stochastic rounding |
| `server/anvil/vendor/clef/` | Cloudflare's Clef reference code, unmodified (Apache-2.0) |
| `web/src/` | The React app; the pipeline editor is in `web/src/pipeline/` |
| `docker/` | The image, and the scripts that the containers run |

## Tech stack

| Area | Tools |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn, SQLite (accounts) |
| Training | PyTorch 2.14 on ROCm 10.1, Hugging Face Transformers, PEFT (LoRA), torchrun, flash-linear-attention and causal-conv1d kernels |
| Data | Hugging Face Datasets, pandas, PyArrow, datasketch (MinHash) |
| Decision models | Cloudflare's Clef joint schema head |
| Export and sharing | llama.cpp v0.7.0 (GGUF conversion and quantisation), huggingface_hub |
| Frontend | React 19, TypeScript, Vite, React Flow, Tabler icons |
| Infrastructure | Docker Compose, a dev container |
| Tests | pytest, with the trainer tests on the CPU |

## Quick start

You need:
- Linux with Docker and Docker Compose.
- An AMD GPU that ROCm 10.1 supports. Anvil was built on two Radeon AI PRO R9700s (`gfx1201`).
  The image is built on AMD's ROCm PyTorch image, so it does not support NVIDIA cards.
- About 40 GB of disk for the image, and space for models, datasets and runs.

Get the code and make your settings:

```bash
git clone https://github.com/TariqKirsten/anvil.git
cd anvil
cp .env.example .env
```

In `.env`, set `ANVIL_PASSWORD`, the four folders, `ANVIL_GPU_ARCH` and your user and group ids
(see [Configuration](#configuration)). Then make the four folders, so that they belong to you and
not to root. With the default folders:

```bash
sudo mkdir -p /srv/anvil/{data,hf-cache,models,datasets}
sudo chown -R "$(id -u):$(id -g)" /srv/anvil
```

Build the image and start the app:

```bash
docker compose build          # the first build takes a while: a large ROCm image, and kernels compiled for your GPU
docker compose up -d app
```

Open `http://localhost:8420` (or `http://<this machine>:8420`) and log in as `admin`, with the
password from `ANVIL_PASSWORD`.

For development, `docker compose up dev` serves the app with live reload at
`http://localhost:5173`. Run the tests with:

```bash
docker compose exec -w /workspace/server dev python -m pytest
docker compose exec dev npm --prefix web run typecheck
```

## Configuration

`.env` holds the settings ([.env.example](.env.example) has placeholders and notes). Compose passes
it to the container.

| Variable | What it sets |
|---|---|
| `ANVIL_PASSWORD` | The first password of the `admin` account, used once, on the first start. Change it in the app after that. |
| `ANVIL_RESET_PASSWORD` | Set it to `1`, with a new `ANVIL_PASSWORD`, and restart to reset a forgotten password. Then remove it. |
| `ANVIL_DATA` | The host folder for Anvil's own data: accounts, runs, uploaded datasets and pipelines. It is `/data` in the container. |
| `ANVIL_HF_CACHE` | The host folder for the Hugging Face cache. It is `/hf-cache` in the container. |
| `ANVIL_MODELS` | A host folder of local models to pick from. It is `/models` in the container. |
| `ANVIL_DATASETS` | A host folder of dataset files to pick from. It is `/datasets` in the container. |
| `HF_TOKEN` | Optional: a Hugging Face token for everyone on this Anvil. Usually each account adds its own in Settings. |
| `ANVIL_GPU_ARCH` | Your GPU's architecture, for the kernels compiled into the image: `gfx1201` for an R9700, `gfx1100` for an RX 7900. Rebuild after you change it. |
| `ANVIL_UID`, `ANVIL_GID` | Your user and group ids (`id -u`, `id -g`), so that the files the container writes belong to you. |
| `ANVIL_SAFE_KERNELS` | Set it to `1` to switch off the fast linear-attention kernels if a model misbehaves. |

In the app, Settings holds your password, your Hugging Face token, the assistant's model endpoint
and key, and the address that Anvil calls to free shared GPUs. API keys stay on the server and are
never sent back to the browser. To mount more folders, such as a Hugging Face cache that you
already have, copy `compose.override.example.yaml` to `compose.override.yaml` and fill in the
paths.

## Usage

1. **Make data (optional).** Under Pipelines, start a pipeline from a template, or describe the data
   you want to the assistant. Point its Model steps at your endpoint. Preview it, run it, then use
   **Train on it**.
2. **Train.** Under New run, pick a model (a local folder or a Hugging Face repo), the data, what to
   train it as, LoRA or full fine-tuning, and the GPUs. Then select **Start training**.
3. **Watch and pick.** The run's page shows progress, charts, the validation checks and the scores
   for each question. The judge keeps the best checkpoints, and the best one becomes the model.
4. **Ship.** From the run's page, export the model as GGUF or upload it to Hugging Face.

[docs/GUIDE.md](docs/GUIDE.md) describes each part in detail: the data formats, the pipeline steps,
the training settings, the judge, GPUs, export and upload.
[docs/DECISION-MODELS.md](docs/DECISION-MODELS.md) shows how to run a decision model that Anvil
exported.

## Status and roadmap

Anvil is a personal project, built for my own machine. It is not packaged or released, so you
build it from source.

Done:
1. The screen: GPUs, Hugging Face search, local model folders and their checks, accounts and
   settings.
2. Data: datasets from this machine, uploads or Hugging Face, row checks, duplicate removal and
   splits.
3. Training: decision models (letters) and language models, checkpoints, Stop and Resume, and a
   live run page.
4. The judge: each checkpoint scored, the best ones kept, early stopping and a final test.
5. Several GPUs.
6. The Clef head for decision models.
7. GGUF export.
8. Pipelines for synthetic data, with previews, runs, balancing, second opinions and templates.

Since then: scores for each question by its type, standard errors and a minimum gain for picking,
resume with more epochs, uploads to Hugging Face, stochastic rounding for bf16, restarts after
freezes, and the pipeline assistant. In a live test, the assistant built a yes/no fraud pipeline
from one sentence in about 2 minutes.

Next:
- Describe the assistant on the app's Docs page.
- Try each pipeline template for real.
- QLoRA (4-bit). Today a run uses LoRA or full fine-tuning.
- Try the Hugging Face upload on a real account. So far, tests check it against a fake Hub.

Limits:
- The image supports AMD GPUs only.
- The templates' Model steps point at a model server on my machine. Change them to your endpoint.

## License

[MIT](LICENSE), except `server/anvil/vendor/clef/`, which is Cloudflare's code under the
Apache-2.0 licence ([its LICENSE](server/anvil/vendor/clef/LICENSE) and
[NOTICE.md](server/anvil/vendor/clef/NOTICE.md)).
