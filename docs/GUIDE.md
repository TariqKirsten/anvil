# Anvil guide

The [README](../README.md) has the overview and the quick start. This guide describes each part of the app in detail:
setup and configuration, the folders, the data, pipelines, training, export and upload, several GPUs, the judge, and
accounts. The app's own Docs page covers much of the same ground.

## Workflow

1. **Make data (optional).** Under Pipelines, start a pipeline from a template, or describe the data you want to the
   assistant. Point its Model steps at your endpoint. Preview it, run it, then use **Train on it**.
2. **Train.** Under New run, pick a model (a local folder or a Hugging Face repo), the data, what to train it as, LoRA
   or full fine-tuning, and the GPUs. Then select **Start training**.
3. **Watch and pick.** The run's page shows progress, charts, the validation checks and the scores for each question.
   The judge keeps the best checkpoints, and the best one becomes the model.
4. **Ship.** From the run's page, export the model as GGUF or upload it to Hugging Face.

## Setup

You need:
- Linux with Docker and Docker Compose.
- An AMD GPU that ROCm 10.1 supports. Anvil was built on two Radeon AI PRO R9700s (`gfx1201`). The image is built on
  AMD's ROCm PyTorch image, so it does not support NVIDIA cards.
- About 40 GB of disk for the image, and space for models, datasets and runs.

Copy `.env.example` to `.env` and set `ANVIL_PASSWORD`, the four folders, `ANVIL_GPU_ARCH` and your user and group ids
(below). Make the four folders first, so that they belong to you and not to root. With the default folders:

```bash
sudo mkdir -p /srv/anvil/{data,hf-cache,models,datasets}
sudo chown -R "$(id -u):$(id -g)" /srv/anvil
docker compose build          # the first build takes a while: a large ROCm image, and kernels compiled for your GPU
docker compose up -d app
```

Open `http://localhost:8420` (or `http://<this machine>:8420`) and log in as `admin`, with the password from
`ANVIL_PASSWORD`.

### Configuration

`.env` holds the settings ([.env.example](../.env.example) has placeholders and notes). Compose passes it to the
container.

| Variable | What it sets |
|---|---|
| `ANVIL_PASSWORD` | The first password of the `admin` account, used once, on the first start. Change it in the app after that. |
| `ANVIL_RESET_PASSWORD` | Set it to `1`, with a new `ANVIL_PASSWORD`, and restart to reset a forgotten password. Then remove it. |
| `ANVIL_DATA` | The host folder for Anvil's own data: accounts, runs, uploaded datasets and pipelines (`/data`). |
| `ANVIL_HF_CACHE` | The host folder for the Hugging Face cache (`/hf-cache`). |
| `ANVIL_MODELS` | A host folder of local models to pick from (`/models`). |
| `ANVIL_DATASETS` | A host folder of dataset files to pick from (`/datasets`). |
| `HF_TOKEN` | Optional: a Hugging Face token for everyone on this Anvil. Usually each account adds its own in Settings. |
| `ANVIL_GPU_ARCH` | Your GPU's architecture, for the kernels compiled into the image: `gfx1201` for an R9700, `gfx1100` for an RX 7900. Rebuild after you change it. |
| `ANVIL_UID`, `ANVIL_GID` | Your user and group ids (`id -u`, `id -g`), so that the files the container writes belong to you. |
| `ANVIL_SAFE_KERNELS` | Set it to `1` to switch off the fast linear-attention kernels if a model misbehaves. |

In the app, Settings holds your password, your Hugging Face token, the assistant's model endpoint and key, and the
address that Anvil calls to free shared GPUs. API keys stay on the server and are never sent back to the browser.

### Development

`docker compose up dev` serves the app with live reload at `http://localhost:5173`. Run the tests with:

```bash
docker compose exec -w /workspace/server dev python -m pytest
docker compose exec dev npm --prefix web run typecheck
```

## Folders

Inside the container everything lives at fixed paths. `.env` says which host folders they are.

| Inside | `.env` | What |
|---|---|---|
| `/data` | `ANVIL_DATA` | Accounts, runs, uploaded datasets, pipelines and the datasets they make |
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
share an input stay on one side of the split. For decision data it also says which questions the validation set
asks, by type, since that decides what the checks can score. The result is cached: preparing the same data with the
same settings again is instant.

## Pipelines: making data

Pipelines (top left) make datasets with a model, drawn as steps on a canvas. Every line of steps starts at a **Model**,
which talks to an OpenAI-compatible endpoint (vLLM, llama.cpp's server, radiance, Ollama, OpenAI). Rows flow down the
arrows; each step adds a column to every row or drops some, and a **Save** at the end of a line writes them as a
dataset in one of the formats above.

| Step | What it does |
|---|---|
| Model | The endpoint's base URL, its API key (kept on the server, never shown again, and shared by Models at the same address), the model (Connect lists them), temperature, max tokens, thinking on or off, requests at once. Try it sends one message |
| Seed data | Rows from a dataset you have: a file or a Hugging Face split; the columns you tick join each row |
| List | A value for each row from a list you type, at random or taking turns |
| Generate | The model writes text from a prompt with `{{ column }}` slots; asked for several, it answers with a JSON list and each item becomes a row |
| Question | A decision question (yes/no, choice or score), answered by the model or taken from a column through a mapping |
| Filter | Keeps the rows that pass its rules (is, contains, matches, longer or shorter than, empty), every rule or any |
| Dedupe | Drops exact and near-copies of a column, by Prepare data's rule |
| Save | Writes the rows (decision, chat, prompt and completion, or text), optionally balanced by a List's values or a Question's answers |

- A step asks the Model its line starts from, unless another Model is plugged into its diamond handle (a dashed line).
  A Question also takes a second opinion: two models' answers must agree; a column's answer stands unless the second
  model is sure it's wrong. An answer left out is asked, not scored; a decision row with no answers is dropped.
- Inside the container, this machine is `host.docker.internal` (compose maps it), so a server on this machine's port
  8000 is `http://host.docker.internal:8000/v1`. That server must listen on an address containers can reach, not only
  on 127.0.0.1. A firewall that only lets containers reach some ports needs a rule for others.
- The server checks the graph as it's saved (every change, a moment later) and marks what stops it from running, on
  the step and under the setting.
- **Preview** makes 5 rows for each Save and shows every row's way through the steps: the prompts, the replies, the
  answers, and why rows were dropped. **Run** makes them all in its own process (`anvil.generate`), so a restart of
  the server doesn't stop it; Stop keeps what it made and Resume carries on. Each step counts its rows in, out and
  dropped, live on the canvas.
- A run's datasets go to `ANVIL_DATA/datasets/generated/<name>-<run>.jsonl`, which New run's data picker lists, and
  **Train on it** opens New run with one chosen. Every row also gets `source` (the pipeline and the List values that
  made it, for Keep together by source) and `meta` (all its columns, as JSON).
- **New pipeline** offers six templates (`server/anvil/templates.py`), each a working pipeline with a note on every
  step: labelling messages you have, writing a yes/no dataset from scratch, routing to a team (a choice
  question), several questions per message, chat replies for a language model, and harmless Afrikaans and isiZulu
  messages as hard negatives. The first labels a sample of 24 bank messages that comes with anvil. Their Model steps
  point at a Gemma model on a radiance server on the author's machine: point them at your own endpoint.
- **Delete** (the bin on a pipeline's row in the list, or Delete pipeline on its About tab) removes the pipeline and
  its runs; the datasets it made stay. A pipeline with a run or a preview going can't be deleted.

### The pipeline assistant

- The assistant is any OpenAI-compatible model that can call tools, set up once in Settings: its address, key, model,
  context, thinking and reasoning effort.
- Describe the data you want, and it asks what it needs to know, proposes the steps and builds them when you say so. A
  clear change that you ask for, it just makes. It only previews: a full run is yours to start. In a live test, it
  built a yes/no fraud pipeline from one sentence in about 2 minutes.
- Each message starts a turn in a process of its own (`anvil.assistant`), so a turn outlives a restart of the server.
  A turn asks the model, carries out the tools it calls (building the pipeline, adding, changing and removing steps,
  previewing, listing models and datasets, looking at a dataset or a template), and asks again until the model answers
  without calling one: at most 24 calls and 5 previews a turn.
- The editor's side panel has an Assistant tab: the messages, a line for each thing the assistant did, Stop, Undo and
  Start over. While a turn works, the canvas is locked. Undo puts back what the newest turn changed.
- A Model step at the assistant's address that has no key of its own uses the assistant's key.
- A server that holds one model at a time can't answer the assistant while a run asks it for another. So a turn waits
  while a run or preview uses the assistant's address with a different model; the panel offers to stop that run (its
  rows are kept, and Resume carries on) or to ask anyway.

### Sharing the GPUs with a model server

- If a model server on this machine uses the same GPUs, Settings takes an address that anvil calls (a POST) before a
  training run starts or resumes, such as the server's unload endpoint, so that the server lets go of the cards.
  **Free them now** calls it by hand. If nothing answers, that's fine; an error stops the run from starting.
- While a run trains, anvil sends no requests to models at that server (the same host and port): a pipeline run there
  doesn't start, and the assistant waits.

## Training

**Start training** checks the whole form, then starts a separate trainer process. The run gets its own page (Runs,
top left) with progress (steps and epochs), speed, time left, GPU memory, the gradient norm, the tokens trained on, the
padding and the rows cut by max length, the loss and validation charts (the validation loss on the training chart too,
with the loss on trained rows, below), the gradient norm and learning rate charts (a click on any chart opens it bigger, and
it follows the run live there too), a table of every validation check (newest first, in a box of its own that
scrolls), the final test, the checkpoints and the log. A language model's page also has sample replies: at every
check the model answers the same three validation prompts, shown next to the replies the data gives.

- An **epoch** is one pass over the training rows. A **step** learns from one batch: batch size × the number of
  training cards × gradient accumulation rows. The form shows how many steps an epoch takes; **Max steps** cuts a run
  short (handy for quick tests).

- Every token a language model learns from, and every question a decision model answers, weighs the same in a step:
  the step's loss is averaged over all of its batches (gradient accumulation) and all of its cards together, so a
  batch of short rows doesn't count as much as a batch of long ones. The training loss on the run page is that
  average, so it compares with the validation loss.

- **Decision models** train one of two ways (Method, under Train as):
  - **Letters**: each question lists its options as A, B, C… and the model's scores for the letters are its answer.
    Option order is shuffled per row, and differently in each epoch, so no letter is favoured; validation asks every
    question in both orders and averages. Nothing is added to the model. One pass per question.
  - **Clef head**: Cloudflare's design. A row's state and all its questions are one prompt, and a small new network
    (the head, about 30M parameters on a 2B model) reads the model's hidden states at each question and option and
    scores every option: one pass per row. A new head trains alone for the first 10% of the steps, at its own learning
    rate (Head learning rate, default 1e-4), then the model joins in. Picking Clef sets the model's learning rate to
    5e-5: from a plain model at letters' 2e-4, Clef learned nothing (the model's features moved faster than the new head
    could follow), while 5e-5 learned faster than letters. A Clef release as the base model brings its
    trained head along. The head is saved as Clef releases ship it (`joint_head_config.json`,
    `joint_head.safetensors`), so Cloudflare's own code loads the result. The prompt holds every question, so give it
    room in Max length.
  - Validation scores every question by its type, whatever the dataset names them, the same way for both methods so
    they can be compared:
    - yes/no: AUROC, the catch rate (yes cases caught when at most 1% of no cases may be flagged; **False alarms
      allowed** under Checkpoints changes the 1%), and the false alarms at 50%;
    - choice: accuracy and macro-F1 (F1 per answer, averaged, so a rare answer counts as much as a common one; only
      answers the rows really have count, so a stray pick of one they never have is a miss, not an answer scoring 0);
    - score: exact accuracy and the mean absolute error (how many levels off, on average).
  - Over all questions: loss, accuracy and calibration error, the averages of each type, and **Overall**: every
    question's own 0-to-1 score (AUROC, macro-F1, or 1 minus the absolute error over the levels' range), averaged.
  - The run page has a **By question** table for any check or the final test, with how many rows asked each question
    and how often each answer came up. An answer seen fewer than 30 times shows in amber: the scores beside it are
    rough.
- **Language models** learn chat (only the assistant's replies, each with the token that ends its turn, so the model
  learns to stop), prompt and completion (only the completion, ended the same way), or plain text. Plain text and
  prompts start with the model's beginning-of-text token when its tokenizer uses one (Llama, Gemma; Qwen has none).
  A row longer than Max length is cut to fit; the run page counts them, since a cut reply loses its ending.
- **Full** fine-tuning keeps the weights in bf16, about 8 bytes a parameter with AdamW's two moments plus the
  activations, so a 2B model fits a 32 GB card. bf16 can't hold a small update: a step of 2e-5 on a weight near 0.02
  is below its resolution, so a plain optimizer rounds most updates away (on Qwen3.5-0.8B half the weights didn't
  move at all in a step at 2e-5) and rounds the rest up to a whole unit. anvil's optimizer (`anvil/optim.py`) does
  the maths in fp32 and rounds the results into bf16 stochastically, up or down with the probability of where the
  value lies, so the weights and Adam's moments are right on average and small updates add up over steps. **Weights
  while training** under Advanced puts the weights in fp32 instead, as mixed precision does, at twice the memory
  (about 1B parameters on a 32 GB card); the finished model is saved in bf16 either way. The form's learning rate
  for Full is 2e-5 (1e-5 with the Clef head), a tenth of LoRA's.
- Checkpoints keep the trainable weights and the optimizer, so **Stop** loses at most one step and **Resume** continues
  exactly where it stopped, after a crash too.
- **Resume** and **Keep training** (below) ask how many epochs the run makes in all, so it can train longer or shorter
  than it was started with, and a run that has finished can carry on with more. New epochs lift a Max steps limit; a
  run that Max steps cut short can also resume at that length. The learning rate's schedule then runs to the new last
  step, while its warmup (and a new Clef head's steps alone) stay as they were. Where the schedule is higher than where
  the run left off (a finished run's has decayed to nothing), the learning rate climbs back to it over the warmup's
  steps instead of jumping.
- A trainer that can't get to the end of its step after Stop (stuck in a GPU collective or other native code) is ended
  outright 10 minutes later, or at once with **Stop now**, which the run page offers while it waits. That loses the
  steps since the newest checkpoint, which Resume trains again. A checkpoint being saved always gets to finish, and a
  run you've stopped is never started again by itself, even if its trainer crashes after Stop.
- A trainer that crashes in native code (a segfault, or memory corruption in the GPU libraries) starts again by itself
  from the newest checkpoint, losing only the steps since. After three crashes in a row from the same checkpoint it
  gives up and the run fails; Resume tries again.
- A trainer can also freeze: a card stops answering, the GPUs go idle, nothing errors, and the run page still says
  Training. The trainer reports at least once a minute while it works, so one that has been quiet for 15 minutes is
  ended and started again the same way (ending it is also what gets the kernel to reset a hung card). Saving a
  checkpoint gets 30 minutes, and loading the model isn't watched, since a download takes as long as it takes.
- Both restarts are the supervisor's, and each run can switch it off: under Checkpoints when you start it, or on the
  run page at any time, even while it trains. Off, a crash fails the run and a freeze waits for you to Stop it. Stop
  works either way.
- A batch whose loss or gradients aren't numbers (NaN or infinity: a rare overflow, or a fault in the machine) is
  skipped on every card, so it can't spoil the weights; the log says which batch, and the run page counts them. After
  20 skipped in a row the run fails instead of skipping forever; Resume continues from the newest checkpoint.
- Skips that come and go are often a card rather than the run: one whose memory has no ECC on, or that's failing, can
  turn a gradient infinite now and then. Run each card alone on the same batches; if the skips follow one card, carry
  the run on without it (Resume with fewer cards, below). Crashes in native code that hit CPU-only processes too point
  to the computer's own memory: test it with memtest86+.
- The finished model is in the run's folder: `model/adapter` and `model/merged` (for LoRA), or `model/merged`. Its
  `anvil.json` says which checkpoint it is and what it scored on the test set.
- **Trained rows** show whether the model learned the pattern or just its training rows. Every check also scores rows
  the model trained on early in its first epoch, as many of each kind as validation has, halved (a decision question,
  or a Clef row's set of questions), so a check takes about 1.5 times as long; the final test does the same with the
  picked checkpoint. Their loss shows on the training chart, in the checks table and, per question, under By question,
  which marks a question in amber when its loss on new rows is at least 0.1 higher and 1.5 times its loss on the
  trained rows: the model has memorised it, and more training won't help. The rows are the first training learns
  from, listed in the run's `trained_rows.json`, and checks before training has learned from all of them skip it.
- **Delete** (on the page of a run that has finished, failed or stopped) removes the run for good: checkpoints, the
  finished model, metrics and the log. It says how much space that frees and asks first. A run that's still training,
  whose judge is still checking, or that's being exported or uploaded can't be deleted.

### Exporting as GGUF

- **Export as GGUF** on a run's page (once it has finished or stopped): pick the finished model or any checkpoint
  still kept (each shows its validation score), pick the types (BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M; sizes are
  estimated), and export. The files appear on the page to download, with an `anvil.json` that says which version they
  are, how it scored and how to prompt it. They're in the run's folder under `exports/<version>/`.
- It runs on the CPU, so the GPUs stay free, with llama.cpp v0.7.0 (built into the image): its converter writes a
  full-quality GGUF, and `llama-quantize` makes the smaller types from it. A LoRA checkpoint is merged into the base
  model first.
- Letters models become ordinary GGUFs (prompt them the letters way). Clef models become decision GGUFs with their
  head: llama-server answers SystemOne requests with them at `/v1/systemone`; give it batch sizes that fit your longest
  input (`-c 8192 -b 8192 -ub 8192`), since a decision model reads a whole input in one batch.
  [DECISION-MODELS.md](DECISION-MODELS.md) walks through one, decision-0.8B: starting the server, a request and its answers, the questions it was trained on,
  and loading it in Python without llama.cpp.
- Checked on 60 validation questions each: BF16, Q8_0 and Q4_K_M files of a letters model, and a Q8_0 Clef file, all
  gave the same answers as the model they came from (probabilities within 0.001 to 0.013 on average).

### Uploading to Hugging Face

- **Upload to Hugging Face** on a run's page (once it has finished or stopped) sends a version, the finished model or
  any checkpoint still kept, to the account whose token is in Settings: its weights, its GGUF files from Export as
  GGUF, or both. Pick **Private** or **Public**.
- The weights are the whole model in Hugging Face format (safetensors; a LoRA checkpoint is merged into the base model
  on the way), or for LoRA the adapter alone. A Clef model's head files go along, as Cloudflare's releases ship them.
- The GGUF files go to a repo of their own, `<name>-GGUF`, Hugging Face's usual way; give the weights' repo instead to
  keep them together. Repos can be on your account or an organisation's that the token may write to.
- A repo that doesn't exist yet is made. One that does gets the files added (replacing any of the same name) and is
  set to the visibility picked; the page says which will happen before you upload.
- Each repo gets a model card (`README.md`): what the model is and what it was fine-tuned from, how it scored on the
  test set (each question's score too, for decision models), how to use it, and how it was trained. It never names
  this machine's folders. A card you've edited on Hugging Face stays as it is: anvil only replaces a card it wrote
  that nobody has changed since.
- It runs as a process of its own on the CPU, so a restart of the server doesn't cut it off, with progress on the page.
  A repo's files change only once its upload has finished (one commit per repo), so **Cancel**, or a failed upload,
  leaves them as they were. The details are in the run's folder under `uploads/<when>/`.
- It needs a token that can write: a write token, or a fine-grained one allowed to write to your account or the
  organisation. A read token can't upload.

### Several GPUs

- Every GPU you tick trains. Each card holds a whole copy of the model and takes its own share of every batch, and
  after each step the cards average their gradients, so the copies stay identical (data parallel, through torchrun;
  `anvil/parallel.py`). Batch size is per card, so two cards learn from twice the rows per step and an epoch takes
  half the steps.
- Process 0 runs on a card that doesn't drive a screen if there is one. It writes the run's files, saves checkpoints
  and finishes the run. Stop reaches every card, and they agree every step on whether to stop.
- A free GPU you didn't tick becomes the judge (below). Without one, the training cards pause at each checkpoint and
  check it together, each scoring its share of the validation rows. Untick "Use GPU N for checks" to keep that card
  free, for another run say: two experiments side by side, one per card.
- A run splits every batch into one share per card it started on, and keeps that split. Resume can carry on with fewer
  cards, as long as they divide it (a two-card run on one card, say, when the other card faults): each card takes
  several shares in turn and averages them, so the run learns what it would have on all of them, only slower. The
  run's `gpus` in its `config.json` say which cards Resume uses.
- A card that drives your screen can train too, but the desktop may stutter. Close video calls and games on it until
  the run ends: on an R9700 a two-card run once hung the screen card during a video call (it needed a reboot); with
  the call closed, the same run was fine.

### The judge: picking the best checkpoint

- Every checkpoint gets a validation check, and so does the model before training (step 0, shown as the starting
  point, never picked). **Pick best by** names the number that ranks them; the run page highlights it. It offers what
  the validation data can score: AUROC and the catch rate need yes/no questions, macro-F1 choice questions, the
  absolute error score questions, and Overall is there when the data mixes types. Language models are picked by
  validation loss.
- If there's a free GPU you didn't tick, a separate process, the judge, does the checks there, so training never
  pauses for them. Without one, training pauses briefly at each checkpoint. If the judge crashes, the trainer takes
  over the checks.
- Only the best **Keep best** checkpoints are kept, plus the newest (Resume starts there) and any not checked yet.
- **Stop after no gain for** N checks ends training early once the score hasn't improved for N checks in a row
  (0 turns this off). Only the checks after the warmup count, and for a new Clef head after its steps alone too:
  until then the scores say little about where the model is going.
- **Least gain that counts**: a check beats the best only by more than this, in the score's own units (0, the
  default, counts any rise, as Hugging Face's and Keras's early stopping do by default). Every score comes with ± one
  standard error, from a bootstrap over the validation rows; the run page shows it beside the picked score. Two checks
  within about one of each other are a toss-up, so a gain of that size is noise: set the least gain to about the ±
  and the run neither stops late for a drift nor picks a checkpoint for a gain that wasn't one.
- **Keep training**, on a run that has finished, carries on from the newest checkpoint: one that stopped early before
  its last step, or, given more epochs, one that got there. It asks again: stop early after N more checks without a
  gain, or train to the last step.
- When training ends, the best checkpoint becomes the model and is scored once on the test set (up to 5,000 rows).
  The test set plays no part in picking, so that's the fair number to quote.

The image compiles `causal_conv1d` for your GPU (`ANVIL_GPU_ARCH` in `.env`) and includes flash-linear-attention, the
fast kernels for linear-attention models such as Qwen3.5. On an R9700 that makes Qwen3.5 training about 22% faster.
`ANVIL_SAFE_KERNELS=1` switches both off if a model misbehaves.

## Accounts and Hugging Face

- anvil starts with one account, `admin`. Change its password in Settings (the gear, top right).
- Paste your Hugging Face token in Settings. anvil checks it with Hugging Face, keeps it on the server, and never
  shows it again. Public models work without a token; gated ones (like Llama) need it, and uploading needs one that
  can write.
- Forgot the password? Put a new one in `ANVIL_PASSWORD`, add `ANVIL_RESET_PASSWORD=1`, restart, then remove that
  line again.

## Code

| Path | What |
|---|---|
| `docker/` | The image (`Dockerfile`), plus the scripts the containers run |
| `server/anvil/` | The Python server: accounts and login, GPUs, the Hugging Face search, model checks, datasets, runs |
| `server/anvil/train.py` | The trainer process; `tasks.py` holds what each kind of model learns and how it's scored |
| `server/anvil/judge.py` | The judge process: checks checkpoints on a spare GPU; `picking.py` ranks them and decides when to stop |
| `server/anvil/parallel.py` | Training on several GPUs at once: one process per card, gradients averaged every step |
| `server/anvil/export.py` | Exporting a version of a run as GGUF with llama.cpp: merge, convert, quantize |
| `server/anvil/upload.py` | Uploading a version of a run to Hugging Face: its weights and GGUF files, with model cards |
| `server/anvil/pipelines.py` | Pipelines: the graph, its checks, storing them and their API keys, starting and stopping runs |
| `server/anvil/generate.py` | A pipeline's run, in a process of its own: rows through the steps, in parallel, until each Save has its rows |
| `server/anvil/llm.py` | Talking to an OpenAI-compatible endpoint: its models, and chat requests with retries |
| `server/anvil/templates.py` | The pipelines New pipeline offers to start from; their sample data is in `samples/` |
| `server/anvil/assistant.py` | The pipeline assistant: a turn in a process of its own, the tools it calls, and Undo |
| `server/anvil/sharing.py` | GPUs shared with a model server: the call that frees them, and holding requests while training |
| `server/anvil/supervisor.py` | Restarting a trainer after a crash in native code or a freeze, and seeing Stop through |
| `server/anvil/vendor/clef/` | Cloudflare's Clef reference code and licence (Apache-2.0), unmodified; see its `NOTICE.md` |
| `server/tests/` | `docker compose exec -w /workspace/server dev python -m pytest` |
| `web/src/` | The React screen: `Finetune.tsx`, `Runs.tsx`, `RunPage.tsx`, `Settings.tsx`, sections in `sections/` |
| `web/src/pipeline/` | The pipeline editor (React Flow): the canvas, each step's settings, preview and runs; `Pipelines.tsx` lists them |

