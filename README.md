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
share an input stay on one side of the split. For decision data it also says which questions the validation set
asks, by type, since that decides what the checks can score. The result is cached: preparing the same data with the
same settings again is instant.

## Training

**Start training** checks the whole form, then starts a separate trainer process. The run gets its own page (Runs,
top left) with progress (steps and epochs), speed, time left, GPU memory, the loss and validation charts, a table of
every validation check, the final test, the checkpoints and the log.

- An **epoch** is one pass over the training rows. A **step** learns from one batch: batch size × the number of
  training cards × gradient accumulation rows. The form shows how many steps an epoch takes; **Max steps** cuts a run
  short (handy for quick tests).

- **Decision models** train one of two ways (Method, under Train as):
  - **Letters**: each question lists its options as A, B, C… and the model's scores for the letters are its answer.
    Option order is shuffled per row so no letter is favoured; validation asks every question in both orders and
    averages. Nothing is added to the model. One pass per question.
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
    - choice: accuracy and macro-F1 (F1 per answer, averaged, so a rare answer counts as much as a common one);
    - score: exact accuracy and the mean absolute error (how many levels off, on average).
  - Over all questions: loss, accuracy and calibration error, the averages of each type, and **Overall**: every
    question's own 0-to-1 score (AUROC, macro-F1, or 1 minus the absolute error over the levels' range), averaged.
  - The run page has a **By question** table for any check or the final test, with how many rows asked each question
    and how often each answer came up. An answer seen fewer than 30 times shows in amber: the scores beside it are
    rough.
- **Language models** learn chat (only the assistant's replies), prompt and completion (only the completion), or plain
  text.
- Checkpoints keep the trainable weights and the optimizer, so **Stop** loses at most one step and **Resume** continues
  exactly where it stopped, after a crash too.
- A trainer that can't get to the end of its step after Stop (stuck in a GPU collective or other native code) is ended
  outright 10 minutes later, or at once with **Stop now**, which the run page offers while it waits. That loses the
  steps since the newest checkpoint, which Resume trains again. A checkpoint being saved always gets to finish, and a
  run you've stopped is never started again by itself, even if its trainer crashes after Stop.
- A trainer that crashes in native code (a segfault, or memory corruption in the GPU libraries) starts again by itself
  from the newest checkpoint, losing only the steps since. After three crashes in a row from the same checkpoint it
  gives up and the run fails; Resume tries again.
- A batch whose loss or gradients aren't numbers (NaN or infinity: a rare overflow, or a fault in the machine) is
  skipped on every card, so it can't spoil the weights; the log says which batch, and the run page counts them. After
  20 skipped in a row the run fails instead of skipping forever; Resume continues from the newest checkpoint.
- Skips that come and go, like crashes in native code, are often the machine rather than the run: when a replay of the
  same steps from the same checkpoint trains cleanly, test the memory (memtest86+), especially with four DIMMs or a
  memory overclock (XMP/EXPO).
- The finished model is in the run's folder: `model/adapter` and `model/merged` (for LoRA), or `model/merged`. Its
  `anvil.json` says which checkpoint it is and what it scored on the test set.
- **Delete** (on the page of a run that has finished, failed or stopped) removes the run for good: checkpoints, the
  finished model, metrics and the log. It says how much space that frees and asks first. A run that's still training,
  or whose judge is still checking, can't be deleted.

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
- Checked on 60 validation questions each: BF16, Q8_0 and Q4_K_M files of a letters model, and a Q8_0 Clef file, all
  gave the same answers as the model they came from (probabilities within 0.001 to 0.013 on average).

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
- **Keep training**, on a run that stopped early before its last step, carries on from the newest checkpoint. It asks
  again: stop early after N more checks without a gain, or train to the last step.
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
| `server/anvil/parallel.py` | Training on several GPUs at once: one process per card, gradients averaged every step |
| `server/anvil/export.py` | Exporting a version of a run as GGUF with llama.cpp: merge, convert, quantize |
| `server/anvil/vendor/clef/` | Cloudflare's Clef reference code and licence (Apache-2.0), unmodified; see its `NOTICE.md` |
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
5. **Several GPUs** (done): every ticked card trains on its share of each batch; checks are shared, or run on a
   free unticked card.
6. **The Clef head** (done): decision models the Clef way, next to letters, on one GPU or several, with the judge.
7. **GGUF export** (done): the finished model or any kept checkpoint, as BF16 to Q4_K_M, for llama.cpp and the like.

Since then: every question scored by its type (yes/no, choice, score) for any dataset, a By question table, Pick best
by that follows the data (with Overall for mixed data), and the catch rate's false alarm allowance as a setting.
