# anvil's image: AMD's ROCm 10.1 + PyTorch 2.14, plus the Python server, the training libraries and Node for the screen.
# ROCm 10.1 carries AMD's fix for the hipBLASLt memory faults on RDNA4 cards during training (ROCm/TheRock#7699).
# Built by compose (`docker compose build`); the source is mounted at /workspace, not copied in.
FROM rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0

# Node from nodejs.org's official builds, checked against the release's SHASUMS256.txt.
ARG NODE_MAJOR=24
RUN set -eux; cd /tmp; \
    curl -fsSLO "https://nodejs.org/dist/latest-v${NODE_MAJOR}.x/SHASUMS256.txt"; \
    f="$(grep -oE "node-v${NODE_MAJOR}\.[0-9]+\.[0-9]+-linux-x64\.tar\.xz" SHASUMS256.txt | head -1)"; \
    curl -fsSLO "https://nodejs.org/dist/latest-v${NODE_MAJOR}.x/$f"; \
    grep " $f\$" SHASUMS256.txt | sha256sum -c -; \
    tar -xJf "$f" -C /usr/local --strip-components=1 --no-same-owner; \
    rm -f "$f" SHASUMS256.txt; node --version

# The image only has python3. pip keeps the image's ROCm torch because nothing here pins another one.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python \
 && python3 -m pip install --no-cache-dir \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "huggingface_hub>=0.26" \
        "transformers>=5.15" peft datasets pandas pyarrow safetensors datasketch \
        pytest httpx \
 && python3 -c "import torch; assert 'rocm' in torch.__version__, torch.__version__; print(torch.__version__)"

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /workspace

# The container's default user, "anvil", with your user and group ids (ANVIL_UID/ANVIL_GID in .env), so everything it
# writes into the mounted folders belongs to you. The Ubuntu base already has a user with id 1000; it's renamed.
ARG ANVIL_UID=1000
ARG ANVIL_GID=1000
RUN set -eux; \
    if getent group "$ANVIL_GID" >/dev/null; then groupmod -n anvil "$(getent group "$ANVIL_GID" | cut -d: -f1)"; \
    else groupadd -g "$ANVIL_GID" anvil; fi; \
    if getent passwd "$ANVIL_UID" >/dev/null; then \
        usermod -l anvil -g "$ANVIL_GID" -d /home/anvil -m "$(getent passwd "$ANVIL_UID" | cut -d: -f1)"; \
    else useradd -u "$ANVIL_UID" -g "$ANVIL_GID" -m -d /home/anvil -s /bin/bash anvil; fi
USER anvil
