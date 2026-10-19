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
 && python3 -m pip install --no-cache-dir --no-deps flash-linear-attention fla-core einops \
 && python3 -c "import torch; assert 'rocm' in torch.__version__, torch.__version__; print(torch.__version__)"
# flash-linear-attention: fast kernels for linear-attention models such as Qwen3.5. --no-deps so pip never swaps in a
# non-ROCm triton. On ROCm 10.1 they trained without faults; ANVIL_SAFE_KERNELS=1 switches them off if a model misbehaves.

# causal-conv1d: the fast kernel for the short convolutions in linear-attention layers, compiled for your card
# (GPU_ARCH; 2.6-7x faster than PyTorch's plain path on an R9700). Its build script asks for C++17, but PyTorch 2.14's
# headers need C++20, hence the sed. The source is pinned by checksum. Takes about 4 minutes.
ARG GPU_ARCH=gfx1201
RUN set -eux; cd /tmp; \
    f=causal_conv1d-1.7.0.tar.gz; \
    curl -fsSLO "https://files.pythonhosted.org/packages/f4/d1/eba27735f31bd5527d39a3c7f693e6ded7b844cab275db719a15ad37b6cd/$f"; \
    echo "3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b  $f" | sha256sum -c -; \
    tar xzf "$f"; cd causal_conv1d-1.7.0; sed -i 's/c++17/c++20/g' setup.py; \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE HIP_ARCHITECTURES="$GPU_ARCH" MAX_JOBS=8 \
        python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .; \
    cd /tmp; rm -rf causal_conv1d-1.7.0 "$f"; python3 -c "import causal_conv1d"

# llama.cpp (MIT) for exporting models as GGUF: its converter writes a full-quality GGUF (Qwen3.5 models, and Clef
# models with their decision head), llama-quantize shrinks it (Q8_0, Q4_K_M, ...), and llama-cli and llama-server can try
# the result. Built for the CPU only, since exporting needs no GPU. Release v0.7.0, pinned by checksum. A few minutes.
ARG LLAMA_CPP=0.7.0
RUN set -eux; cd /tmp; \
    python3 -m pip install --no-cache-dir cmake sentencepiece protobuf; \
    f="llama.cpp-v${LLAMA_CPP}.tar.gz"; \
    curl -fsSL -o "$f" "https://github.com/ggml-org/llama.cpp/archive/refs/tags/v${LLAMA_CPP}.tar.gz"; \
    echo "55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1  $f" | sha256sum -c -; \
    mkdir -p /opt/llama.cpp; tar xzf "$f" -C /opt/llama.cpp --strip-components=1 --no-same-owner; rm -f "$f"; \
    cd /opt/llama.cpp; \
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF \
          -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF; \
    cmake --build build --target llama-quantize llama-cli llama-server -j 8; \
    ./build/bin/llama-quantize --help 2>&1 | head -3 || true

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LLAMA_CPP_DIR=/opt/llama.cpp \
    LLAMA_CPP_VERSION=v${LLAMA_CPP}
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
