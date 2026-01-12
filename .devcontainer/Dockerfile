FROM ubuntu:24.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
    apt-utils \
    dialog \
    # Install essential tools
    && apt-get -y install \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    sudo \
    cmake \
    # LibreOffice for PPTX export
    libreoffice-impress \
    # PDF tools for multi-slide PNG export
    poppler-utils \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

# Install Python 3.12 (default in Ubuntu 24.04)
RUN apt-get update \
    && apt-get -y install \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    pipx \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# Install pre-commit using pipx (the proper way in Ubuntu 24.04)
RUN pipx install pre-commit && pipx ensurepath
ENV PATH="/root/.local/bin:${PATH}"

# Install UV (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Build Filament from source for ARM64 Linux
# Pre-built binaries are x86_64 only, so we must compile from source
ENV FILAMENT_VERSION=1.68.1
ENV FILAMENT_TOOLS_DIR=/opt/filament-tools
ENV FILAMENT_WEBGL_DIR=/opt/filament-webgl

# Install build dependencies for Filament
RUN apt-get update \
    && apt-get -y install \
    clang \
    libglu1-mesa-dev \
    ninja-build \
    libxi-dev \
    libc++-dev \
    libc++abi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Emscripten SDK (needed for WebGL build)
# Filament v1.68.1 requires Emscripten 3.1.60 (NOT latest - 4.x has breaking changes)
ENV EMSDK=/opt/emsdk
ENV EMSDK_VERSION=3.1.60
RUN git clone https://github.com/emscripten-core/emsdk.git $EMSDK --depth 1 \
    && cd $EMSDK \
    && ./emsdk install ${EMSDK_VERSION} \
    && ./emsdk activate ${EMSDK_VERSION} \
    && echo 'source "/opt/emsdk/emsdk_env.sh"' >> /etc/bash.bashrc

# Download pre-built Filament WebGL files (cross-platform WASM)
RUN mkdir -p ${FILAMENT_WEBGL_DIR} \
    && cd /tmp \
    && curl -LO https://github.com/google/filament/releases/download/v${FILAMENT_VERSION}/filament-v${FILAMENT_VERSION}-web.tgz \
    && tar -xzf filament-v${FILAMENT_VERSION}-web.tgz \
    && cp filament.js filament.wasm filament.d.ts ${FILAMENT_WEBGL_DIR}/ \
    && rm -rf /tmp/filament-v${FILAMENT_VERSION}-web.tgz /tmp/filament.*

# Build matc (native) from source - pre-built binaries are x86_64 only
RUN mkdir -p ${FILAMENT_TOOLS_DIR}/bin \
    && cd /tmp \
    && git clone --depth 1 --branch v${FILAMENT_VERSION} https://github.com/google/filament.git filament-src \
    && cd filament-src \
    && CC=/usr/bin/clang CXX=/usr/bin/clang++ CXXFLAGS="-stdlib=libc++" ./build.sh -c release matc \
    && cp out/cmake-release/tools/matc/matc ${FILAMENT_TOOLS_DIR}/bin/ \
    && chmod +x ${FILAMENT_TOOLS_DIR}/bin/matc \
    && cd / \
    && rm -rf /tmp/filament-src

ENV PATH="${FILAMENT_TOOLS_DIR}/bin:${PATH}"

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create group and user, handling the case where GID/UID might already exist
RUN if getent group $USER_GID > /dev/null 2>&1; then \
        # GID exists, use existing group or create with different name \
        EXISTING_GROUP=$(getent group $USER_GID | cut -d: -f1); \
        if [ "$EXISTING_GROUP" != "$USERNAME" ]; then \
            groupmod -n $USERNAME $EXISTING_GROUP || groupadd --gid $((USER_GID + 1)) $USERNAME; \
        fi; \
    else \
        groupadd --gid $USER_GID $USERNAME; \
    fi \
    && if getent passwd $USER_UID > /dev/null 2>&1; then \
        # UID exists, modify existing user \
        EXISTING_USER=$(getent passwd $USER_UID | cut -d: -f1); \
        usermod -l $USERNAME -d /home/$USERNAME -m $EXISTING_USER 2>/dev/null || useradd --uid $((USER_UID + 1)) --gid $USERNAME -m $USERNAME -s /bin/bash; \
    else \
        useradd --uid $USER_UID --gid $USERNAME -m $USERNAME -s /bin/bash; \
    fi \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install UV for the non-root user
USER $USERNAME
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install pre-commit for the user using pipx
RUN pipx install pre-commit && pipx ensurepath

# Install Open WebUI for the user using pipx (takes a while, but pre-built in container)
RUN pipx install open-webui

# Configure npm for user
RUN mkdir -p /home/${USERNAME}/.npm-global \
    && npm config set prefix "/home/${USERNAME}/.npm-global"

ENV PATH="/home/${USERNAME}/.local/bin:/home/${USERNAME}/.npm-global/bin:${PATH}"

# Set working directory
WORKDIR /workspaces/local-llm-sim

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
