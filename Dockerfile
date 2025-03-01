FROM nvcr.io/nvidia/pytorch:24.05-py3
ENV FORCE_CUDA="1"

# Obtain the UID and GID of the current user to create a user with the same ID, this is to avoid permission issues when mounting local volumes.
ARG USER
ARG UID
ARG GID

# Timezone. Avoid user interaction.
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add user.
RUN groupadd -g $GID $USER \
    && useradd --uid $UID --gid $GID -m $USER \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

# Install Python and some utilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-setuptools \
    # Graphics libraries.
    libgl1-mesa-glx \
    mesa-utils \
    libglapi-mesa \
    libqt5gui5 \
    # Utilities.
    curl \
    wget \
    git \
    build-essential \
    ca-certificates \
    cmake \
    jupyter-notebook

# Copy requirements.txt and install Python packages.
USER $USER
WORKDIR /.cache


USER root
RUN pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 uninstall -y flash-attn || true
RUN pip3 install flash-attn --no-build-isolation
RUN pip3 install 'mmcv>=2.0.0rc4,<2.2.0'
RUN pip3 install 'mmdet>=3.0.0'
RUN pip3 install 'mmdet3d>=1.1.0'
RUN pip3 install torch_scatter
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade pip

USER $USER
RUN pip install 'opencv-python<4.9'
RUN pip install --upgrade timm
RUN pip install "numpy<2"

# Disable jupyter authentication
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.token = ''" >> /home/$USER/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /home/$USER/.jupyter/jupyter_notebook_config.py

# # Workaround to avoid permission issues when going to uni computers.
# USER root
# RUN echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
#     && chmod 0440 /etc/sudoers.d/$USER
# USER $USER

# Change terminal color
ENV TERM=xterm-256color
RUN echo "PS1='\[\e[1;91m\]\u\[\e[1;37m\]@\[\e[1;93m\]\h\[\e[1;37m\]:\[\e[1;35m\]\w\[\e[1;37m\] → \[\e[0m\]'" >> ~/.bashrc


WORKDIR /workspace

CMD [ "/bin/bash" ]