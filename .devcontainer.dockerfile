FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y \
    python3 \
    python3-pip \
    sudo \
    && apt-get clean

# Snippet from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME=user-name-goes-here
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME

ADD . /tmp/tensor_tracker
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /tmp/tensor_tracker/requirements-dev.txt
RUN pip install --upgrade jupyter ipywidgets
