# hash:sha256:bc7d6d9fa647dc1c6b67d11e8887a4e076787a9e9c6c0429ed16acd9c6735d97
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/c1-vscode:4.20.0-pylint2023.10.1-copilot1.143.601

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN mamba install -y -c conda-forge \
    python==3.11 \
    ipykernel==6.29.5 \
    && conda clean -ya

# fonts for matplotlib figures
RUN apt-get update -y
RUN apt install fonts-lato fonts-open-sans fonts-roboto fonts-mononoki -y

RUN pip install -U --no-cache-dir \
    matplotlib \
    numpy==1.26.4 \
    scipy \
    pandas \
    pillow \
    pyyaml \
    xarray \
    scikit-learn \
    tables \
    aind-data-schema==1.2.0

RUN cd / \
    && git clone https://github.com/AllenNeuralDynamics/comb \
    && cd comb \
    && pip install -e .

RUN cd / \
    && git clone https://github.com/AllenNeuralDynamics/lamf-analysis \
    && cd lamf-analysis \
    && pip install -e .

COPY postInstall /
RUN /postInstall
