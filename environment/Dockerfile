# hash:sha256:bc7d6d9fa647dc1c6b67d11e8887a4e076787a9e9c6c0429ed16acd9c6735d97
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG API_KEY
ARG API_SECRET

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /


# fonts for gifs
RUN apt-get update -y
RUN apt install fonts-lato fonts-open-sans fonts-roboto fonts-mononoki -y

RUN pip install -U --no-cache-dir \
    matplotlib \
    numpy \
    seaborn \
    scipy \
    pytest \
    pillow \
    pandas \
    pyyaml \
    dask \
    distributed \
    aind-codeocean-utils \
    aind-data-access-api \
    aind-ophys-utils \
    pymongo \
    scikit-learn

RUN git clone https://github.com/AllenNeuralDynamics/aind-codeocean-api \
    && cd aind-codeocean-api \
    && pip install -e .

RUN git clone https://github.com/AllenNeuralDynamics/aind-ophys-data-access \
    && cd aind-ophys-data-access \
    && pip install -e .

RUN git clone https://github.com/AllenNeuralDynamics/brain_observatory_utilities_ophys \
    && cd brain_observatory_utilities_ophys \
    && pip install -e .

RUN git clone https://github.com/AllenNeuralDynamics/comb \
    && cd comb \
    && pip install -e .

RUN git clone https://github.com/AllenNeuralDynamics/lamf-analysis \
    && cd lamf-analysis \
    && pip install -e .    

# install versions for github copilot
ADD "https://github.com/coder/code-server/releases/download/v4.21.1/code-server-4.21.1-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.21.1-linux-amd64/bin/code-server  /usr/bin/code-server

COPY postInstall /
RUN /postInstall

