# hash:sha256:977e89ee1b4bbd4811c635bf0a6c97f44c9ef027428be416cd91c565b2fb2493
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG CUSTOM_KEY

RUN conda install -y \
        rich \
    && conda clean -ya

RUN pip install --no-cache-dir openpyxl

COPY postInstall /
RUN /postInstall

