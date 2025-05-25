ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /workspace

COPY env.yml /tmp/
RUN conda env update -n base -f /tmp/env.yml && conda clean -afy

COPY . /workspace

CMD ["bash"] 