ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace

# use mamba for solving
RUN conda install -y -n base -c conda-forge mamba

COPY env.nobuilds.yml /tmp/
RUN mamba env create -n hpdirc -f /tmp/env.nobuilds.yml && conda clean -afy 

ENV PATH="/opt/conda/envs/hpdirc/bin:$PATH"

RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /workspace
USER appuser

COPY . /workspace

CMD ["bash"]