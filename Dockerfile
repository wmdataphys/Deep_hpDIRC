ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace

# # Remove NVIDIA channel & global Arrow pins,
# # keep conda-forge first *but* retain pytorch for GPU wheels
# RUN conda config --system --remove channels rapidsai || true \
#  && conda config --system --remove channels nvidia   || true \
#  && conda config --system --remove channels pytorch  || true \
#  && conda config --add channels conda-forge \
#  && conda config --add channels pytorch              \
#  && conda config --set channel_priority strict       \
#  && rm -f $CONDA_PREFIX/conda-meta/pinned || true

# use mamba for solving
RUN conda install -y -n base -c conda-forge mamba

COPY env.yml /tmp/
RUN mamba env create -n hpdirc -f /tmp/env.yml && conda clean -afy --no-builds

ENV PATH="/opt/conda/envs/hpdirc/bin:$PATH"

RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /workspace
USER appuser

COPY . /workspace

CMD ["bash"]