# =========================
# Ubuntu 24.04 + micromamba
# Base image: common stack preinstalled (NO per-run installs)
# Runtime: user auto-mapped via entrypoint (UID/GID)
# =========================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul

# 0) OS packages (runtime libs + build tools + gosu for privilege drop)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 curl ca-certificates git wget build-essential \
    libgl1 libglib2.0-0 \
    libjpeg-turbo8 libtiff6 libpng16-16 \
    openslide-tools \
    libopenjp2-7 libopenjp2-tools \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# 1) micromamba 설치 (binary only)
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
RUN set -eux; \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba; \
    chmod +x /usr/local/bin/micromamba

SHELL ["/bin/bash", "-lc"]

# 2) Base conda env (image-level, shared)
#    - Keep this environment stable for everyone.
RUN set -euxo pipefail; \
    micromamba create -y -r ${MAMBA_ROOT_PREFIX} -n hist python=3.10; \
    micromamba run   -r ${MAMBA_ROOT_PREFIX} -n hist python -V

# 3) Scientific stack via conda-forge (prefer conda-forge)
RUN set -euxo pipefail; \
    micromamba install -y -r ${MAMBA_ROOT_PREFIX} -n hist \
      -c conda-forge --strict-channel-priority \
      numpy pandas matplotlib scipy scikit-image pillow opencv \
      openslide-python h5py shapely tqdm \
      ipykernel jupyter jupyterlab; \
    micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist python -c "import numpy, skimage, PIL, cv2, openslide, h5py; print('Sci stack OK:', numpy.__version__)"

# 4) cucim (CUDA 12.1)
RUN set -euxo pipefail; \
    micromamba install -y -r ${MAMBA_ROOT_PREFIX} -n hist \
      -c rapidsai -c conda-forge -c nvidia --strict-channel-priority \
      "cucim=24.08.*" "cuda-version=12.1" && \
    micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist python -c "import cucim; print('cucim OK:', cucim.__version__)"

# 5) Pinned pip set (optional: keep your pinned versions)
RUN set -euxo pipefail; \
  micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist python -m pip install --upgrade \
    'pip<25' 'setuptools<75' wheel && \
  micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist python -m pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4 numba==0.58.1 llvmlite==0.41.1 'scipy<1.12' umap-learn==0.5.6 && \
  micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist python - <<'PY'
import numpy, numba, umap, scipy
print("Pinned OK:", numpy.__version__, numba.__version__, scipy.__version__)
PY

# 6) PyTorch cu121 + extras (image-level)
RUN set -euxo pipefail; \
    micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist pip install \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121; \
    micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist pip install \
      torchaudio --index-url https://download.pytorch.org/whl/cu121 || true; \
    micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist pip install \
      'xformers==0.0.29' 'timm>=1.0.8' loguru natsort timm_ctp; \
    micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist python -c "import torch; print('torch:', torch.__version__, 'cuda?', torch.cuda.is_available())"

# 7) tiatoolbox (image-level)
RUN set -euxo pipefail; \
    micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist pip install --no-cache-dir 'tiatoolbox==1.6.0' && \
    micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist python -c "import tiatoolbox; print('tiatoolbox:', tiatoolbox.__version__)"

# 8) Register jupyter kernel (in env)
RUN set -euxo pipefail; \
    micromamba run -r ${MAMBA_ROOT_PREFIX} -n hist python -m ipykernel install \
      --prefix ${MAMBA_ROOT_PREFIX}/envs/hist \
      --name hist --display-name 'Python (hist)'

# 9) Make env available without activation
ENV PATH=/opt/micromamba/envs/hist/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/micromamba/envs/hist/lib:${LD_LIBRARY_PATH:-}

# 10) Entrypoint (runtime user mapping)
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
