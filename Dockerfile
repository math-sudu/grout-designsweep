FROM dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8
RUN python3 -m pip install --no-cache-dir --break-system-packages PyYAML==6.0.2
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /work
