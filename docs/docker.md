# Docker

PXA 4.0 ships one engine image, `ghcr.io/poisonxa16/pxa:v4.0`. It carries the same binaries as the Linux
tarball (and no models: mount a folder at `/models`). It runs as a normal user and answers a health check
on port 8080. The tag `latest` moves only with the public release; `v4.0` stays fixed.

You need Docker and the NVIDIA Container Toolkit, and an NVIDIA driver 570 or newer on the host. The cards
are Tesla P100, Tesla V100 and the GTX 10-series (sm_60, sm_61 and sm_70).

```bash
docker run -d --name pxa --gpus '"device=0,1"' --shm-size=1g -p 8080:8080 \
    -v /path/to/models:/models:ro \
    -v pxa-cache:/work/.cache/pxa -e PXA_CACHE_DIR=/work/.cache/pxa \
    ghcr.io/poisonxa16/pxa:v4.0
```

- [Install PXA](install.md#container): the same command, with the notes for `--gpus` and `--runtime=nvidia`.
- [`docker/COMPOSE.md`](../docker/COMPOSE.md): the Compose file, four commands from nothing to an endpoint.
- [Guide 07](tutorials/07-docker-compose.md): the Compose file explained line by line.
- PXA Control in the container is opt-in: add `-e PXA_CONTROL=1 -p 7777:7777`.

The image is built from the release tarball; this repository's other Dockerfiles are for building the
engine yourself ([`BUILD-FROM-SOURCE.md`](../BUILD-FROM-SOURCE.md)) and are not the supported way to run PXA.
