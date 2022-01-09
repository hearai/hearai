#!/bin/bash
export USER=$(id -un)
export CONTAINER_NAME="hearai_${USER}"
docker exec -it $CONTAINER_NAME /bin/bash
