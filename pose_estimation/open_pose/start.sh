#!/bin/bash
export UID=$(id -g)
export GID=$(id -u)
export USER=$(id -un)
export COMPOSE_PROJECT_NAME="hearai_${USER}"
export CONTAINER_NAME="hearai_${USER}"

if [ ! -f ./docker-compose ]; then
    source ./download_compose.sh
fi
./docker-compose up -d
docker ps -a | grep hearai
source ./enter.sh
