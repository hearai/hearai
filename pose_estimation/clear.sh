#!/bin/bash
export USER=$(id -un)
export COMPOSE_PROJECT_NAME="hearai_${USER}"
export CONTAINER_NAME="hearai_${USER}"
./docker-compose down
docker rmi "${COMPOSE_PROJECT_NAME}_openpose"
