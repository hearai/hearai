#!/bin/bash
if [ ! -f ./docker-compose ]; then
    source ./download_compose.sh
fi
./docker-compose up -d
docker ps -a | grep hearai_openpose
source ./enter.sh
