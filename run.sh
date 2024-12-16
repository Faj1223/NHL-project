#!/bin/bash

#echo "TODO: fill in the docker run command"

IMAGE_NAME="serving:v1.0"
CONTAINER_NAME="serving_container"

docker run --rm -d -p 5000:5000 --name $CONTAINER_NAME $IMAGE_NAME

# add env var if needed like this:
#docker run --rm -d -p 5000:5000 -e ENV_VAR=value --name $CONTAINER_NAME $IMAGE_NAME

