#!/bin/bash

# echo "TODO: fill in the docker build command"


# # Set up your wandb key here **REMOVE YOURS ONCE ITs DONE**
# echo "setting up env var.."
# export WANDB_API_KEY="your_key"
# 
# # Verify the key is set up:
# echo "building server.."
# echo "wandb api key is: "
# echo $WANDB_API_KEY


# docker build -f <FILENAME> -t <TAG>:<VERSION> .
# eg: 
echo "docker build.."
docker build -f Dockerfile.serving --build-arg WANDB_API_KEY=$WANDB_API_KEY -t serving:latest .
