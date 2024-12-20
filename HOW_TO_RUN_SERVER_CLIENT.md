
**FIRST: If your WANDB_API_KEY is not set up yet**
1 - Open bash 
2 - run "export WANDB_API_KEY="your_secret_key"


# Running the server using shell scripts
1 - open bash
2 - run "./ build.sh"
3 - run "./ run.sh"
Server is now running


# Using compose
1 - open bash
2 - run "docker-compose up"
Server and client should be running





# Useful bash commands:

- To stop all running containers:
docker stop $(docker ps -a -q)

- To list all running containers:
docker ps

- To list all images
docker images

- To remove an image
docker rmi my_image:tag

- To list all containers
docker ps -a

- To remove container
docker container rm some_id_blabla