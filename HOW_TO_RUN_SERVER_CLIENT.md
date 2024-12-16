# Using shell scripts:

**If you WANDB_API_KEY is not set up yet**
Open build.sh, uncomment the part to set up the key.
Enter your key. (Don't forget to remove the secret key after)

1 - open bash
2 - run "./ build.sh"
3 - run "./ run.sh"
Server is now running


# Using compose
**If you WANDB_API_KEY is not set up yet**
Open build.sh, uncomment the part to set up the key.
Enter your key. (Don't forget to remove the secret key after)

1 - open bash
2 - run "docker-compose up"
Server and client should be running





# Useful bash commands:

- To stop all running containers:
docker stop $(docker ps -a -1)

- To list all running containers:
docker ps