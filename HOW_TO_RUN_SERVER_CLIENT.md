
# !Before hand!
**FIRST: If your WANDB_API_KEY is not set up yet**
1 - Open bash 
2 - run "export WANDB_API_KEY="your_secret_key"

**SECONDLY: if either port 8080 or 5000 is not availble on your localhost**
**A)** 8080 is not available!
2 - Change the *streamlit* service port of "docker-compose.yml" to be XXXX:XXXX instead of 8080 :8080 
3 - Change "Dockerfile.serving" last two commands to use XXXX instead of 8080 . (The commands to change are *EXPOSE* and *CMD[streamlit ..*)

**B)** 5000 is not available!
1 - Open ift6758/client/app.py and change all 5000 with another port of your choice. A simple ctl+f should give you the two instances to change
2 - Change the *serving* service port of "docker-compose.yml" to be XXXX:XXXX instead of 5000:5000
3 - Change "Dockerfile.serving" last two commands to use XXXX instead of 5000. (The commands to change are *EXPOSE* and *CMD[waitress-serve ..*)
4 - Change ift6758/client/serving_client.py __init__ function to use XXXX as default port in the parameters

(note: keep the dockerfiles IPs to 0.0.0.0, otherwise you be able to access outside docker)

## Run the whole thing in docker
1 - open bash
2 - navigate to project root where the dockerfiles stand
2 - run "docker-compose up"
*Note you can use "docker-compose **--build**" to force the whole rebuild
3 - You should see where the client is hosted in the last bash logged lines, just open this in your browser (replace 0.0.0.0 with 127.0.0.1 in the browser)

Check useful command at the bottom of this file if you need to clean existing images/containers

# Client features
### Get model button
Way to request to load a model for the server to use

The workspace input must be filled with
 *toma-allary-universit-de-montr-al/IFT6758.2024-A09*
The downloadable models are 
- *distance_and_angle_model*
- *angle_only_model*
- *distance_only_model*
(You should use **latest** as version, but you can find all existing versions on https://wandb.ai/toma-allary-universit-de-montr-al/IFT6758.2024-A09/artifacts)

### Ping Game button
Use to get prediction from a given gameid

You must make sure that a model is loaded in the server beforehand! Use the **get model** button if it is not the case.


# Running the server only
### Using shell scripts
1 - open bash
2 - run "./ build.sh"
3 - run "./ run.sh"
Server is now running


### using compose
1 - open bash
2 - run "docker-compose up"
Server and client should be running




# Useful docker bash commands:

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

- To remove all containers
docker container rm $(docker ps -a -q)
