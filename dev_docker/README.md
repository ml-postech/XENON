# Docker setup

1. Build a docker image using `Dockerfile`

```shell
docker build --tag xenon:0.1 .
```

2. Run a docker container

```shell
docker run --rm --gpus '"device=0,1"' \
 -v {path/to/this/repo}:/app/repo \
 -v ~/.cache/huggingface:/app/LLM \
 --shm-size=32g \
 -it sjlee1218/xenon:0.1  /bin/bash
```

3. Inside the container, run the following commands:

```shell
cd /app/repo/minerl

git config --global --add safe.directory /app/repo
git config --global --add safe.directory /app/repo/
git config --global --add safe.directory /app/repo/minerl
git config --global --add safe.directory /app/repo/minerl/minerl
git config --global --add safe.directory /app/repo/minerl/minerl/MCP-Reborn

pip install -e . # This is very slow

cd /app/repo/minerl/minerl
rm -rf MCP-Reborn
tar -xzvf MCP-Reborn.tar.gz --no-same-owner # Install MCP-Reborn using `gdown 1GLy9IpFq5CQOubH7q60UhYCvD6nwU_YG`
cd MCP-Reborn
./gradlew clean build shadowJar

cd /app/repo
pip install -e .
```

4. From outside the Docker container, commit the docker container to an image
```shell
# outside the container (i.e. new shell from the host machine)
docker commit {docker_container_id} xenon:latest
```

Then, you can run XENON using the xenon:latest docker image!
