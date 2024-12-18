# Build the Docker image
docker build -t pacman-train -f Dockerfile.train .

# Run the training with volume mount for hot reloading and increased shared memory
docker run --gpus all --rm --shm-size=10g -v $(pwd):/app pacman-train
