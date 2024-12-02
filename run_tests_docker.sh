#!/bin/bash

# Build the Docker image
docker build -t pacman-test -f Dockerfile.test .

# Run the tests with volume mount for hot reloading and increased shared memory
docker run --gpus all --rm --shm-size=1g -v $(pwd):/app pacman-test python -m unittest tests/test_pacman_dataset.py -v