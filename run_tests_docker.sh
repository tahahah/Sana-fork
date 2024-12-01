#!/bin/bash

# Build the Docker image
docker build -t sana-test -f Dockerfile.test .

# Run the tests
docker run --rm sana-test python -m unittest tests/test_pacman_dataset.py -v
