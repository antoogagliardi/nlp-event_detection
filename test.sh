#!/bin/bash

# initial check

if [ "$#" != 1 ]; then
    echo "$# parameters given. Only 1 expected. Use -h to view command format"
    exit 1
fi

if [ "$1" == "-h" ]; then
  echo "Usage: $(basename "$0") [file to evaluate upon]"
  exit 1
fi

test_path=$1

# delete old docker if exists
docker ps -q --filter "name=nlp-event_detection" | grep -q . && docker stop nlp-event_detection
docker ps -aq --filter "name=nlp-event_detection" | grep -q . && docker rm nlp-event_detection

# build docker file
docker build . -f Dockerfile -t nlp-event_detection

# bring model up
docker run -d -p 12345:12345 --name nlp-event_detection nlp-event_detection

# perform evaluation
/usr/bin/env python docker/evaluate.py "$test_path"

# stop container
docker stop nlp-event_detection

# dump container logs
docker logs -t nlp-event_detection > logs/server.stdout 2> logs/server.stderr

# remove container
docker rm nlp-event_detection