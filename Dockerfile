FROM ubuntu:latest
MAINTAINER username
WORKDIR /home/projects

RUN apt-get update && apt-get install -y \
  vim \
  gcc \
  g++ \
