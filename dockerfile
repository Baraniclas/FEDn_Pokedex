# base image with Python, 3.9 supposed to be stable might update later
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Install Git
RUN apt-get update && apt-get install -y git

# Install dependencies
RUN pip install --no-cache-dir fedn

# set work directory
WORKDIR /app

# fetch data
COPY /data/water.csv .