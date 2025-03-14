# base image with Python, 3.9 supposed to be stable might update later
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Install Git
RUN apt-get update && apt-get install -y git

# set work directory
WORKDIR /app

# fetch data
COPY /data/water.csv .

# Copy evironment.yml into image
COPY environment.yml .

# Update the conda environment in the base image with the dependencies in the environment.yml
RUN conda env update --file environment.yml --name base
