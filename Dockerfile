FROM ubuntu:latest

USER root
RUN apt-get update && \
apt-get install -y \
python3-pip python3 python-is-python3

WORKDIR /workdir/
COPY requirements.txt .
RUN pip install -r requirements.txt
