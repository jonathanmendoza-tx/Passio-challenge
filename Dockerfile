FROM tensorflow/tensorflow:latest

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /model

COPY /model/ /model/

RUN apt-get update && apt-get install -y curl

ENV LANG C.UTF-8

CMD [ "python", "model.py"]