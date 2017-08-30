FROM solarbase:latest

MAINTAINER "Jason Jiao <jason.jiao@honeywell.com>"

COPY prototype/ /usr/src/app

ENV PYTHONPATH /usr/src/app

RUN mkdir -p /usr/src/app/data

CMD [ "python", "app/main.py" ]
