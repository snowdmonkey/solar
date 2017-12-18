FROM solarapi:base

MAINTAINER "Jason Jiao <jason.jiao@honeywell.com>"

COPY prototype/ /usr/src/app

ENV PYTHONPATH /usr/src/app

RUN mkdir -p /usr/src/app/data \
    && mkdir -p /usr/src/app/sqlite
	
EXPOSE 5000

CMD [ "python", "app/main.py" ]
