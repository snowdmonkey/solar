FROM solarapi:base

MAINTAINER "Jason Jiao <jason.jiao@honeywell.com>"

COPY prototype/ /usr/src/app

ENV PYTHONPATH /usr/src/app

RUN mkdir -p /usr/src/app/data
	
EXPOSE 5000

CMD [ "python", "-m", "spi_app"]
