FROM solarui:base

MAINTAINER "Jason Jiao <jason.jiao@honeywell.com>"

COPY . /var/www/html/

RUN mkdir -p /var/www/html/data \
    && rm -fr /var/www/html/Dockerfile /var/www/html/frontend.iml \
    && mv /var/www/html/docker-php-entrypoint /usr/local/bin

