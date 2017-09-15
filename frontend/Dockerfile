FROM solarui:base

MAINTAINER "Jason Jiao <jason.jiao@honeywell.com>"

COPY . /var/www/html/

RUN mkdir -p /var/www/html/data \
    && rm -fr /var/www/html/Dockerfile \
    && rm -fr /var/www/html/frontend.iml \
    && rm -fr /var/www/html/Dockerfile.base \
    && rm -fr /var/www/html/build.sh \
    && mv /var/www/html/docker-php-entrypoint /usr/local/bin

