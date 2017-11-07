FROM solarui:base

MAINTAINER "Jason Jiao <jason.jiao@honeywell.com>"

RUN mkdir -p /var/www/html/spi

COPY . /var/www/html/spi/

RUN mkdir -p /var/www/html/spi/data \
    && rm -fr /var/www/html/spi/Dockerfile \
    && rm -fr /var/www/html/spi/frontend.iml \
    && rm -fr /var/www/html/spi/Dockerfile.base \
    && rm -fr /var/www/html/spi/build.sh \
    && mv /var/www/html/spi/docker-php-entrypoint /usr/local/bin

