#! /bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $BASEDIR

docker run -it --rm -v $PWD:/usr/src/app/conf -v /opt:/opt setup python setup.py $1

if [ -f docker-compose.yml ] then;
	docker-compose -d


