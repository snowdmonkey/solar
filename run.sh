#! /bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

REGISTRY="159.99.255.163:5000"


cd $BASEDIR

docker run -it --rm -v $PWD:/usr/src/app/conf -v /opt:/opt setup python setup.py $1

docker-compose up -d

