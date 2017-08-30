#! /bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

REGISTRY="159.99.255.163:5000"

function remove()
{
	if [ "$(docker ps -q -f name=uavsolarpanel_backend_1)" ]; then
		echo 'Stopping and removing the container uavsolarpanel_backend_1 ...'
    	docker rm -fv uavsolarpanel_backend_1
    	echo 'Container removed.'
	else
		if [ "$(docker ps -aq -f name=uavsolarpanel_backend_1)" ]; then
    		echo 'Removing container uavsolarpanel_backend_1 ...'
        	docker rm uavsolarpanel_backend_1
        	echo 'Container removed.'
    	fi
	fi
}

function clean()
{
	remove
	
	if [ "$(docker images -q solarapi:latest)" ]; then
		echo 'Removing the image solarapi ...'
		docker rmi -f solarapi:latest
		echo 'Image removed.'
	fi
	echo 'Environment cleaned.'
}

function build()
{
    clean

    if [ -n "$1" ]; then
        docker build --no-cache -t solarapi:$1 .
    else
        docker build --no-cache -t solarapi .
    fi
}

function cleanbase()
{
    remove
    if [ "$(docker images -q solarbase)" ]; then
        echo 'Removing the image solarbase ...'
        docker rmi -f solarbase
        echo 'Image removed.'
    fi
    echo 'Environment cleaned.'
}

function buildbase()
{
    cleanbase
    docker build --no-cache -f Dockerfile.base -t solarbase .
}

cd $BASEDIR

if [ "$1" == "clean" ]; then
	clean
elif [ "$1" == "build" ]; then
    clean
    build $2
elif [ "$1" == "buildbase" ]; then
    cleanbase
    buildbase
elif [ "$1" == "remove" ]; then
    remove
else
    echo "Usage: build.sh COMMAND"
    echo "Available Commands:"
    echo "build	Clean and build the new image"
    echo "buildbase Build the base image"
    echo "clean	Clean the environment"
    echo "remove	Remove the portal container and files"
fi
