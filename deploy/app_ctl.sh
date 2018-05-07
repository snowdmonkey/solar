#! /bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function stop()
{
	cd $BASEDIR
	docker-compose -f $BASEDIR/compose/{customerid}.yml -p {customerid} down
}

function start()
{
	cd $BASEDIR
	docker-compose -f $BASEDIR/compose/{customerid}.yml -p {customerid} up -d
}

function clean()
{
	docker rmi solarui:{customerid}
	docker rmi solarapi:latest
}

function cleanui()
{
	docker rmi solarui:{customerid}
}

function cleanuibase()
{
	docker rmi uibuilder
}

function buildall()
{
	$BASEDIR/uav-solar-panel/backend/build.sh buildbase
	$BASEDIR/uav-solar-panel/backend/build.sh build
	cleanui
        cleanuibase
	buildui
}

function buildui()
{
	cd $BASEDIR/web-app
	docker build -f Dockerfile.base -t uibuilder .
	docker run --rm -v $PWD:/dist uibuilder /ng-app/build.sh {customerid}
	docker build --build-arg BRAND={customerid} -t solarui:{customerid} .
	rm -f dist.tar.gz
	cd $BASEDIR
}

function buildapi()
{
	$BASEDIR/uav-solar-panel/backend/build.sh build
}

function build()
{
    pull
    buildapi
    buildui
}

function prune()
{
	docker rmi $(docker images -f "dangling=true" -q)
}

function pull()
{
	cd $BASEDIR/uav-solar-panel
	git pull
	cd $BASEDIR/web-app
	git pull
	cd $BASEDIR
}

function clone()
{
    git clone git@159.99.234.54:uav/uav-solar-panel.git
    git clone git@159.99.234.54:solar-panel-ui/web-app.git
}

$1
