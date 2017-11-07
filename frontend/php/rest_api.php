<?php
/**
 * Created by PhpStorm.
 * User: H230809
 * Date: 10/18/2017
 * Time: 7:43 PM
 */


include ('./httpful.phar');

function getDefectsByDate($date)
{
    $date = base64_decode($date);
    $response = \Httpful\Request::get('http://solarapi:5000/defects?date='.$date)->send();
    if($response->code == 200)
    {
        echo $response->body;
    }
    else
    {
        echo '[]';
    }
}

function getAPIHost()
{
    $host = getenv('HOST');
    $port = getenv('PORT');
    if (strcmp($port, '80') != 0)
    {
        return $host.':'.$port;
    }
    else
    {
        return $host;
    }
}
