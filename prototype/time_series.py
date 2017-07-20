from os import listdir
from os.path import isfile, join, sep
import sys
import time
import urllib

from influxdb import InfluxDBClient

global client


def get_device_id(image_path):
    return file(image_path).name.split(sep)[-1][:-4]


def load_images(image_folder):
    """Loads the images to database
    :param image_folder: the string representing the folder holding the image files
    """
    images = [join(image_folder, f) for f in listdir(image_folder) if isfile(join(image_folder, f))]
    for i in images:
        insert_image(get_device_id(i), i)


def make_point(device_id, image_path, tags=None, ts=None):
    return {
        "measurement": device_id,
        "points": [{
            "time": time.strftime('%Y-%m-%dT%H:%M:%SZ') if ts is None else ts,
            "fields": {
                "path": urllib.quote_plus(image_path)
            },
            "tags": {} if tags is None else tags
        }]
    }


def insert_image(device_id, image_path, tag=None, timestamp=None):
    """Insert the path of an image for a specific device to the database"""
    client.write(make_point(device_id, image_path), params={'precision': 'm', 'db': client._database}, protocol='json')


def is_db_exist(db_name):
    """Check whether the database with name db_name exists
    :param db_name: the database name to check
    """
    for item in client.get_list_database():
        if db_name == item.get('name'):
            return True
    return False


def get_series_by_tag(tags, field=None, start=None, end=None):
    if type(tags) is not dict:
        return None
    if field is None:
        sql = "select * from /sn/ where %s" % _get_tags_clause(tags)
    else:
        sql = "select %s from /sn/ where %s" % (field, _get_tags_clause(tags))

    time_c = _get_time_clause(start, end)
    if time_c == '':
        sql += ';'
    else:
        sql += ' and %s;' % time_c

    rs = client.query(sql, params={'pretty': 'true'}, database=client._database)
    for p in rs.get_points():
        print(urllib.unquote_plus(p.get('path')))


def _get_tags_clause(tags):
    where_clause = " and "
    keys = ["%s='%s'" % (k, tags.get(k)) for k in tags.keys()]
    where_clause = where_clause.join(keys)
    return where_clause


def get_series_by_time(start=None, end=None):
    sql = _get_time_clause(start, end)
    if sql == '':
        return None
    sql = "select * from /sn/ where %s;" % sql
    rs = client.query(sql, params={'pretty': 'true'}, database=client._database)
    for p in rs.get_points():
        print(urllib.unquote_plus(p.get('path')))


def _get_time_clause(start, end):
    if start is None:
        if end is None:
            sql = ''
        else:
            sql = "time < '%s'" % end
    else:
        if end is None:
            sql = "time > '%s'" % start
        else:
            sql = "time > '%s' and time < '%s'" % (start, end)
    return sql


def init():
    if not is_db_exist(db_name):
        client.create_database(db_name)
    folder_path = 'C:\\Users\\h230809\\PycharmProjects\\test\\images\\2017-07-18'
    load_images(image_folder=folder_path)


def update():
    result = client.query("select * from /sn/ where time>'2017-07-18T18:59:00Z';", params={'pretty': 'true'},
                          database=client._database)
    for p in result.get_points():
        if p.get('status') == '' or p.get('status') is None:
            print(urllib.unquote_plus(p.get('path')))
            # client.write(make_point('sn004', urllib.unquote_plus(p.get('path')), {'status': 'nice'}, p.get('time')), params={'precision': 'm', 'db': client._database},
            #              protocol='json')


def query():
    get_series_by_tag({'status': ''}, 'path', '2017-07-18T11:00:00Z', '2017-07-18T16:00:00Z')
    # get_series_by_time('2017-07-18T11:00:00Z', '2017-07-18T16:00:00Z')


def clean():
    if is_db_exist(db_name):
        client.drop_database(db_name)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python time_series.py [init|update|clean] db_host db_name')
        exit(0)

    cmd = sys.argv[1]
    host = sys.argv[2]
    db_name = sys.argv[3]
    client = InfluxDBClient(host, database=db_name)

    locals()[cmd]()
