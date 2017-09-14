from configparser import ConfigParser
from os import sep, remove
from os.path import isdir, isfile
import sys

from get_map_coordinates import get_tif_coordinates

class TenantInfo(object):
    def __init__(self, section: str):
        config = ConfigParser()
        config.read('conf/config.ini')
        self._vals = {}
        if config.has_section(section):
            for item in config.items(section):
                self._vals.update({item[0].upper(): item[1]})
        else:
            print('No configuration for Tenant %s' % section)

    def __getattr__(self, item):
        if item in self._vals.keys():
            return self._vals.get(item)
        else:
            raise AttributeError

    def setattr(self, key, value):
        self._vals.update({key: value})

    def get_attrs(self):
        return self._vals.keys()


def check_info(info: TenantInfo):
    ret = True
    if not isdir(sep.join((info.DB_ROOT, 'data'))) or not isdir(sep.join((info.DB_ROOT, 'conf'))):
        print('Invalid DB_ROOT.')
        ret = False
    elif not isdir(info.DATA_ROOT):
        print('Invalid DATA_ROOT')
        ret = False
    elif not isfile(sep.join((info.DATA_ROOT, info.PANORAMA))):
        print('Invalid PANORAMA')
        ret = False
    elif not isfile(sep.join((info.DATA_ROOT, info.UI_MAP))):
        print('Invalid UI_MAP')
        ret = False
    else:
        try:
            float(info.GSD_PANORAMA)
        except ValueError:
            print('Invalid GSD_PANORAMA')
            ret = False
        try:
            float(info.GSD_IR)
        except ValueError:
            print('Invalid GSD_IR')
            ret = False
    return ret


def update_gps_info(info: TenantInfo):
    center, top, bottom = get_tif_coordinates(sep.join((info.DATA_ROOT, info.PANORAMA)))
    info.setattr('GPS_CENTER', center)
    info.setattr('GPS_TOP', top)
    info.setattr('GPS_BOTTOM', bottom)


def build_compose_file(template, info):
    path = 'conf/docker-compose.yml'
    if isfile(path):
        remove(path)

    update_gps_info(info)

    with open('conf/docker-compose.yml', 'w') as ofile:
        for line in template:
            for attr in info.get_attrs():
                formatted = '{$%s}' % attr
                line = line.replace(formatted, info.__getattr__(attr))
            ofile.write(line)


if __name__ == "__main__":
    tinfo = TenantInfo(sys.argv[1])

    if check_info(tinfo):
        with open('docker-compose.template', 'r') as f:
            build_compose_file(f, tinfo)
