from configparser import ConfigParser
from os import sep, remove
from os.path import isdir, isfile
import sys


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
            ret = self._vals.get(item)
            if type(ret) is tuple:
                sv = []
                for v in ret:
                    sv.append(str(v))
                ret = ','.join(sv)
            return ret
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
    if not isdir(info.DATA_ROOT):
        print('Invalid DATA_ROOT')
        ret = False
    if not isdir(info.TILES):
        print('Invalid TILES')
        ret = False
    try:
        float(info.GSD_IR)
    except ValueError:
        print('Invalid GSD_IR')
        ret = False
    return ret


def build_compose_file(template, info):
    path = 'conf/docker-compose.yml'
    if isfile(path):
        remove(path)

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
