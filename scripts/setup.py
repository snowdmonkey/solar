from configparser import ConfigParser
from os import sep
from os.path import isdir, isfile
import sys


class TenantInfo(object):
    def __init__(self, section: str):
        config = ConfigParser()
        config.read('conf/config.ini')
        if config.has_section(section):
            self.host = config.get(section, 'HOST')
            self.dbroot = config.get(section, 'DB_ROOT')
            self.dataroot = config.get(section, 'DATA_ROOT')
            self.panorama = config.get(section, 'PANORAMA')
            self.gsdpanorama = config.get(section, 'GSD_PANORAMA')
            self.gsd = config.get(section, 'GSD')
        else:
            print('No configuration for Tenant %s' % section)


if __name__ == "__main__":
    tinfo = TenantInfo(sys.argv[1])
    if not isdir(sep.join((tinfo.dbroot, 'data'))) or not isdir(sep.join((tinfo.dbroot, 'conf'))):
        print('Invalid DB_ROOT.')
    elif not isdir(tinfo.dataroot):
        print('Invalid DATA_ROOT')
    elif not isfile(sep.join((tinfo.dataroot, tinfo.panorama))):
        print('Invalid PANORAMA')
    else:
        try:
            float(tinfo.gsdpanorama)
        except ValueError:
            print('Invalid GSD_PANORAMA')
            exit(-1)
        try:
            float(tinfo.gsd)
        except ValueError:
            print('Invalid GSD')
            exit(-1)
        with open('docker-compose.template', 'r') as f:
            for l in f:
                print(l)
