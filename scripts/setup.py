from configparser import ConfigParser
import sys

config = ConfigParser()
config.read('conf/config.ini')

section = sys.argv[1]
if config.has_section(section):
    print(config.items(section))
else:
    print('No configuration for %s' % section)
