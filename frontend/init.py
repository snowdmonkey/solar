from io import open
import os
from os import remove
from os.path import isfile


def get_conf():
    ret = {}
    for k in os.environ.keys():
        if k.count('GPS') > 0:
            ret.update({k: os.environ.get(k)})
    if 'HOST' in os.environ.keys():
        ret.update({'HOST': os.environ.get('HOST')})
    return ret


if __name__ == '__main__':
    conf = get_conf()

    path = 'html/index.html'
    if isfile(path):
        remove(path)

    with open('html/index.html.template', 'r', encoding='utf-8') as tfile:
        with open(path, 'w', encoding='utf-8') as ofile:
            for line in tfile:
                if line.count('{$') > 0:
                    for key in conf.keys():
                        formatted = '{$%s}' % key
                        line = line.replace(formatted, conf.get(key))
                ofile.write(line)
