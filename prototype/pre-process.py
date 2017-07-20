from os import listdir
from os.path import isfile, join

import exifread


def convert_gps(gps_info):
    info = str(gps_info)[1:-1]
    a, b ,c = info.split(',')
    ret = float(a) + float(b)/60
    d, e = c.split('/')
    ret += float(d)/float(e)/60/60
    return ret

def process_exif(image, is_visual, output):
    imgf = open(image, 'rb')
    tags = exifread.process_file(imgf)
    rec = '"%s", %s, %s, %s, %s\n' \
          % (image, tags.get('EXIF DateTimeOriginal'), convert_gps(tags.get('GPS GPSLongitude')),
             convert_gps(tags.get('GPS GPSLatitude')), is_visual)
    output.write(rec)


def load_images(image_folder, is_visual, output):
    """Loads the images to database
    :param image_folder: the string representing the folder holding the image files
    """
    images = [join(image_folder, f) for f in listdir(image_folder) if isfile(join(image_folder, f)) and f.lower().endswith('.jpg')]
    for i in images:
        process_exif(i, is_visual, output)


if __name__ == '__main__':
    # of = open('C:\\SolarPanel\\2017-06-20\\exif.csv', 'w')
    # load_images('C:\\SolarPanel\\2017-06-20\\6-20-DJI', True, of)
    # load_images('C:\\SolarPanel\\2017-06-20\\6-20-FLIR', False, of)
    # of.close()
    # of = open('C:\\SolarPanel\\2017-06-21\\exif.csv', 'w')
    # load_images('C:\\SolarPanel\\2017-06-21\\6-21-FLIR', False, of)
    # of.close()
    of = open('C:\\SolarPanel\\2017-07-04\\exif.csv', 'w')
    load_images('C:\\SolarPanel\\2017-07-04\\7-04-1', True, of)
    load_images('C:\\SolarPanel\\2017-07-04\\7-04-2', True, of)
    of.close()