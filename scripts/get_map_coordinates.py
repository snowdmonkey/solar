import sys
from geo_mapper import TifGeoMapper


def get_tif_coordinates(tif_path: str):
    """
    pre-process the tif background map and return the center gps, top-left gps, and bottom-right gps
    :param tif_path: path to the background tif file
    :return: tuple ((center_lat, center_lon), (top_left_lat, top_left_lon), (bottom_right_lat, bottom_left_lon))
    """
    geo_mapper = TifGeoMapper(tif_path)
    top_pixel = (0, 0)
    center_pixel = (round(geo_mapper.height/2), round(geo_mapper.width/2))
    bottom_pixel = (geo_mapper.height - 1, geo_mapper.width - 1)

    top_gps = geo_mapper.pixel2gps(*top_pixel)
    center_gps = geo_mapper.pixel2gps(*center_pixel)
    bottom_gps = geo_mapper.pixel2gps(*bottom_pixel)

    return center_gps, top_gps, bottom_gps


if __name__ == "__main__":
    results = get_tif_coordinates(sys.argv[1])
    print("center corner GPS: {}".format(results[0]))
    print("top left corner GPS: {}".format(results[1]))
    print("bottom right corner GPS: {}".format(results[2]))
