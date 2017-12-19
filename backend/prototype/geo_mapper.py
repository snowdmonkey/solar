# from sklearn import linear_model
from abc import ABC, abstractmethod
from pyproj import Proj
from typing import Tuple
import subprocess
import json
import re
import gdal
import utm


class GeoMapper(ABC):

    @abstractmethod
    def gps2pixel(self, latitude: float, longitude: float) -> Tuple[int, int]:
        """
        transfer a pair of gps coordinates to a pixel position on image
        :return: (row index, col index), counting from 0
        """
        pass

    @abstractmethod
    def pixel2gps(self, row: int, col: int) -> Tuple[float, float]:
        """
        transfer a pixel position on an image to a pair of gps coordinates
        :param row: row index of the pixel, counts from 0
        :param col: col index of the pixel, counts from 0
        :return: gps coordinates
        """
        pass

    @abstractmethod
    def pixel2utm(self, row: int, col: int) -> Tuple[float, float, int]:
        """
        map a pixel to a utm position
        :param row: row index of the pixel
        :param col: col index of the pixel
        :return: (easting, northing, zone_number)
        """
        pass

    @abstractmethod
    def utm2pixel(self, easting: float, northing: float, zone_number: int) -> Tuple[int, int]:
        """
        map a utm position to image pixel
        :param easting: utm easting, in meters
        :param northing: utm northing, in meters
        :param zone_number: utm zone number
        :return: (row_index, col_index)
        """
        pass

    @property
    @abstractmethod
    def gsd(self) -> float:
        """
        :return: ground sampling distance of the mapper, in meters
        """
        pass

# class AnchorGeoMapper(GeoMapper):
#
#     def __init__(self, pixel_anchors, gps_anchors):
#         """
#         :param pixel_anchors: list of [row_index, col_index], row and col index count from 0
#         :param gps_anchors: list of [latitude, longitude]
#         """
#
#         self.pixel_anchors = pixel_anchors
#         self.gps_anchors = gps_anchors
#
#         self._gps2pixel_models = None
#         self._pixel2gps_models = None
#
#     def _get_gps2pixel_modes(self):
#         latitude_model = linear_model.LinearRegression()
#         latitudes = [[x[0]] for x in self.gps_anchors]
#         pixel_row_indices = [x[0] for x in self.pixel_anchors]
#         latitude_model.fit(latitudes, pixel_row_indices)
#
#         longitude_model = linear_model.LinearRegression()
#         longitudes = [[x[1]] for x in self.gps_anchors]
#         pixel_col_indices = [x[1] for x in self.pixel_anchors]
#         longitude_model.fit(longitudes, pixel_col_indices)
#
#         self._gps2pixel_models = (latitude_model, longitude_model)
#
#     def _get_pixel2gps_models(self):
#         row_model = linear_model.LinearRegression()
#         latitudes = [x[0] for x in self.gps_anchors]
#         row_indices = [[x[0]] for x in self.pixel_anchors]
#         row_model.fit(row_indices, latitudes)
#
#         col_model = linear_model.LinearRegression()
#         longitudes = [x[1] for x in self.gps_anchors]
#         col_indices = [[x[1]] for x in self.pixel_anchors]
#         col_model.fit(col_indices, longitudes)
#
#         self._pixel2gps_models = (row_model, col_model)
#
#     def gps2pixel(self, latitude: float, longitude: float):
#         if self._gps2pixel_models is None:
#             self._get_gps2pixel_modes()
#         row_index = self._gps2pixel_models[0].predict([[latitude]])[0]
#         col_index = self._gps2pixel_models[1].predict([[longitude]])[0]
#         return int(round(row_index)), int(round(col_index))
#
#     def pixel2gps(self, row_index: int, col_index: int):
#         if self._pixel2gps_models is None:
#             self._get_pixel2gps_models()
#         latitude = self._pixel2gps_models[0].predict([[row_index]])[0]
#         longitude = self._pixel2gps_models[1].predict([[col_index]])[0]
#         return latitude, longitude


class TifGeoMapper(GeoMapper):

    def __init__(self, tif_path: str):
        command = ['exiftool', '-j', '-c', '%+.10f', tif_path]
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        out = proc.stdout
        r_json = json.loads(out.decode("utf-8"))[0]
        project_str = r_json.get("ProjectedCSType")
        origin_str = r_json.get("ModelTiePoint")
        if project_str is not None:
            zone = int(re.search("zone [0-9]+", project_str).group(0).split(" ")[1])
        else:
            dataset = gdal.Open(tif_path)
            zone = int(re.search("UTM zone [0-9]+", dataset.GetProjection()).group(0).split(" ")[-1])
        self._utm_zone = zone
        self._x_origin = float(origin_str.split(" ")[3])
        self._y_origin = float(origin_str.split(" ")[4])
        self._x_pixel_scale = float(r_json.get("PixelScale").split(" ")[0])
        self._y_pixel_scale = float(r_json.get("PixelScale").split(" ")[1])
        self._projector = Proj(proj="utm", zone=zone, ellps="WGS84")
        self.width = r_json.get("ImageWidth")
        self.height = r_json.get("ImageHeight")

    def pixel2utm(self, row: int, col: int) -> Tuple[float, float, int]:
        """
        take in pixel position and return utm coordinates
        :param row: row index, start from 0
        :param col: col index, start from 0
        :return: utm x, y, zone_number
        """
        x = self._x_origin + self._x_pixel_scale * col
        y = self._y_origin - self._y_pixel_scale * row

        return x, y, self._utm_zone

    def utm2pixel(self, x: float, y: float, zone_number: int) -> Tuple[int, int]:
        """
        take in the utm coordinates in the same zone and return the pixel position
        :param x: utm x
        :param y: utm y
        :param zone_number: utm zone number
        :return: row, col in the image, start from 0
        """
        if zone_number != self._utm_zone:
            raise ValueError("the point is not in the same utm zone with the image")

        col = round((x - self._x_origin) / self._x_pixel_scale)
        row = round((self._y_origin - y) / self._y_pixel_scale)

        return row, col

    def pixel2gps(self, row: int, col: int) -> Tuple[float, float]:

        longitude, latitude = self._projector(*(self.pixel2utm(row, col)[:2]), inverse=True)
        return latitude, longitude

    def gps2pixel(self, latitude: float, longitude: float) -> Tuple[int, int]:

        x, y = self._projector(longitude, latitude)
        row, col = self.utm2pixel(x, y, self._utm_zone)
        return row, col

    @property
    def gsd(self) -> float:
        return self._x_pixel_scale


class UTMGeoMapper(GeoMapper):

    def __init__(self, gsd: float,
                 origin_utm: Tuple[float, float] = None,
                 utm_zone: int = None,
                 origin_gps: Tuple[float, float] = None,
                 origin_pixel: Tuple[float, float] = (0.0, 0.0)):
        """
        this geomapper is based on a UTM transformation
        :param gsd: ground sample distance, in meters
        :param origin_utm: the utm coordinates of the origin pixel, in meters
        :param origin_gps: the gps coordinates of the origin pixel, if it is provided, then origin_utm and utm_zone can
        be none
        :param origin_pixel: pixel corresponding to origin, default to be the top left corner, (row, col) format
        """
        self._gsd = gsd
        self._origin = origin_utm
        self._utm_zone = utm_zone
        self._origin_pixel = origin_pixel
        if origin_gps is not None:
            easting, northing, utm_zone, _ = utm.from_latlon(*origin_gps)
            self._origin = (easting, northing)
            self._utm_zone = utm_zone
        self._projector = Proj(proj="utm", zone=utm_zone, ellps="WGS84")

    def pixel2utm(self, row: int, col: int) -> Tuple[float, float, int]:
        """
        take in a pixel position and return utm coordinates
        :param row: pixel row position
        :param col: pixel col position
        :return: corresponding utm coordinate in (easting, northing, zone) format
        """
        x = self._origin[0] + (col - self._origin_pixel[1]) * self._gsd
        y = self._origin[1] - (row - self._origin_pixel[0]) * self._gsd
        return x, y, self._utm_zone

    def utm2pixel(self, x: float, y: float, zone_number: int) -> Tuple[int, int]:
        """
        take in a pair of utm coordinates and return pixel position
        :param x: easting of the utm coordinates
        :param y: northing of the utm coordinates
        :return: pixel position in form of (row, col)
        """
        row = self._origin_pixel[0] - (y - self._origin[1]) / self._gsd
        col = self._origin_pixel[1] + (x - self._origin[0]) / self._gsd
        row = int(row)
        col = int(col)
        return row, col

    def pixel2gps(self, row: int, col: int) -> Tuple[float, float]:
        """
        take in pixel location (counting from 0) and output gps coordinates
        :param row: row index
        :param col: col index
        :return: gps coordinates
        """
        x, y, _ = self.pixel2utm(row, col)
        lng, lat = self._projector(x, y, inverse=True)
        return lat, lng

    def gps2pixel(self, latitude: float, longitude: float) -> Tuple[int, int]:
        """
        take in gps coordinates and output pixel location (counting from 0)
        :param latitude:
        :param longitude:
        :return: pixel location
        """
        x, y = self._projector(longitude, latitude)
        row, col = self.utm2pixel(x, y, self._utm_zone)
        return row, col

    @property
    def gsd(self) -> float:
        return self._gsd


def main():
    pass


if __name__ == "__main__":
    main()

