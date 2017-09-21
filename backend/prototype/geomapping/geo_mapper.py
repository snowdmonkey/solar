from sklearn import linear_model
from abc import ABC, abstractmethod
from pyproj import Proj
from typing import Tuple
import subprocess
import json
import re
import gdal


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


class AnchorGeoMapper(GeoMapper):

    def __init__(self, pixel_anchors, gps_anchors):
        """
        :param pixel_anchors: list of [row_index, col_index], row and col index count from 0 
        :param gps_anchors: list of [latitude, longitude]
        """

        self.pixel_anchors = pixel_anchors
        self.gps_anchors = gps_anchors

        self._gps2pixel_models = None
        self._pixel2gps_models = None

    def _get_gps2pixel_modes(self):
        latitude_model = linear_model.LinearRegression()
        latitudes = [[x[0]] for x in self.gps_anchors]
        pixel_row_indices = [x[0] for x in self.pixel_anchors]
        latitude_model.fit(latitudes, pixel_row_indices)

        longitude_model = linear_model.LinearRegression()
        longitudes = [[x[1]] for x in self.gps_anchors]
        pixel_col_indices = [x[1] for x in self.pixel_anchors]
        longitude_model.fit(longitudes, pixel_col_indices)

        self._gps2pixel_models = (latitude_model, longitude_model)

    def _get_pixel2gps_models(self):
        row_model = linear_model.LinearRegression()
        latitudes = [x[0] for x in self.gps_anchors]
        row_indices = [[x[0]] for x in self.pixel_anchors]
        row_model.fit(row_indices, latitudes)

        col_model = linear_model.LinearRegression()
        longitudes = [x[1] for x in self.gps_anchors]
        col_indices = [[x[1]] for x in self.pixel_anchors]
        col_model.fit(col_indices, longitudes)

        self._pixel2gps_models = (row_model, col_model)

    def gps2pixel(self, latitude: float, longitude: float):
        if self._gps2pixel_models is None:
            self._get_gps2pixel_modes()
        row_index = self._gps2pixel_models[0].predict([[latitude]])[0]
        col_index = self._gps2pixel_models[1].predict([[longitude]])[0]
        return int(round(row_index)), int(round(col_index))

    def pixel2gps(self, row_index: int, col_index: int):
        if self._pixel2gps_models is None:
            self._get_pixel2gps_models()
        latitude = self._pixel2gps_models[0].predict([[row_index]])[0]
        longitude = self._pixel2gps_models[1].predict([[col_index]])[0]
        return latitude, longitude


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
        self._x_origin = float(origin_str.split(" ")[3])
        self._y_origin = float(origin_str.split(" ")[4])
        self._x_pixel_scale = float(r_json.get("PixelScale").split(" ")[0])
        self._y_pixel_scale = float(r_json.get("PixelScale").split(" ")[1])
        self._projector = Proj(proj="utm", zone=zone, ellps="WGS84")
        self.width = r_json.get("ImageWidth")
        self.height = r_json.get("ImageHeight")

    def _pixel2utm(self, row: int, col: int) -> Tuple[float, float]:
        """
        take in pixel position and return utm coordinates
        :param row: row index, start from 0
        :param col: col index, start from 0
        :return: utm x, y
        """
        x = self._x_origin + self._x_pixel_scale * col
        y = self._y_origin - self._y_pixel_scale * row

        return x, y

    def _utm2pixel(self, x: float, y: float) -> Tuple[int, int]:
        """
        take in the utm coordinates in the same zone and return the pixel position
        :param x: utm x
        :param y: utm y
        :return: row, col in the image, start from 0
        """
        col = round((x - self._x_origin) / self._x_pixel_scale)
        row = round((self._y_origin - y) / self._y_pixel_scale)

        return row, col

    def pixel2gps(self, row: int, col: int) -> Tuple[float, float]:

        longitude, latitude = self._projector(*self._pixel2utm(row, col), inverse=True)
        return latitude, longitude

    def gps2pixel(self, latitude: float, longitude: float) -> Tuple[int, int]:

        x, y = self._projector(longitude, latitude)
        row, col = self._utm2pixel(x, y)
        return row, col


def main():
    # pixel_anchors = [[639, 639],  [639, 1328],  [639, 2016],
    #                  [1358, 639], [1358, 1328], [1358, 2016],
    #                  [2076, 639], [2076, 1328], [2076, 2016]]
    # gps_anchors = [[33.59034075, 119.63160525], [33.59034075, 119.6334535], [33.59034075, 119.63530175],
    #                [33.58873250, 119.63160525], [33.58873250, 119.6334535], [33.58873250, 119.63530175],
    #                [33.58712425, 119.63160525], [33.58712425, 119.6334535], [33.58712425, 119.63530175]]
    #
    # geo_mapper = AnchorGeoMapper(pixel_anchors=pixel_anchors, gps_anchors=gps_anchors)
    # print(geo_mapper.pixel2gps(0, 0))
    # print(geo_mapper.pixel2gps(2715, 2655))

    tif_path = r"C:\Users\h232559\Desktop\odm_orthophoto.tif"
    geo_mapper = TifGeoMapper(tif_path)
    # print(geo_mapper.gps2pixel(33.803890, 119.731152))
    print(geo_mapper.pixel2gps(0, 0))
    # print(geo_mapper.pixel2gps(11465, 8642))


if __name__ == "__main__":
    main()

