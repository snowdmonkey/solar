from sklearn import linear_model


class GeoMapper:

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

    def gps2pixel(self, latitude, longitude):
        if self._gps2pixel_models is None:
            self._get_gps2pixel_modes()
        row_index = self._gps2pixel_models[0].predict([[latitude]])[0]
        col_index = self._gps2pixel_models[1].predict([[longitude]])[0]
        return int(round(row_index)), int(round(col_index))

    def pixel2gps(self, row_index, col_index):
        if self._pixel2gps_models is None:
            self._get_pixel2gps_models()
        latitude = self._pixel2gps_models[0].predict([[row_index]])[0]
        longitude = self._pixel2gps_models[1].predict([[col_index]])[0]
        return latitude, longitude


def main():
    pixel_anchors = [[639, 639],  [639, 1328],  [639, 2016],
                     [1358, 639], [1358, 1328], [1358, 2016],
                     [2076, 639], [2076, 1328], [2076, 2016]]
    gps_anchors = [[33.59034075, 119.63160525], [33.59034075, 119.6334535], [33.59034075, 119.63530175],
                   [33.58873250, 119.63160525], [33.58873250, 119.6334535], [33.58873250, 119.63530175],
                   [33.58712425, 119.63160525], [33.58712425, 119.6334535], [33.58712425, 119.63530175]]

    geo_mapper = GeoMapper(pixel_anchors=pixel_anchors, gps_anchors=gps_anchors)
    # print(geo_mapper.gps2pixel(33.59034075, 119.6334535))
    print(geo_mapper.pixel2gps(0, 0))
    print(geo_mapper.pixel2gps(2715, 2655))


if __name__ == "__main__":
    main()

