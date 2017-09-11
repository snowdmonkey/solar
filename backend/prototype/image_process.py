from geomapping.geo_mapper import GeoMapper, TifGeoMapper, AnchorGeoMapper
from os.path import join
from pymongo import MongoClient
from image_processing_functions import batch_process_exif, batch_process_rotation, batch_process_label, \
    batch_process_locate
import logging
import sys
import os


class ImageProcessPipeline:
    """
    this class provides the whole pipeline of process a folder of images. The folder should typically be IMG/{date}
    """

    def __init__(self, image_folder: str, date: str):
        """
        initial the process pipeline
        :param image_folder: the folder where the images rest, there should be sub-folder ir and visual, image_folder
        :param date: str of format YYYY-mm-dd e.g. 2017-06-21
        """
        self._image_folder = image_folder
        self._geo_mapper = self._get_geo_mapper()
        self._mongo_client = self._get_mongo_client()
        self._date = date
        self.logger = logging.getLogger("ImageProcessPipeline")

    @staticmethod
    def _get_geo_mapper() -> GeoMapper:
        panorama_folder = os.getenv("BG_PATH")
        file_names = os.listdir(panorama_folder)
        if "panorama.tif" in file_names:
            panorama_path = join(panorama_folder, "panorama.tif")
            geo_mapper = TifGeoMapper(panorama_path)
            return geo_mapper
        else:  # testing purpose only
            pixel_anchors = [[639, 639],  [639, 1328],  [639, 2016],
                             [1358, 639], [1358, 1328], [1358, 2016],
                             [2076, 639], [2076, 1328], [2076, 2016]]
            gps_anchors = [[33.59034075, 119.63160525], [33.59034075, 119.6334535], [33.59034075, 119.63530175],
                           [33.58873250, 119.63160525], [33.58873250, 119.6334535], [33.58873250, 119.63530175],
                           [33.58712425, 119.63160525], [33.58712425, 119.6334535], [33.58712425, 119.63530175]]
            geo_mapper = AnchorGeoMapper(pixel_anchors=pixel_anchors, gps_anchors=gps_anchors)
            return geo_mapper

    @staticmethod
    def _get_mongo_client() -> MongoClient:
        mongo_host = os.getenv("MONGO_HOST")
        if mongo_host is None:
            raise Exception("MONGO_HOST not found")
        else:
            return MongoClient(host=mongo_host, port=27017)

    def _process_exif(self):

        self.logger.info("starts to process exif")

        exif_dict = batch_process_exif(folder_path=join(self._image_folder, "ir"))
        results = dict()
        results["date"] = self._date
        results["value"] = exif_dict
        self._mongo_client.solar.exif.insert_one(results)

        self.logger.info("processing exif ends")

    def _process_rotate(self):
        self.logger.info("starts to process rotate")
        batch_process_rotation(folder_path=join(self._image_folder, "ir"))
        self.logger.info("processing rotation ends")

    def _process_label(self):
        self.logger.info("starts to process labeling")
        rect_dict = batch_process_label(folder_path=join(self._image_folder, "ir"))
        results = dict()
        results["date"] = self._date
        results["value"] = rect_dict
        self._mongo_client.solar.rect.insert_one(results)
        self.logger.info("defects labeling ends")

    def _process_locate(self):
        self.logger.info("starts to process defects locating")
        pixel_ratio = self._get_pixel_ratio()
        defect_dict = batch_process_locate(folder_path=join(self._image_folder, "ir"),
                                           geo_mapper=self._geo_mapper,
                                           pixel_ratio=pixel_ratio)
        results = dict()
        results["date"] = self._date
        results["value"] = defect_dict
        self._mongo_client.solar.defect.insert_one(results)

        self.logger.info("defects locating ends")

    @staticmethod
    def _get_pixel_ratio() -> float:
        """
        pixel ratio means one pixel on the panorama image equals to how many pixels on the ir image physically.
        Usually it equals to GSD_PANORAMA / GSD_IR
        :return: pixel_ratio as a float
        """
        gsd_panorama = float(os.getenv("GSD_PANORAMA"))
        gsd_ir = float(os.getenv("GSD_IR"))
        return gsd_panorama / gsd_ir

    def run(self):
        self._process_exif()
        self._process_rotate()
        self._process_label()
        self._process_locate()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    folder_path = sys.argv[1]
    date = sys.argv[2]
    pipeline = ImageProcessPipeline(folder_path, date)
    pipeline.run()


if __name__ == "__main__":
    main()
