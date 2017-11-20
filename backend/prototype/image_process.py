import logging
import os
import sys
from os.path import join

from pymongo import MongoClient

from geo_mapper import GeoMapper, TifGeoMapper, AnchorGeoMapper
from image_processing_functions import batch_process_exif, batch_process_rotation, batch_process_label, \
    batch_process_locate


class ImageProcessPipeline:
    """
    this class provides the whole pipeline of create_profile a folder of images. The folder should typically be IMG_ROOT/{date}
    """

    def __init__(self, image_folder: str, station: str, date: str):
        """
        initial the create_profile pipeline
        :param image_folder: the folder where the images rest, there should be sub-folder ir and visual, image_folder
        :param station
        :param date: str of format YYYY-mm-dd e.g. 2017-06-21
        """
        self._image_folder = image_folder
        # self._geo_mapper = self._get_geo_mapper()
        self._mongo_client = self._get_mongo_client()
        self._date = date
        self._station = station
        self._gsd_ir = float(os.getenv("GSD_IR"))
        self.logger = logging.getLogger("ImageProcessPipeline")

    # @staticmethod
    # def _get_geo_mapper() -> GeoMapper:
    #     panorama_path = os.getenv("BG_PATH")
    #
    #     if os.path.isfile(panorama_path):
    #         geo_mapper = TifGeoMapper(panorama_path)
    #         return geo_mapper
    #     else:  # testing purpose only
    #         pixel_anchors = [[639, 639],  [639, 1328],  [639, 2016],
    #                          [1358, 639], [1358, 1328], [1358, 2016],
    #                          [2076, 639], [2076, 1328], [2076, 2016]]
    #         gps_anchors = [[33.59034075, 119.63160525], [33.59034075, 119.6334535], [33.59034075, 119.63530175],
    #                        [33.58873250, 119.63160525], [33.58873250, 119.6334535], [33.58873250, 119.63530175],
    #                        [33.58712425, 119.63160525], [33.58712425, 119.6334535], [33.58712425, 119.63530175]]
    #         geo_mapper = AnchorGeoMapper(pixel_anchors=pixel_anchors, gps_anchors=gps_anchors)
    #         return geo_mapper

    @staticmethod
    def _get_mongo_client() -> MongoClient:
        mongo_host = os.getenv("MONGO_HOST")
        if mongo_host is None:
            raise Exception("MONGO_HOST not found")
        else:
            return MongoClient(host=mongo_host, port=27017)

    def _process_exif(self):

        self.logger.info("starts to create_profile exif")

        exif_list = batch_process_exif(folder_path=join(self._image_folder, "ir"))
        collection = self._get_mongo_client().get_database("solar").get_collection("exif")
        for exif in exif_list:
            exif.pop("ThumbnailImage", None)
            exif.pop("RawThermalImage", None)
            exif.update({"station": self._station, "date": self._date})
        collection.delete_many({"station": self._station, "date": self._date})
        collection.insert_many(exif_list)

        self.logger.info("processing exif ends")

    def _process_rotate(self):
        self.logger.info("starts to create_profile rotate")
        batch_process_rotation(folder_path=join(self._image_folder, "ir"))
        self.logger.info("processing rotation ends")

    def _process_label(self):
        self.logger.info("starts to create_profile labeling")
        # rect_dict = batch_process_label(folder_path=join(self._image_folder, "ir"))
        # results = dict()
        # results["date"] = self._date
        # results["value"] = rect_dict
        # self._mongo_client.solar.rect.update_one({"date": self._date}, {"$set": results}, upsert=True)
        results = batch_process_label(folder_path=join(self._image_folder, "ir"))
        for d in results:
            d.update({"date": self._date, "station": self._station})
        collection = self._mongo_client.get_database("solar").get_collection("rect")
        collection.delete_many({"station": self._station, "date": self._date})
        collection.insert_many(results)
        self.logger.info("defects labeling ends")

    def _process_locate(self):
        self.logger.info("starts to create_profile defects locating")
        # pixel_ratio = self._get_pixel_ratio()
        # group_criteria = 200 / float(os.getenv("GSD_PANORAMA"))
        defects = batch_process_locate(folder_path=join(self._image_folder, "ir"),
                                       gsd=self._gsd_ir,
                                       group_criteria=2.0)

        for d in defects:
            d.update({"station": self._station, "date": self._date, "gsd": self._gsd_ir})

        # results = dict()
        # results["date"] = self._date
        # results["value"] = defect_dict
        # self._mongo_client.solar.defect.update_one({"date": self._date}, {"$set": results}, upsert=True)
        collection = self._get_mongo_client().get_database("solar").get_collection("defect")
        collection.delete_many({"station": self._station, "date": self._date})
        collection.insert_many(defects)

        self.logger.info("defects locating ends")

    # @staticmethod
    # def _get_pixel_ratio() -> float:
    #     """
    #     pixel ratio means one pixel on the panorama image equals to how many pixels on the ir image physically.
    #     Usually it equals to GSD_PANORAMA / GSD_IR
    #     :return: pixel_ratio as a float
    #     """
    #     gsd_panorama = float(os.getenv("GSD_PANORAMA"))
    #     gsd_ir = float(os.getenv("GSD_IR"))
    #     return gsd_panorama / gsd_ir

    def run(self):
        self._process_exif()
        self._process_rotate()
        self._process_label()
        self._process_locate()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    folder_path = sys.argv[1]
    station = sys.argv[2]
    date = sys.argv[3]
    pipeline = ImageProcessPipeline(folder_path, station=station, date=date)
    pipeline.run()


if __name__ == "__main__":
    main()
