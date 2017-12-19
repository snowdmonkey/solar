import logging
import os
# import sys
from os.path import join
from typing import Optional
from pymongo import MongoClient
# from geo_mapper import GeoMapper, TifGeoMapper
from image_processing_functions import batch_process_exif, batch_process_rotation, batch_process_label, \
    batch_process_locate, batch_process_profile, batch_process_aggregate


class ImageProcessPipeline:
    """
    this class provides the whole pipeline of create_profile a folder of images.
    The folder should typically be IMG_ROOT/{station}/{date} with subdirectories ir and visual
    """

    def __init__(self, image_folder: str, station: str, date: str):
        """
        initial the create_profile pipeline
        :param image_folder: the folder where the images rest, there should be sub-folder ir and visual
        :param station
        :param date: str of format YYYY-mm-dd e.g. 2017-06-21
        """
        self.logger = logging.getLogger("ImageProcessPipeline")
        self._image_folder = image_folder
        # self._image_folder = join(image_folder, station, date)
        self._mongo_client = self._get_mongo_client()
        self._date = date
        self._station = station
        self._gsd_ir = float(os.getenv("GSD_IR"))

    def _get_mongo_client(self) -> Optional[MongoClient]:
        mongo_host = os.getenv("MONGO_HOST")
        if mongo_host is None:
            # raise Exception("MONGO_HOST not found")
            self.logger.warning("MONGO_HOST not found")
            return None
        else:
            return MongoClient(host=mongo_host, port=27017)

    def _process_exif(self):

        self.logger.info("starts to process exif in {}".format(join(self._image_folder, "ir")))

        exif_list = batch_process_exif(folder_path=join(self._image_folder, "ir"))
        if self._mongo_client is not None:
            collection = self._get_mongo_client().get_database("solar").get_collection("exif")
            for exif in exif_list:
                exif.pop("ThumbnailImage", None)
                exif.pop("RawThermalImage", None)
                exif.update({"station": self._station, "date": self._date})
            collection.delete_many({"station": self._station, "date": self._date})
            collection.insert_many(exif_list)

        self.logger.info("processing exif ends")

    def _process_rotate(self):
        self.logger.info("starts to process rotate")
        batch_process_rotation(folder_path=join(self._image_folder, "ir"))
        self.logger.info("processing rotation ends")

    def _process_label(self):
        self.logger.info("starts to process labeling")

        results = batch_process_label(folder_path=join(self._image_folder, "ir"))
        for d in results:
            d.update({"date": self._date, "station": self._station})
        if self._mongo_client is not None:
            collection = self._mongo_client.get_database("solar").get_collection("rect")
            collection.delete_many({"station": self._station, "date": self._date})
            collection.insert_many(results)
        self.logger.info("defects labeling ends")

    def _process_profile(self):
        self.logger.info("start to process profiling")

        results = batch_process_profile(folder_path=join(self._image_folder, "ir"), gsd=self._gsd_ir)
        for d in results:
            d.update({"date": self._date, "station": self._station})
        if self._mongo_client is not None:
            collection = self._mongo_client.get_database("solar").get_collection("rect")
            collection.delete_many({"station": self._station, "date": self._date})
            if len(results) != 0:
                collection.insert_many(results)
        self.logger.info("defects profiling ends")

    def _process_aggregate(self):
        self.logger.info("starts to process defects aggregation")

        defects = batch_process_aggregate(folder_path=join(self._image_folder, "ir"), group_criteria=2.0)

        for d in defects:
            d.update({"station": self._station, "date": self._date, "gsd": self._gsd_ir})

        if self._mongo_client is not None:
            collection = self._get_mongo_client().get_database("solar").get_collection("defect")
            collection.delete_many({"station": self._station, "date": self._date})
            if len(defects) != 0:
                collection.insert_many(defects)

        self.logger.info("defects aggregating ends")

    def _process_locate(self):
        self.logger.info("starts to process defects locating")

        defects = batch_process_locate(folder_path=join(self._image_folder, "ir"),
                                       gsd=self._gsd_ir,
                                       group_criteria=2.0)

        for d in defects:
            d.update({"station": self._station, "date": self._date, "gsd": self._gsd_ir})

        if self._mongo_client is not None:
            collection = self._get_mongo_client().get_database("solar").get_collection("defect")
            collection.delete_many({"station": self._station, "date": self._date})
            collection.insert_many(defects)

        self.logger.info("defects locating ends")

    def run(self):
        self._process_exif()
        self._process_rotate()
        # self._process_label()
        # self._process_locate()
        self._process_profile()
        self._process_aggregate()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--station", help="set station name", required=True)
    parser.add_argument("--date", help="set date", required=True)
    parser.add_argument("--gsd", help="set ir gsd in meters", type=float, required=True)
    parser.add_argument("--mongo-host", dest="mongo_host", help="set mongo host")
    parser.add_argument("folder_path", type=str, help="the folder contain ir and visual sub directories")

    args = parser.parse_args()

    os.environ["GSD_IR"] = str(args.gsd)
    if args.mongo_host is not None:
        os.environ["MONGO_HOST"] = args.mongo_host

    pipeline = ImageProcessPipeline(image_folder=args.folder_path, station=args.station, date=args.date)
    pipeline.run()


if __name__ == "__main__":
    main()
