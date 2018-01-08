from typing import NamedTuple

UTM = NamedTuple("UTM", [("easting", float), ("northing", float), ("zone", int)])

GPS = NamedTuple("GPS", [("latitude", float), ("longitude", float)])
