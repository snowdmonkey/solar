from typing import NamedTuple

UTM = NamedTuple("UTM", [("easting", float), ("northing", float), ("zone", int)])

GPS = NamedTuple("GPS", [("latitude", float), ("longitude", float)])


def get_gsd(pitch_size: float, focal_length: float, relative_altitude: float) -> float:
    """
    calculate ground sampling distance
    :param pitch_size: distance between centers of two adjacent pitches on the sensor, in microns
    :param focal_length: absolute focal length in millimeters
    :param relative_altitude: relative altitude in meters
    :return: gsd in meters
    """
    gsd = relative_altitude * pitch_size * 1e-3 / focal_length
    return gsd
