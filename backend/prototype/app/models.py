from typing import NamedTuple, Tuple


GPS = NamedTuple("GPS", [("lat", float), ("lng", float)])

Station = NamedTuple("Station", [("stationId", str),
                                 ("stationName", str),
                                 ("description", str),
                                 ("gps", Tuple[float, float])])

StationStatus = NamedTuple("StationStatus", [("date", str),
                                             ("healthy", int),
                                             ("toconfirm", int),
                                             ("infix", int),
                                             ("confirmed", int),
                                             ("overallStatus", str)])

