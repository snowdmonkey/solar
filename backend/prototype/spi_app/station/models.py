from typing import NamedTuple, Tuple

Station = NamedTuple("Station", [("stationId", str),
                                 ("stationName", str),
                                 ("description", str),
                                 ("gps", Tuple[float, float])])