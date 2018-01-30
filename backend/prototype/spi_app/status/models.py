from typing import NamedTuple

StationStatus = NamedTuple("StationStatus", [("date", str),
                                             ("healthy", int),
                                             ("toconfirm", int),
                                             ("infix", int),
                                             ("confirmed", int),
                                             ("overallStatus", str)])