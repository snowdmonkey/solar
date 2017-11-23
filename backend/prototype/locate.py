"""this module is about how to get the exact geographical location of the defects, panels, panel groups, etc.
The position is obtained by leveraging provided gps information and image matching
"""

from typing import List, Tuple, Optional

import utm
from shapely.geometry import Point, Polygon
from semantic import IRProfile
from geo_mapper import GeoMapper


class PanelGroup:
    """
    this class is mainly used to store a panel group's geographical information
    """

    def __init__(self, group_id: str,
                 vertices_gps: List[Tuple[float, float]] = None,
                 vertices_utm: List[Tuple[float, float, int]] = None):
        """
        can be constructed with gps position or utm position, but better not both
        :param group_id: id of the panel group
        :param vertices_gps: list of (latitude, longitude)
        :param vertices_utm: list of (easting, northing, utm_zone_number)
        """
        if vertices_gps is None:
            if vertices_utm is None:
                raise ValueError("gps or utm coordinates need to be provided")
            else:
                self._vertices_utm = vertices_utm
                self._vertices_gps = [utm.to_latlon(*x, northern=True) for x in vertices_utm]

        if vertices_gps is not None:
            if vertices_utm is None:
                self._vertices_gps = vertices_gps
                self._vertices_utm = [utm.from_latlon(x[0], x[1])[:3] for x in vertices_gps]
            else:
                raise ValueError("provide gps or utm only")

        # a polygon based on panel group's utm coordinates
        self._polygon = Polygon([(x[0], x[1]) for x in self._vertices_utm])

        self._utm_zone = self._vertices_utm[0][2]
        self._group_id = group_id
        self._bounds = self._polygon.bounds

    @property
    def panel_group_id(self) -> Optional[str]:
        return self._group_id

    @property
    def utm_zone(self) -> int:
        return self._utm_zone

    @property
    def least_east(self) -> float:
        """
        :return: least easting utm value, in meters
        """
        return self._bounds[0]

    @property
    def most_east(self) -> float:
        """
        :return: most easting utm value, in meters
        """
        return self._bounds[2]

    @property
    def most_north(self) -> float:
        """
        :return: most northing utm value, in meters
        """
        return self._bounds[3]

    @property
    def least_north(self) -> float:
        """
        :return: least northing utm value, in meters
        """
        return self._bounds[1]

    def distance_to_utm(self, utm_pos: Tuple[float, float, int]) -> float:
        """
        calculate the distance between the panel group to a position based on utm coordinates
        :param utm_pos: utm coordinates in (easting, northing, zone_number)
        :return: distance in meters
        """
        if utm_pos[2] != self._utm_zone:
            raise ValueError("the point is not in the same utm zone with the panel group")

        point = Point(utm_pos[:2])
        return self._polygon.distance(point)

    def distance_to_gps(self, gps: Tuple[float, float]) -> float:
        """
        calculate the distance between the panel group and a pair of given gps coordinates
        :param gps: gps coordinates in (latitude, longitude)
        :return: distance in meters
        """
        utm_pos = utm.from_latlon(*gps)[:3]
        return self.distance_to_utm(utm_pos)


class Station:
    """
    this class is mainly used to store the panel and panel group geographical position in a farm
    """

    def __init__(self):

        self._panel_groups = list()
        self._utm_zone = None

    @property
    def panel_groups(self) -> List[PanelGroup]:
        return self._panel_groups

    def add_panel_group(self, panel_group: PanelGroup):

        if self._utm_zone is None:
            self._utm_zone = panel_group.utm_zone
        elif self._utm_zone != panel_group.utm_zone:
            raise ValueError("utm zone mismatch")

        self._panel_groups.append(panel_group)

    def get_closest_group(self, utm_pos: Tuple[float, float, int] = None,
                          gps: Tuple[float, float] = None, distance_th: float = 50.0) -> Optional[PanelGroup]:
        """
        the method will take in a geographical position and return the most close panel group to it in the station. For
        the coordinates, either utm or gps should be provided
        :param utm_pos: utm coordinates of the given point, of format (easting, northing, zone_number)
        :param gps: gps coordinates of the given point, of format (latitude, longitude)
        :param distance_th: in meters, this is a threshold to shorten the panel group candidates list. Only if all of a
        panel group extreme points is within given point +/- this threshold, then it will be considered as a possible
        candidate
        :return: the closest panel group, or none if there is no panel group nearby
        """

        if utm_pos is None:
            if gps is None:
                raise ValueError("both utm and gps cannot be None")
            else:
                utm_pos = utm.from_latlon(*gps)[:3]

        if utm_pos[2] != self._utm_zone:
            return None

        # shorten the candidates list
        groups = [group for group in self.panel_groups if all([
            (utm_pos[0] - distance_th) < group.most_east < (utm_pos[0] + distance_th),
            (utm_pos[0] - distance_th) < group.least_east < utm_pos[0] + distance_th,
            (utm_pos[1] - distance_th) < group.most_north < (utm_pos[1] + distance_th),
            (utm_pos[1] - distance_th) < group.least_north < (utm_pos[1] + distance_th)])]

        if len(groups) == 0:
            return None

        groups.sort(key=lambda group: group.distance_to_utm(utm_pos))
        return groups[0]


class Positioner:
    """ this class aims to get the geographical positions for an IRProfile
    """

    def __init__(self):
        pass

    def locate(self, profile: IRProfile, geo_mapper: GeoMapper, farm: Station):
        """
        the method will map an IRProfile to geographical locations with a geo mapper and calibrate it with a Station
        :param profile: the IRProfile that will be positioned
        :param geo_mapper: geo mapper that can map pixels to geographical locations
        :param farm: Station information
        :return: None
        """
        for panel_group in profile.panel_groups:

            # map a panel group profile to a physical panel group
            centroid_xy = panel_group.centroid_xy
            centroid_gps = geo_mapper.pixel2gps(centroid_xy[1], centroid_xy[0])
            closest_panel = farm.get_closest_group(gps=centroid_gps)
            panel_group.set_panel_group_id(closest_panel.panel_group_id)

            # adjust a defect's utm with its relative position on a panel group
            for defect in panel_group.defects:
                defect_rc = defect.points_rc[0]
                to_top_ratio = (defect_rc[0] - panel_group.most_top) / (panel_group.most_bottom - panel_group.most_top)
                assert 0.0 <= to_top_ratio <= 1.0

                defect_utm = geo_mapper.pixel2utm(*defect_rc)
                corrected_northing = \
                    closest_panel.most_north - (closest_panel.most_north-closest_panel.least_north)*to_top_ratio
                corrected_utm = defect_utm[0], corrected_northing, defect_utm[2]

                # make sure the defect location is inside the closest panel
                if closest_panel.distance_to_utm(corrected_utm) != 0:
                    # TODO
                    pass

                defect.set_utm(corrected_utm)



