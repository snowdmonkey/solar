"""
this module provide some utilities based on shapely
"""
from shapely.geometry.base import BaseGeometry
from shapely.geometry.point import Point
from typing import NamedTuple
from shapely.affinity import translate, scale, rotate, affine_transform, interpret_origin
from scipy.optimize import minimize
from misc import UTM
import numpy as np
import logging


logger = logging.getLogger(__name__)


_TargetTransformParams = NamedTuple("TransformParams", [("x_offset", float),
                                                        ("y_offset", float),
                                                        ("scale", float),
                                                        ("angle", float)])

TransformMatrix = NamedTuple("TransformMatrix",
                             [("a", float), ("b", float), ("d", float), ("e", float), ("xoff", float), ("yoff", float)])


def _target_affine_transform(shape: BaseGeometry, params: _TargetTransformParams) -> BaseGeometry:
    """
    transform a shape with an affine transformation described with TransformParams
    :param shape: the shape to transform
    :param params: parameters describe an affine transformation
    :return: a transformed shape
    """
    translated = translate(shape, xoff=params.x_offset, yoff=params.y_offset)
    scaled = scale(translated, xfact=params.scale, yfact=params.scale)
    rotated = rotate(scaled, angle=params.angle)
    return rotated


def affine_transform_utm(utm: UTM, matrix: TransformMatrix) -> UTM:
    """
    transform a given utm coordinates with a given affine transformation parameter
    :param utm: a utm coordinates
    :param matrix: a set of affine transformation parameters
    :return: transformed utm
    """
    p = Point(utm[0], utm[1])
    p = affine_transform(p, matrix)  # type: Point
    return UTM(p.x, p.y, utm[2])


class Aligner:

    def align(self, pattern: BaseGeometry, pos: BaseGeometry) -> TransformMatrix:
        """
        this method try to apply an affine transformation on pos, so the overlap between pattern and pos is maximized
        :param pattern: keep-it-still background
        :param pos: a shape we try to align it with pattern
        :return: transformation parameters
        """
        def target(x, pattern: BaseGeometry, pos: BaseGeometry):
            params = _TargetTransformParams(x[0], x[1], x[2], x[3])
            pos_trans = _target_affine_transform(pos, params)
            # neg_trans = affine_transform(neg, params)
            pos_overlap = pattern.intersection(pos_trans).area
            # neg_overlap = pattern.intersection(neg_trans).area
            false_pos_overlap = pos_trans.difference(pattern).area
            # return neg_overlap+false_pos_overlap-pos_overlap
            return false_pos_overlap - 2.0*pos_overlap

        bounds = ((-3, 3), (-3, 3), (0.7, 1.5), (-20, 20))

        # bounds = ((-50, 50), (-50, 50), (0.7, 1.5), (-20, 20))

        start_points = np.array([[0.0, 0.0, 1.0, 0.0],
                                 [2.0, 0.0, 1.0, 0.0],
                                 [-2.0, 0.0, 1.0, 0.0],
                                 [0.0, 2.0, 1.0, 0.0],
                                 [0.0, -2.0, 1.0, 0.0]])

        # start_points = np.array([[0.0, 0.0, 1.0, 0.0],
        #                          [2.0, 0.0, 1.0, 0.0],
        #                          [-2.0, 0.0, 1.0, 0.0],
        #                          [0.0, 2.0, 1.0, 0.0],
        #                          [0.0, -2.0, 1.0, 0.0],
        #                          [2.0, 2.0, 1.0, 0.0],
        #                          [-2.0, 2.0, 1.0, 0.0],
        #                          [2.0, -2.0, 1.0, 0.0],
        #                          [-2.0, -2.0, 1.0, 0.0]
        #                          ])

        # res = minimize(target, np.array([0.0, 0.0, 1.0, 0.0]), args=(pattern, pos), bounds=bounds, method="L-BFGS-B")

        results = list()

        for start_point in start_points:
            res = minimize(target, start_point, args=(pattern, pos), bounds=bounds,
                           method="L-BFGS-B", options={"disp": False})
            results.append(res)

        results.sort(key=lambda x: x.fun)

        data = results[0].x.data

        result = _TargetTransformParams(data[0], data[1], data[2], data[3])

        logger.info("affine transformation to for position calibration: {}".format(result))

        return self._get_transform_matrix(pos, result)

    @staticmethod
    def _get_transform_matrix(shape: BaseGeometry, params: _TargetTransformParams) -> TransformMatrix:
        """
        previous optimization is toggled with a given geometry, this function will release this toggle
        :param shape: a shape which the transformation toggled with
        :param params: params that toggle with the shape
        :return: transformMatrix elements
        """
        translated = translate(shape, xoff=params.x_offset, yoff=params.y_offset)
        scaled = scale(translated, xfact=params.scale, yfact=params.scale)

        scale_origin = interpret_origin(translated, "center", 2)
        rotate_origin = interpret_origin(scaled, "center", 2)

        p00 = Point(0, 0)
        p10 = Point(1, 0)
        p01 = Point(0, 1)

        p00_translated = translate(p00, xoff=params.x_offset, yoff=params.y_offset)
        p00_scaled = scale(p00_translated, xfact=params.scale, yfact=params.scale, origin=scale_origin)
        p00_rotated = rotate(p00_scaled, angle=params.angle, origin=rotate_origin)  # type: Point

        p01_translated = translate(p01, xoff=params.x_offset, yoff=params.y_offset)
        p01_scaled = scale(p01_translated, xfact=params.scale, yfact=params.scale, origin=scale_origin)
        p01_rotated = rotate(p01_scaled, angle=params.angle, origin=rotate_origin)  # type: Point

        p10_translated = translate(p10, xoff=params.x_offset, yoff=params.y_offset)
        p10_scaled = scale(p10_translated, xfact=params.scale, yfact=params.scale, origin=scale_origin)
        p10_rotated = rotate(p10_scaled, angle=params.angle, origin=rotate_origin)  # type: Point

        c = p00_rotated.x
        f = p00_rotated.y

        a = p10_rotated.x - c
        d = p10_rotated.y - f

        b = p01_rotated.x - c
        e = p01_rotated.y - f

        return TransformMatrix(a, b, d, e, c, f)



