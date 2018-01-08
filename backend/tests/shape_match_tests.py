import unittest
from shape_match import Aligner, _target_affine_transform, _TargetTransformParams
from shapely.geometry import Polygon, MultiPolygon
from matplotlib import pyplot


def plot_polygon(ax, polygon: Polygon, **kwargs):
    x, y = polygon.exterior.xy
    ax.plot(x, y, **kwargs)


class AlignerTest(unittest.TestCase):

    def test_rect(self):
        aligner = Aligner()
        shape = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
        params = _TargetTransformParams(x_offset=0.2, y_offset=0.3, scale=1, angle=0)
        pos = _target_affine_transform(shape, params)
        # box = Polygon([(-3, -3), (3, -3), (3, 3), (-3, 3)])
        # neg = box.difference(pos)
        res = aligner.align(shape, pos)
        print(res)
        self.assertAlmostEqual(params.x_offset, -res.x_offset, places=3)
        self.assertAlmostEqual(params.y_offset, -res.y_offset, places=3)
        self.assertAlmostEqual(params.scale, 1 / res.scale, places=3)
        # self.assertAlmostEqual(params.y_scale, 1 / res.y_scale, places=3)
        self.assertAlmostEqual(params.angle, -res.angle, places=3)

    def test_multi_rect(self):
        aligner = Aligner()
        r1 = Polygon([(1, 1), (3, 1), (3, 2), (1, 2)])
        r2 = _target_affine_transform(r1, _TargetTransformParams(0, 2, 1, 0))
        r3 = _target_affine_transform(r2, _TargetTransformParams(0, 2, 1, 0))

        r4 = Polygon([(4, 1), (5, 1), (5, 2), (4, 2)])
        r5 = _target_affine_transform(r4, _TargetTransformParams(0, 2, 1, 0))
        r6 = _target_affine_transform(r5, _TargetTransformParams(0, 2, 1, 0))

        shape = MultiPolygon([r1, r2, r3, r4, r5, r6])
        params = _TargetTransformParams(x_offset=0.2, y_offset=0.3, scale=1.1, angle=20)

        pos = _target_affine_transform(shape, params)
        # box = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        # neg = box.difference(pos)
        res = aligner.align(shape, pos)
        out = _target_affine_transform(pos, res)
        print(res)

        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        for polygon in shape:
            plot_polygon(ax, polygon, color="b")
        for polygon in out:
            plot_polygon(ax, polygon, color="r")
        for polygon in pos:
            plot_polygon(ax, polygon, color="g")

        fig.savefig("result_multi_rect.png")

        self.assertAlmostEqual(params.x_offset, -res.x_offset, places=3)
        self.assertAlmostEqual(params.y_offset, -res.y_offset, places=3)
        self.assertAlmostEqual(params.scale, 1 / res.scale, places=3)
        # self.assertAlmostEqual(params.y_scale, 1 / res.y_scale, places=3)
        self.assertAlmostEqual(params.angle, -res.angle, places=3)

    def test_unequal_target(self):
        aligner = Aligner()
        shape = Polygon([(0, 0), (5, 0), (5, 1), (0, 1)])
        pos = Polygon([(2, 0.2), (3, 0.2), (3, 1.2), (2, 1.2)])
        res = aligner.align(shape, pos)
        print(res)

        out = _target_affine_transform(pos, res)

        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        plot_polygon(ax, shape, color="b")
        plot_polygon(ax, out, color="r")
        plot_polygon(ax, pos, color="g")

        fig.savefig("result_unequal_target.png")


if __name__ == "__main__":
    unittest.main()
