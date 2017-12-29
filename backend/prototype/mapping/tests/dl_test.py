import unittest
import cv2
from mapping.train import BigImageDataset, get_acc, get_iou
from torch import LongTensor


class TestDataLoader(unittest.TestCase):

    # def test_big_image_loader(self):
    #
    #     dataset = BigImageDataset("../data/linuo.tif", "../data/label.png", 400, 100)
    #
    #     feature, label = dataset[0]
    #
    #     cv2.imshow("blue", feature[0])
    #     # cv2.waitKey(0)
    #
    #     cv2.imshow("label", label*255)
    #     cv2.waitKey(0)
    #
    #     cv2.destroyAllWindows()
    #

    def test_iou_calculation(self):
        x = LongTensor([[1, 1, 2, 2], [2, 2, 1, 1]])
        y = LongTensor([[1, 2, 1, 2], [2, 1, 1, 1]])
        iou1 = get_iou(x, y, 1)
        iou2 = get_iou(x, y, 2)
        self.assertAlmostEqual(iou1, 0.5)
        self.assertAlmostEqual(iou2, 0.4)

    def test_acc_calculation(self):
        x = LongTensor([[1, 1, 2, 2], [2, 2, 1, 1]])
        y = LongTensor([[1, 2, 1, 2], [2, 1, 1, 1]])
        acc = get_acc(x, y)
        self.assertAlmostEqual(acc, 0.625)


if __name__ == "__main__":
    unittest.main()
