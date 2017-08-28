import sys
from extract_rect import PanelCropper
import numpy as np
from matplotlib import pyplot as plt


class HotSpot:

    def __init__(self, points=None, absolute_brightness=None, relative_brightness=None):
        self.points = points
        self.absolute_brightness = absolute_brightness
        self.relative_brightness = relative_brightness


class HotSpotDetector:

    def __init__(self, img, threshold_coefficient=2.0):
        """
        :param img: numpy.ndarray, image
        :param threshold_coefficient:  positive float, upper threshold for 
        hot spots will be mean + threshold_coefficient * std
        """
        self.raw_img = img
        self.threshold_coefficient = threshold_coefficient
        self.hot_spot = None

    def set_threshold_coefficient(self, threshold_coefficient):
        self.threshold_coefficient = threshold_coefficient
        self.hot_spot = None

    def _calculate_hot_spot(self):
        img = self.raw_img
        data = [x for x in img.reshape(img.size) if x != 0]
        limits = np.percentile(data, (10, 90))
        trimmed_data = [x for x in data if limits[0] < x < limits[1]]
        mu = np.mean(trimmed_data)
        sd = np.std(trimmed_data)
        threshold = mu + self.threshold_coefficient * sd
        points = np.argwhere(img > threshold)
        if len(points) > 0:
            hot_data = list()
            for coord in points:
                hot_data.append(img[tuple(coord)])
            absolute_brightness = np.mean(hot_data)
            relative_brightness = absolute_brightness - mu
        else:
            absolute_brightness = None
            relative_brightness = None
        self.hot_spot = HotSpot(points, absolute_brightness, relative_brightness)

    def get_hot_spot(self):
        if self.hot_spot is None:
            self._calculate_hot_spot()
        return self.hot_spot


def main():
    panel_cropper = PanelCropper("../pic/DJI_0001.jpg")
    sub_imgs = panel_cropper.get_sub_imgs(rotate_n_crop=True)
    n = len(sub_imgs)
    n_row = n // 3 + 1
    n_col = 3

    coefficient = 1.0

    if len(sys.argv) > 1:
        try:
            coefficient = float(sys.argv[1])
        except ValueError:
            pass

    plt.figure()
    j = 0
    for img in sub_imgs:
        j += 1
        hot_spot_detector = HotSpotDetector(img, coefficient)
        hot_spot = hot_spot_detector.get_hot_spot()
        mask = np.zeros_like(img)
        for coord in hot_spot.points:
            mask[tuple(coord)] = 255
        plt.subplot(n_row, n_col, j)
        plt.imshow(mask, cmap='gray')
    # plt.savefig('../pic/output.png')

    plt.show()


if __name__ == "__main__":
    main()

