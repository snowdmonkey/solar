import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import List
from scipy.cluster.hierarchy import linkage, cut_tree


def crop_and_rotate(img, cnt):
    masked_image = get_masked_image(img, [cnt])
    rect = cv2.minAreaRect(cnt)
    dst = rotate_and_scale(masked_image, rect[2])
    ii = np.where(dst != 0)
    corp_img = dst[min(ii[0]): (max(ii[0]) + 1), min(ii[1]): (max(ii[1]) + 1)]
    return corp_img


def get_masked_image(img, contours):
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, 255, -1)
    mask[mask == 255] = img[mask == 255]
    return mask


def rotate_and_scale(img, degree, scale_factor=1.0):
    (oldY, oldX) = img.shape  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=degree,
                                scale=scale_factor)  # rotate about center of image.
    # choose a new image size.
    newX, newY = oldX * scale_factor, oldY * scale_factor
    # include this if you want to prevent corners being cut off
    r = np.deg2rad(degree)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty
    rotated_img = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
    return rotated_img


class PanelCropper:
    def __init__(self, pic_path):
        self.raw_img = cv2.imread(pic_path, 0)
        self.blur_img = cv2.blur(self.raw_img, (3, 3))
        self.binary_seg = None
        self.contours = None

    def _cal_binary_seg(self):
        _, th = cv2.threshold(self.raw_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.binary_seg = th

    def _cal_contours(self, min_area, max_area):
        _, contours, h = cv2.findContours(self.binary_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [x for x in contours if min_area <= cv2.contourArea(x) <= max_area]

    def _verify_rectangle(self, poly_threshold=10, n_vertices_threshold=6):
        """
        The function aims to verify whether the selected sub images are rectangle. It approximate the contours to
        polygons and count the number of vertices. If the number if vertices are more a specified threshold, the
        selected part is not considered as a rectangle
        :param poly_threshold: the threshold when apply cv2.approxPolyDP
        :param n_vertices_threshold: threshold for the number of vertices
        :return: nothing
        """
        approx = [cv2.approxPolyDP(cnt, poly_threshold, True) for cnt in self.contours]
        n_vertices = [len(x) for x in approx]
        self.contours = [self.contours[i] for i in range(len(self.contours)) if n_vertices[i] <= n_vertices_threshold]

    def get_sub_imgs(self, rotate_n_crop=False, min_area=500, max_area=86000, verify_rectangle=-1,
                     n_vertices_threshold=6):
        if self.binary_seg is None:
            self._cal_binary_seg()
        # if self.contours is None:
        #     self.get_contours()
        self._cal_contours(min_area=min_area, max_area=max_area)

        if verify_rectangle >= 0:
            self._verify_rectangle(verify_rectangle, n_vertices_threshold)

        sub_imgs = list()
        for cnt in self.contours:
            masked_image = get_masked_image(self.raw_img, [cnt])
            if rotate_n_crop:
                corp_img = crop_and_rotate(masked_image, cnt)
                sub_imgs.append(corp_img)
            else:
                sub_imgs.append(masked_image)
        return sub_imgs

    def get_panels(self, min_area: int = 100, max_area: int = 1000, n_vertices_threshold: int = 6,
                   approx_threshold: int = 2) -> List:
        """
        get the solar panels from the image
        :param min_area: the minimum area of a panel in unit of pixels
        :param max_area: the maximum area of a panel in unit of pixels
        :param n_vertices_threshold: the maximum number of vertices of a panel
        :param approx_threshold: threshold when call cv.approxPolyDP to verify whether an area is a rectangle
        :return: list of sub images, each sub image contains a panel
        """
        img = self.raw_img
        blur = self.blur_img
        imgf = np.float32(img)
        mu = cv2.blur(imgf, (3, 3))
        mu2 = cv2.blur(imgf*imgf, (3, 3))
        sigma = np.sqrt(mu2 - mu * mu)
        _, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, th2 = cv2.threshold(np.uint8(sigma), 2, 255, cv2.THRESH_BINARY_INV)
        th1[th2 == 0] = 0
        _, contours, h = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sub_imgs = list()
        corners = list()
        for cnt in contours:
            if cv2.contourArea(cnt) < 100 or cv2.contourArea(cnt) > 1000:
                continue
            approx = cv2.approxPolyDP(cnt, approx_threshold, True)
            if len(approx) > 8:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # mask = np.zeros_like(img)
            # mask[y:(y+h), x:(x+w)] = blur[y:(y+h), x:(x+w)]
            # sub_imgs.append(mask)
            corners.extend([[i, j] for i in range(x, x + w, 5) for j in range(y, y + h, 5)])
        if len(corners) == 0:
            return sub_imgs
        linkage_matrix = linkage(np.array(corners), method="single", metric="chebyshev")
        ctree = cut_tree(linkage_matrix, height=[10])
        cluster = [x[0] for x in ctree]
        corner_groups = list()
        for group in range(max(cluster) + 1):
            corner_group = [corners[i] for i in range(len(corners)) if cluster[i] == group]
            corner_groups.append(np.array(corner_group))
        # corner_groups = np.array(corner_groups)
        group_contours = [cv2.convexHull(x) for x in corner_groups]
        for cnt in group_contours:
            masked_image = get_masked_image(blur, [cnt])
            sub_imgs.append(masked_image)

        return sub_imgs

    def get_plate_part(self, min_area=500):
        if self.binary_seg is None:
            self._cal_binary_seg()
        # if self.contours is None:
        #     self.get_contours()
        self._cal_contours(min_area=min_area)

        masked_image = get_masked_image(self.raw_img, self.contours)
        return masked_image


def main():
    plate_cropper = PanelCropper("../pic/DJI_0001.jpg")
    sub_imgs = plate_cropper.get_sub_imgs(rotate_n_crop=True)
    n = len(sub_imgs)
    n_row = n // 3 + 1
    n_col = 3
    plt.figure()
    for i in range(n):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(sub_imgs[i], cmap='gray')
    plt.show()
    #
    # plate_part = plate_cropper.get_plate_part()
    # plt.imshow(plate_part, cmap='gray')
    # plt.show()


if __name__ == "__main__":
    main()
