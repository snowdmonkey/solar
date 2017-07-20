import cv2
import numpy as np
from matplotlib import pyplot as plt


def crop_and_rotate(img, cnt):
    masked_image = get_masked_image(img, [cnt])
    rect = cv2.minAreaRect(cnt)
    dst = rotate_and_scale(masked_image, rect[2])
    ii = np.where(dst != 0)
    corp_img = dst[min(ii[0]): max(ii[0]), min(ii[1]): max(ii[1])]
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


class PlateCropper:

    def __init__(self, pic_path):
        self.raw_img = cv2.imread(pic_path, 0)
        self.binary_seg = None
        self.contours = None

    def get_binary_seg(self):
        _, th = cv2.threshold(self.raw_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.binary_seg = th

    def get_contours(self, min_area=500):
        _, contours, h = cv2.findContours(self.binary_seg, 1, 2)
        self.contours = [x for x in contours if cv2.contourArea(x) >= min_area]

    def get_sub_imgs(self, rotation_n_crop=False):
        if self.binary_seg is None:
            self.get_binary_seg()
        if self.contours is None:
            self.get_contours()
        sub_imgs = list()
        for cnt in self.contours:
            masked_image = get_masked_image(self.raw_img, [cnt])
            if rotation_n_crop:
                corp_img = crop_and_rotate(masked_image, cnt)
                sub_imgs.append(corp_img)
            else:
                sub_imgs.append(masked_image)
        return sub_imgs

    def get_plate_part(self):
        if self.binary_seg is None:
            self.get_binary_seg()
        if self.contours is None:
            self.get_contours()

        masked_image = get_masked_image(self.raw_img, self.contours)
        return masked_image


def main():
    plate_cropper = PlateCropper("../pic/DJI_0001.jpg")
    sub_imgs = plate_cropper.get_sub_imgs(rotation_n_crop=True)
    n = len(sub_imgs)
    n_row = n // 3 + 1
    n_col = 3
    plt.figure()
    for i in range(n):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(sub_imgs[i], cmap='gray')
    plt.show()
    #
    # plate_part = plate_cropper.get_plate_part()
    # plt.imshow(plate_part, cmap='gray')
    # plt.show()


if __name__ == "__main__":
    main()

