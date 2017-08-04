import gdal
import utm
import cv2
import numpy as np


def dmsToDD(degree,minutes,seconds):
    decimal= np.float64(degree) + np.float64(minutes)/60 + np.float64(seconds)/3600
    return decimal


def MapPixelCoords(mp,x,y):
    wgsCoords = utm.from_latlon(x,y)
    dataset = gdal.Open(mp)
    geotransform = dataset.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    
   
    hold1 = wgsCoords[0]-originX
    xPix = round(np.float64(hold1)/np.float64(pixelWidth))
    
    hold2 = wgsCoords[1]-originY
    yPix = round(np.float64(hold2)/np.float64(pixelHeight))
    return [xPix,yPix]



