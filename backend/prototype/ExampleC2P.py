# Import the api
import Coords2Pix.C2P as C2P

# Call this method to convert Degree Minute Second to Degree Decimal Format I just used coordinates from a random image
# method has the following form dmsToDD(degree,minutes,seconds) returns a Decimal coordinate value
x = C2P.dmsToDD(33, 35, 13.511899999997752)
y = C2P.dmsToDD(119, 38, 9.328199999988982)

# Then pass these coords to the following method with a GeoTiff image and it will return the nearest pixel location
# of that gps location method has the following form MapPixelCoords(img,XCoord,YCoord) returns a 2 elements list,
# result[0] is the x pixel, result[1] is the y pixel
print(C2P.MapPixelCoords('6-21.tif', x, y))

# should print [3328.0,2679.0]
