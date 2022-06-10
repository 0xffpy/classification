from pyproj import CRS, Proj, transform

# last occurnce of text gps
PIXEL_TO_METER = 10  # pixel ratio
CENTER_OF_CAMERA_Y = 2000
CENTER_OF_CAMERA_X = 3000


class Loclization:
    __center_of_y = 0
    __center_of_x = 0
    __gps_drone = 0
    __orientation = 0
    __PIXEL_TO_METER = 10  # pixel ratio
    __CENTER_OF_CAMERA_Y = 2000
    __CENTER_OF_CAMERA_X = 3000

    def __init__(self, center_of_y, center_of_x, gps_drone, oriantaion):
        self.__center_of_y = center_of_y
        self.__center_of_x = center_of_x
        self.__gps_drone = gps_drone
        self.oriantaion = oriantaion

    def __meters_from_xy(self):  # center of object in the orignal image get it from the mser
        y_dist = self.__CENTER_OF_CAMERA_Y - self.__center_of_y
        x_dist = self.__CENTER_OF_CAMERA_X - self.__center_of_x
        return x_dist * self.__PIXEL_TO_METER, y_dist * self.__PIXEL_TO_METER

    def __lang_lat_from_meters(self):
        x, y = self.__meters_from_xy()
        proj_latlon = Proj(proj='latlong', datum='WGS84')
        proj_xy = Proj(proj="utm", zone=33, datum='WGS84')
        lonlat = transform(proj_xy, proj_latlon, x, y)
        return lonlat[0], lonlat[1]

    def object_location(self):
        lon, lat = self.__lang_lat_from_meters()
        return self.gps_drone[0] + lon, self.gps_drone[1] + lat
# don't forget to encode the oriantation

    def oriantation(self):
        return None

