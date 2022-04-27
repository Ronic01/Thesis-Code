import cv2
import numpy as np


class OrthographicImage:
    def __init__(self, mat, pixel_size, min_depth=None, max_depth=None, camera=None, pose=None):
        self.mat = mat
        self.pixel_size = pixel_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.camera = camera
        self.pose = pose

        self.min_value = 0
        self.max_value = 255 * 255

    def project(self, point):
        return (
            int(round(self.mat.shape[1] / 2 - self.pixel_size * point[1])),
            int(round(self.mat.shape[0] / 2 - self.pixel_size * point[0])),
        )

    def depth_from_value(self, value: float) -> float:
        return self.max_depth + (value / self.max_value) * (self.min_depth - self.max_depth)

    def value_from_depth(self, depth: float) -> float:
        value = round((depth - self.max_depth) / (self.min_depth - self.max_depth) * self.max_value)
        return np.clip(value, self.min_value, self.max_value)
