from typing import Any, Dict, List, Sequence, Tuple, Optional, Union

from affine import Affine
import cv2
import numpy as np

from utils.orthographic_image import OrthographicImage


def get_transformation(x: float, y: float, a: float, center: Sequence[float], scale=1.0, cropped: Sequence[float] = None):  # [rad]
    trans = cv2.getRotationMatrix2D((round(center[0] - x), round(center[1] - y)), a * 180.0 / np.pi, scale)  # [deg]
    trans[0][2] += x + scale * cropped[0] / 2 - center[0] if cropped else x
    trans[1][2] += y + scale * cropped[1] / 2 - center[1] if cropped else y
    return trans


def get_area_of_interest(image: OrthographicImage, pose: dict, size_cropped: Tuple[float, float] = None, size_result: Tuple[float, float] = None):
    size_input = (image.mat.shape[1], image.mat.shape[0])
    center_image = (size_input[0] / 2, size_input[1] / 2)

    if size_result and size_cropped:
        scale = size_result[0] / size_cropped[0]
        assert scale == (size_result[1] / size_cropped[1])
    elif size_result:
        scale = size_result[0] / size_input[0]
        assert scale == (size_result[1] / size_input[1])
    else:
        scale = 1.0

    size_final = size_result or size_cropped or size_input

    trans = get_transformation(
        image.pixel_size * pose['y'],
        image.pixel_size * pose['x'],
        -pose['a'],
        center_image,
        scale=scale,
        cropped=size_cropped,
    )
    return cv2.warpAffine(image.mat, trans, size_final, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # INTERPOLATION_METHOD


def get_rect_contour(center: Sequence[float], size: Sequence[float]) -> List[Sequence[float]]:
    return [
        [center[0] + size[0] / 2, center[1] + size[1] / 2],
        [center[0] + size[0] / 2, center[1] - size[1] / 2],
        [center[0] - size[0] / 2, center[1] - size[1] / 2],
        [center[0] - size[0] / 2, center[1] + size[1] / 2],
    ]


def draw_line(image: OrthographicImage, action_affine: Affine, pt1, pt2, color, thickness=1):
    cm = 1. / 255 if image.mat.dtype == np.float32 else np.iinfo(image.mat.dtype).max / 255
    pt1_projection = image.project(action_affine * pt1)
    pt2_projection = image.project(action_affine * pt2)
    cv2.line(image.mat, tuple(pt1_projection), tuple(pt2_projection), (color[0] * cm, color[1] * cm, color[2] * cm), thickness, lineType=cv2.LINE_AA)

def draw_line_simple(image: OrthographicImage, addonimage, action_affine: Affine, pt1, pt2, color, thickness=1):
    cm = 1. / 255 if image.mat.dtype == np.float32 else np.iinfo(image.mat.dtype).max / 255
    pt1_projection = image.project(action_affine * pt1)
    pt2_projection = image.project(action_affine * pt2)
    print(pt1_projection,pt2_projection)
    cv2.line(addonimage, pt1_projection, pt2_projection, (255,255,255), thickness, lineType=cv2.LINE_AA)



def draw_polygon(image: OrthographicImage, action_affine: Affine, polygon, color, thickness=1):
    cm = 1. / 255 if image.mat.dtype == np.float32 else np.iinfo(image.mat.dtype).max / 255
    polygon_projection = np.asarray([tuple(image.project(action_affine * p)) for p in polygon])
    cv2.polylines(image.mat, [polygon_projection], True, (color[0] * cm, color[1] * cm, color[2] * cm), thickness, lineType=cv2.LINE_AA)


def draw_around_box(image: OrthographicImage, box_data):
    if not box_data:
        return

    image_border = [p[:2] for p in get_rect_contour([0.0, 0.0, 0.0], [10.0, 10.0, box_data['contour'][0][2]])]

    number_channels = image.mat.shape[-1] if len(image.mat.shape) > 2 else 1
    color_multiplier = 1. / 255 if image.mat.dtype == np.float32 else np.iinfo(image.mat.dtype).max / 255

    box_border_projection = [image.project(p[:2]) for p in box_data['contour']]

    color_color = [box_data['color'][0] * color_multiplier, box_data['color'][1] * color_multiplier, box_data['color'][2] * color_multiplier]  # Color of box
    depth_color = [max(image.value_from_depth(image.pose['z'] - border[2]) for border in box_data['contour']) * color_multiplier / 255]

    if number_channels == 1:
        color = np.array(depth_color)
    elif number_channels == 3:
        color = np.array(color_color)
    else:
        color = np.array(color_color + depth_color)

    image_border_projection = [image.project(p) for p in image_border]
    cv2.fillPoly(image.mat, np.array([image_border_projection, box_border_projection]), color)


def draw_pose(image: OrthographicImage, action_pose, convert_to_rgb=False, border_thickness=2, line_thickness=1):
    if convert_to_rgb and image.mat.ndim == 2:
        image.mat = cv2.cvtColor(image.mat, cv2.COLOR_GRAY2RGB)

    color_rect = (255, 0, 0)  # Blue
    color_lines = (0, 0, 255)  # Red

    rect = get_rect_contour([0.0, 0.0, 0.0], [200.0 / image.pixel_size, 200.0 / image.pixel_size, 0.0])
    action_affine = Affine.translation(action_pose['x'], action_pose['y']) * Affine.rotation(action_pose['a'] * 180 / 3.1415)

    draw_polygon(image, action_affine, rect, color_rect, 2)

    draw_line(image, action_affine, (90 / image.pixel_size, 0), (100 / image.pixel_size, 0), color_rect, border_thickness)
    # One end of the gripper
    draw_line(image, action_affine, (0.012, action_pose['d'] / 2), (-0.012, action_pose['d'] / 2), color_lines, line_thickness)
    # Another end of the gripper
    draw_line(image, action_affine, (0.012, -action_pose['d'] / 2), (-0.012, -action_pose['d'] / 2), color_lines, line_thickness)
    # Horizontal line through center of the grasp showing gripper closing
    draw_line(image, action_affine, (0, action_pose['d'] / 2), (0, -action_pose['d'] / 2), color_lines, line_thickness) 
    # Vertical line through center of the grasp
    draw_line(image, action_affine, (0.006, 0), (-0.006, 0), color_lines, line_thickness)

def draw_grasp_line(image: OrthographicImage,img, action_pose, convert_to_rgb=False, border_thickness=2, line_thickness=1):
    if convert_to_rgb and image.mat.ndim == 2:
        image.mat = cv2.cvtColor(image.mat, cv2.COLOR_GRAY2RGB)
    color_rect = (255, 0, 0)  # Blue
    color_lines = (0, 0, 255)  # Red
    action_affine = Affine.translation(action_pose['x'], action_pose['y']) * Affine.rotation(action_pose['a'] * 180 / 3.1415)
    draw_line_simple(image,img, action_affine, (0.006, 0), (-0.006, 0), color_lines, line_thickness)