import os
import glob
import sys
import csv
import numpy as np
import cv2
import argparse
from PIL import Image
from distutils.util import strtobool
from common import *


def open_cv2img(filepath):
    pil_img = Image.open(filepath)
    img = np.array(pil_img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def create_rotated_dots(dots, img_width, img_height, rotation_angle_deg):
    rotation_matrix = image_rotation_matrix(img_width, img_height, rotation_angle_deg)
    return [ matmul_ashomo(rotation_matrix, dot).astype(int) for dot in dots]

def create_rotated_img(cv2_img, rotation_angle_deg):
    right_angles = int(rotation_angle_deg / 90)
    if right_angles == 0:
        return cv2_img.copy()
    elif right_angles == 1:
        return cv2.rotate(cv2_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif right_angles == 2:
        return cv2.rotate(cv2_img, cv2.ROTATE_180)
    elif right_angles == 3:
        return cv2.rotate(cv2_img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return None

def create_rect_corners(viewport_size, rect_aspect_ratio):
    width, height = viewport_size
    viewport_aspect_ratio = width / height
    if rect_aspect_ratio < viewport_aspect_ratio:
        rect_width = height * rect_aspect_ratio
        rect_height = height
    else:
        rect_width = width
        rect_height = width / rect_aspect_ratio
    rect_top = (height - rect_height) / 2
    rect_bottom = rect_height
    rect_left = (width - rect_width) / 2
    rect_right = width - rect_left
    return [(rect_left, rect_top), (rect_right, rect_top), (rect_right, rect_bottom), (rect_left, rect_bottom)]
 
def create_average_corners(arrayOfHomographicRectangleCorners, viewport_size, rect_aspect_ratio):
    rect_corners = create_rect_corners(viewport_size, rect_aspect_ratio)
    rect_corners_in_world_coord = [(x, y, 0) for x, y in rect_corners] # 単純な正射影による世界座標系を考える
    rect_corners_in_world_coord = np.array(rect_corners_in_world_coord, np.float32)

    imagePoints = [dots for _, _, dots in data]
    imagePoints = np.array(imagePoints, np.float32)
    objectPoints = [rect_corners_in_world_coord] * len(data)
    objectPoints = np.array(objectPoints, np.float32)
    reprojectionError, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, viewport_size, None, None)

    average_rvec = sum(rvecs) / len(rvecs)
    average_tvec = sum(tvecs) / len(tvecs)

    average_corners, _ = cv2.projectPoints(rect_corners_in_world_coord, average_rvec, average_tvec, cameraMatrix, distCoeffs)

    return average_corners

   
def in_order(dots):
    center = sum(dots) / 4
    topleft_dot = next(dot for dot in dots if dot[0] < center[0] and dot[1] < center[1])
    topright_dot = next(dot for dot in dots if dot[0] >= center[0] and dot[1] < center[1])
    bottomleft_dot = next(dot for dot in dots if dot[0] < center[0] and dot[1] >= center[1])
    bottomright_dot = next(dot for dot in dots if dot[0] >= center[0] and dot[1] >= center[1])
    return [topleft_dot, topright_dot, bottomright_dot, bottomleft_dot]

def parse_aspect_ratio(aspect_ratio_str):
    try:
        value = float(aspect_ratio_str)
        return value
    except ValueError:
        splitted = aspect_ratio_str.split(':')
        if len(splitted) != 2:
            raise ValueError
        width = float(splitted[0])
        height = float(splitted[1])
        ratio = width / height
        return ratio


parser = argparse.ArgumentParser()

parser.add_argument("input_dir", help="Where input images and annotation files exist.")
parser.add_argument("rect_aspect_ratio",
                    type=parse_aspect_ratio,
                    help="The real aspect ratio(<width>:<height> or <real number>) of a rectangle "
                    "the corners of which will match the annotation dots in each image.")
parser.add_argument("output_dir", help="Where to output fitted images.")
parser.add_argument("--average-shape", action="store_true",
                    help="If specified, "
                    "images will be so fitted that the annotation dots match the corners of an average shape "
                    "which is calculated over the dots of all input images. "
                    "If not specified, images will be fitted to a simple rectangle with rect_aspect_ratio."
                    "(The simple rectangle will be located as widely in the image as possible "
                    "and therefore the region out of the rectangle will be lost.)"
                    )
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

data = []

# read annotations
img_filepaths = glob.glob(os.path.join(args.input_dir, '*.jpg'))
for i, img_filepath in enumerate(img_filepaths):
    print("reading annotations ", i, "/", len(img_filepaths), flush=True, end="\r")
    anno_filepath = img_filepath + '.anno'
    if not os.path.exists(anno_filepath):
        continue

    with open(anno_filepath, "rt") as annotationfile:
        rotation_angle_deg, enabled, original_dots = read_annotation(annotationfile)
    if not enabled:
        continue
    if len(original_dots) != 4:
        continue

    original_img = open_cv2img(img_filepath)
    original_img_height, original_img_width, c = original_img.shape

    dots = create_rotated_dots(original_dots, original_img_width, original_img_height, rotation_angle_deg)
    dots = in_order(dots)

    img = create_rotated_img(original_img, rotation_angle_deg)

    data.append((img_filepath, img, dots))

print("")

def img_size(img):
    return (img.shape[1], img.shape[0])

import collections
sizeCounts = collections.Counter([img_size(img) for _, img, _ in data])
output_size = max(sizeCounts, key=sizeCounts.get)

def resize_dot(dot, oldsize, newsize):
    return (dot[0] / oldsize[0] * newsize[0],
            dot[1] / oldsize[1] * newsize[1])

# resize images
for i, d in enumerate(data):
    _, img, dots = d

    if img_size(img) != output_size:
        img = cv2.resize(img, dsize=output_size)
        dots = list(map(lambda dot: resize_dot(dot, img_size(img), output_size), dots))
        data[i] = (img_filepath, img, dots)

if args.average_shape:
    print("creating the average shape (will take a little time)")
    target_corners = create_average_corners([dots for _, img, _dots in data], output_size, args.rect_aspect_ratio)
    target_corners = np.array(target_corners, np.float32)
    print("")
else:
    target_corners = create_rect_corners(output_size, args.rect_aspect_ratio)
    target_corners = np.array(target_corners, np.float32)

# fit images
for i, d in enumerate(data):
    print("fitting ", i, "/", len(data), flush=True, end="\r")

    img_filepath, img, dots = d

    pil_img = Image.open(img_filepath)
    dateTimeOriginal = exif_dateTimeOriginal(pil_img)
    text = dateTimeOriginal.strftime('%Y/%m/%d')

    dots = np.array(dots, np.float32)
    
    homographyTransform = cv2.getPerspectiveTransform(dots, target_corners)

    fitted_img = cv2.warpPerspective(img, homographyTransform, output_size)

    cv2.putText(fitted_img, text, (100, 200), cv2.FONT_HERSHEY_COMPLEX, 5, (255, 255, 255), 20, cv2.LINE_AA)

    output_img_filepath = os.path.join(args.output_dir, os.path.basename(img_filepath) + '_fitted.jpg')
    cv2.imwrite(output_img_filepath, fitted_img)


