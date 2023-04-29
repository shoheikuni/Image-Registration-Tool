import numpy as np
import os
import glob
import csv
import datetime
from distutils.util import strtobool

def tuple_map(func, values):
    return tuple(map(func, values))

def csv_str_to_list(csv_str):
    """ "a,b,c" -> ["a","b","c"] """
    result = []
    for row in csv.reader([csv_str]):
        result = result + row
    return result

def get_imagefilenames_in(dirpath):
    return glob.glob(dirpath + '/*.jpg')

def rotation_matrix_deg(theta_deg):
    theta = np.deg2rad(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
    return R

def affine_matrix(linear_matrix):
    affine = np.identity(3)
    affine[0, 0] = linear_matrix[0, 0]
    affine[0, 1] = linear_matrix[0, 1]
    affine[1, 0] = linear_matrix[1, 0]
    affine[1, 1] = linear_matrix[1, 1]
    return affine

def translation_matrix(translation):
    affine = np.identity(3)
    affine[0, 2] = translation[0]
    affine[1, 2] = translation[1]
    return affine
 
def image_rotation_matrix(img_width, img_height, rotation_angle_deg):
    """ サイズがimg_width x img_heightの画像の座標系から、それを反時計回りにrotation_angle_deg度まわした画像の座標系への変換行列 """
    R = rotation_matrix_deg(-rotation_angle_deg) # 一般的なxy座標平面はy軸が上だが、画像座標系ではy軸は下向き。ゆえに角度の符号を反転する。
    rotated_img_width, rotated_img_height = map(abs, R @ np.array([img_width, img_height]))

    img_center = np.array([img_width / 2, img_height / 2], dtype=int)
    T1 = translation_matrix(-img_center)

    rotated_img_center = np.array([rotated_img_width / 2, rotated_img_height / 2], dtype=int)
    T2 = translation_matrix(rotated_img_center)

    return T2 @ affine_matrix(R) @ T1
   
def homo(vec):
    n = len(vec)
    a = np.empty(n + 1)
    a[0:n] = vec
    a[n] = 1
    return a
def dehomo(vec):
    n = len(vec)
    return vec[0:n-1]

def matmul_ashomo(mat, vec):
    return dehomo(mat @ homo(vec))


def read_annotation(annotationfile):
    annotationlines = annotationfile.readlines()
    rotation_angle_deg = int(annotationlines[0])
    enabled = strtobool(annotationlines[1].strip())
    dots = [tuple_map(int, csv_str_to_list(annotationline)) for annotationline in annotationlines[2:]]
    return (rotation_angle_deg, enabled, dots)

def write_annotation(annotationfile, rotation_angle_deg, enabled, dots):
    lines = []
    lines.append(str(rotation_angle_deg))
    lines.append(str(enabled))
    lines = lines + [','.join(map(str, dot)) for dot in dots]
    annotationfile.write('\n'.join(lines))


def exif_dateTimeOriginal(img):
    exif = img._getexif()
    if not exif:
        return None
    DATE_TIME_ORIGINAL_CODE = 36867
    if DATE_TIME_ORIGINAL_CODE in exif:
        datetime_str = exif[DATE_TIME_ORIGINAL_CODE]
        return datetime.datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
    return None





