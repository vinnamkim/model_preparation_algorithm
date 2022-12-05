import os
from PIL import Image, ImageOps, ImageEnhance
from time import time
import cython_augments.augment as pil_aug
import cython_augments.cv_augment as cv_aug
import numpy as np
import glob
import random
from copy import deepcopy

random.seed(3005)
num_imgs = 64

img_files = glob.glob("/home/vinnamki/dataset/imagenette2/**/*.JPEG", recursive=True)
random.shuffle(img_files)
img_files = img_files[:num_imgs]

size = (224, 224)
# size = (512, 512)

o_imgs = [Image.open(img_file).resize(size) for img_file in img_files]
for img in o_imgs:
    print(img.mode)
    assert img.mode == "RGB"

def rotate(img, degrees, **kwargs):
    kwargs["resample"] = Image.Resampling.BILINEAR
    return img.rotate(degrees, **kwargs)


def shear_x(img, factor, **kwargs):
    kwargs["resample"] = Image.Resampling.BILINEAR
    return img.transform(img.size, Image.Transform.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    kwargs["resample"] = Image.Resampling.BILINEAR
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    kwargs["resample"] = Image.Resampling.BILINEAR
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    kwargs["resample"] = Image.Resampling.BILINEAR
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def _bench(func, *args, n_iter: int = 10):
    dt = 0
    for _ in range(n_iter):
        imgs = [deepcopy(img) for img in o_imgs]

        start = time()
        for img in imgs:
            func(img, *args)
        dt += time() - start
    return dt * 1000.0 / n_iter


dt1 = _bench(ImageOps.autocontrast)
dt2 = _bench(pil_aug.autocontrast)
print(f"Autocontrast, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(ImageOps.equalize)
dt2 = _bench(pil_aug.equalize)
print(f"Equalize, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(ImageOps.posterize, 5)
dt2 = _bench(pil_aug.posterize, 5)
print(f"Posterize, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(ImageOps.solarize, 128)
dt2 = _bench(pil_aug.solarize, 128)
print(f"Solarize, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(lambda img, factor: ImageEnhance.Color(img).enhance(factor), 0.2)
dt2 = _bench(pil_aug.color, 0.2)
print(f"Color, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(lambda img, factor: ImageEnhance.Contrast(img).enhance(factor), 0.8)
dt2 = _bench(pil_aug.contrast, 0.8)
print(f"Contrast, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(lambda img, factor: ImageEnhance.Brightness(img).enhance(factor), 0.2)
dt2 = _bench(pil_aug.brightness, 0.2)
print(f"Brightness, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(lambda img, factor: ImageEnhance.Sharpness(img).enhance(factor), 2.0)
dt2 = _bench(pil_aug.sharpness, 2.0)
print(f"Sharpness, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(lambda img, factor: rotate(img, factor), 20)
dt2 = _bench(pil_aug.rotate, 20)
print(f"Rotate, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(lambda img, factor: translate_x_rel(img, factor), 0.2)
dt2 = _bench(pil_aug.translate_x_rel, 0.2)
print(f"TranslateXRel, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(lambda img, factor: translate_y_rel(img, factor), 0.2)
dt2 = _bench(pil_aug.translate_y_rel, 0.2)
print(f"TranslateYRel, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(lambda img, factor: shear_x(img, factor), 0.2)
dt2 = _bench(pil_aug.shear_x, 0.2)
print(f"ShearX, {dt1:.1f}, {dt2:.1f}")

dt1 = _bench(lambda img, factor: shear_y(img, factor), 0.2)
dt2 = _bench(pil_aug.shear_y, 0.2)
print(f"ShearY, {dt1:.1f}, {dt2:.1f}")
