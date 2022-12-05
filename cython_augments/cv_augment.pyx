# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np
np.import_array()
import cv2
from PIL import Image

ctypedef np.int32_t INT32_t
ctypedef np.uint8_t UINT8_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[INT32_t, ndim=1] c_histogram(const UINT8_t[:, :, :] image):
    cdef INT32_t [:] hist = np.zeros((768,), dtype=np.int32)
    cdef int height, width, y, x
    cdef UINT8_t r, g, b

    height = image.shape[0]
    width = image.shape[1]

    for y in range(height):
        for x in range(width):
            hist[image[y][x][0] + 000] += 1
            hist[image[y][x][1] + 256] += 1
            hist[image[y][x][2] + 512] += 1

    return np.asarray(hist)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _c_lut(np.ndarray[UINT8_t, ndim=3] image, const UINT8_t[:] r_lut):
    cdef np.ndarray[UINT8_t, ndim=3] lut = np.zeros((1, 256, 3), dtype=np.uint8)
    cdef UINT8_t[:, :, :] lut_view = lut
    cdef int i
    for i in range(256):
        lut_view[0][i][0] = r_lut[i + 000]
        lut_view[0][i][1] = r_lut[i + 256]
        lut_view[0][i][2] = r_lut[i + 512]

    return cv2.LUT(image, lut)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def autocontrast(image, cutoff=0, ignore=None, mask=None, preserve_tone=False):
    """
    Maximize (normalize) image contrast. This function calculates a
    histogram of the input image (or mask region), removes ``cutoff`` percent of the
    lightest and darkest pixels from the histogram, and remaps the image
    so that the darkest pixel becomes black (0), and the lightest
    becomes white (255).

    :param image: The image to process.
    :param cutoff: The percent to cut off from the histogram on the low and
                   high ends. Either a tuple of (low, high), or a single
                   number for both.
    :param ignore: The background pixel value (use None for no background).
    :param mask: Histogram used in contrast operation is computed using pixels
                 within the mask. If no mask is given the entire image is used
                 for histogram computation.
    :param preserve_tone: Preserve image tone in Photoshop-like style autocontrast.

                          .. versionadded:: 8.2.0

    :return: An image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = np.asarray(image)

    cdef int layer = 0
    cdef int length = 756
    cdef int* h
    cdef int i, lo, hi, ix, cut
    cdef double scale, offset
    cdef int[:] histogram
    cdef UINT8_t[:] lut = np.zeros((768,), dtype=np.uint8)

    histogram = c_histogram(image)

    while layer < length:
        h = &histogram[layer]

        if ignore is not None:
            # get rid of outliers
            try:
                h[ignore] = 0
            except TypeError:
                # assume sequence
                for ix in ignore:
                    h[ix] = 0
        if cutoff:
            # cut off pixels from both ends of the histogram
            if not isinstance(cutoff, tuple):
                cutoff = (cutoff, cutoff)
            # get number of pixels
            n = 0
            for ix in range(256):
                n = n + h[ix]
            # remove cutoff% pixels from the low end
            cut = n * cutoff[0] // 100
            for lo in range(256):
                if cut > h[lo]:
                    cut = cut - h[lo]
                    h[lo] = 0
                else:
                    h[lo] -= cut
                    cut = 0
                if cut <= 0:
                    break
            # remove cutoff% samples from the high end
            cut = n * cutoff[1] // 100
            for hi in range(255, -1, -1):
                if cut > h[hi]:
                    cut = cut - h[hi]
                    h[hi] = 0
                else:
                    h[hi] -= cut
                    cut = 0
                if cut <= 0:
                    break
        # find lowest/highest samples after preprocessing
        for lo in range(256):
            if h[lo]:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            # don't bother
            for i in range(256):
                lut[layer + i] = i
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            for ix in range(256):
                i = ix
                ix = (int)(ix * scale + offset)
                if ix < 0:
                    ix = 0
                elif ix > 255:
                    ix = 255
                lut[layer + i] = ix

        layer += 256

    result = _c_lut(image, lut)
    return Image.fromarray(result)
