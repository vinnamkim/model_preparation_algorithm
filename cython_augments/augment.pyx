#cython: language_level=3
# cython: profile=False

import cython
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
np.import_array()


def _lut(image, lut):
    if image.mode == "P":
        # FIXME: apply to lookup table, not image data
        raise NotImplementedError("mode P support coming soon")
    elif image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        raise OSError("not supported for this image mode")


cdef struct PixelRGBA:
    unsigned char r
    unsigned char g
    unsigned char b
    unsigned char a


cdef struct ImageInfo:
    int width
    int height
    PixelRGBA** img_ptr


cdef ImageInfo parse_img_info(image):
    cdef ImageInfo info
    cdef unsigned long long ptr_val

    info.width = image.size[0]
    info.height = image.size[1]

    ptr_val = dict(image.getdata().unsafe_ptrs)['image']
    info.img_ptr = (<PixelRGBA**>ptr_val)

    return info


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _c_lut(image, int[:] lut):
    cdef ImageInfo info
    info = parse_img_info(image)

    for y in range(info.height):
        for x in range(info.width):
            info.img_ptr[y][x].r = lut[info.img_ptr[y][x].r]
            info.img_ptr[y][x].g = lut[info.img_ptr[y][x].g + 256]
            info.img_ptr[y][x].b = lut[info.img_ptr[y][x].b + 512]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[:] c_histogram(image):
    cdef ImageInfo info
    cdef int x, y
    cdef int[:] hist = cvarray(shape=(768,), itemsize=sizeof(int), format="i")

    info = parse_img_info(image)

    for x in range(768):
        hist[x] = 0

    for y in range(info.height):
        for x in range(info.width):
            hist[info.img_ptr[y][x].r] += 1
            hist[info.img_ptr[y][x].g + 256] += 1
            hist[info.img_ptr[y][x].b + 512] += 1

    return hist


def histogram(image):
    cdef int[:] hist = c_histogram(image)
    cdef int i
    cdef int return_vals[768]

    for i in range(768):
        return_vals[i] = hist[i]

    return return_vals


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

    cdef int layer = 0
    cdef int length = 756
    cdef int* h
    cdef int i, lo, hi, ix, cut
    cdef double scale, offset
    cdef int[:] histogram
    cdef int[:] lut = cvarray(shape=(768,), itemsize=sizeof(int), format="i")

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

    _c_lut(image, lut)

    return image


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def equalize(image, mask=None):
    """
    Equalize the image histogram. This function applies a non-linear
    mapping to the input image, in order to create a uniform
    distribution of grayscale values in the output image.

    :param image: The image to equalize.
    :param mask: An optional mask.  If given, only the pixels selected by
                 the mask are included in the analysis.
    :return: An image.
    """
    # if image.mode == "P":
    #     image = image.convert("RGB")
    cdef int[:] h
    cdef int[:] lut = cvarray(shape=(768,), itemsize=sizeof(int), format="i")

    h = c_histogram(image)

    cdef int b, histo_len, histo_sum, i, n, step, num
    cdef int histo[256]
    cdef int end = len(h)

    for b in range(0, end, 256):
        histo_len = 0
        histo_sum = 0

        for i in range(256):
            num = h[b + i]
            if num > 0:
                histo[histo_len] = num
                histo_sum += num
                histo_len += 1

        if histo_len <= 1:
            # lut.extend(list(range(256)))
            for i in range(256):
                lut[b + i] = i
        else:
            step = (histo_sum - histo[histo_len - 1]) // 255
            if not step:
                # lut.extend(list(range(256)))
                for i in range(256):
                    lut[b + i] = i
            else:
                n = step // 2
                for i in range(256):
                    # lut.append(n // step)
                    lut[b + i] = n // step
                    n = n + h[i + b]

    _c_lut(image, lut)

    return image


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def posterize(image, bits):
    """
    Reduce the number of bits for each color channel.

    :param image: The image to posterize.
    :param bits: The number of bits to keep for each channel (1-8).
    :return: An image.
    """
    cdef int[:] lut = cvarray(shape=(768,), itemsize=sizeof(int), format="i")
    cdef int i, b, c_bits
    cdef unsigned char mask

    c_bits = bits

    mask = ~(2 ** (8 - c_bits) - 1)
    for b in range(0, 768, 256):
        for i in range(256):
            lut[b + i] = i & mask
    return _c_lut(image, lut)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solarize(image, threshold=128):
    """
    Invert all pixel values above a threshold.

    :param image: The image to solarize.
    :param threshold: All pixels above this greyscale level are inverted.
    :return: An image.
    """
    cdef int[:] lut = cvarray(shape=(768,), itemsize=sizeof(int), format="i")
    cdef int i, b, c_threshold
    cdef ImageInfo info

    c_threshold = threshold

    for b in range(0, 768, 256):
        for i in range(256):
            if i < c_threshold:
                lut[b + i] = i
            else:
                lut[b + i] = 255 - i

    return _c_lut(image, lut)


cdef inline int L24(PixelRGBA rgb):
    return rgb.r * 19595 + rgb.g * 38470 + rgb.b * 7471 + 0x8000


cdef inline unsigned char clip(float v):
    if v < 0.0:
        return 0
    if v >= 255.0:
        return 255

    return <unsigned char>v


@cython.boundscheck(False)
@cython.wraparound(False)
def color(image, factor):
    cdef ImageInfo info
    cdef int grey_val
    cdef float c_factor

    info = parse_img_info(image)
    c_factor = factor

    for y in range(info.height):
        for x in range(info.width):
            grey_val = L24(info.img_ptr[y][x]) >> 16

            info.img_ptr[y][x].r = clip(info.img_ptr[y][x].r * c_factor + grey_val * (1 - c_factor))
            info.img_ptr[y][x].g = clip(info.img_ptr[y][x].g * c_factor + grey_val * (1 - c_factor))
            info.img_ptr[y][x].b = clip(info.img_ptr[y][x].b * c_factor + grey_val * (1 - c_factor))

    return image


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def contrast(image, factor):
    cdef ImageInfo info
    cdef int i_mean
    cdef float c_factor, f_mean

    info = parse_img_info(image)
    c_factor = factor

    f_mean = 0
    for y in range(info.height):
        i_mean = 0
        for x in range(info.width):
            i_mean += L24(info.img_ptr[y][x]) >> 16
        i_mean /= info.width
        f_mean += i_mean
    f_mean /= info.height

    for y in range(info.height):
        for x in range(info.width):
            info.img_ptr[y][x].r = clip(info.img_ptr[y][x].r * c_factor + f_mean * (1 - c_factor))
            info.img_ptr[y][x].g = clip(info.img_ptr[y][x].g * c_factor + f_mean * (1 - c_factor))
            info.img_ptr[y][x].b = clip(info.img_ptr[y][x].b * c_factor + f_mean * (1 - c_factor))

    return image


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def brightness(image, factor):
    cdef ImageInfo info
    cdef int zero_val
    cdef float c_factor

    info = parse_img_info(image)
    c_factor = factor

    zero_val = 0
    for y in range(info.height):
        for x in range(info.width):
            info.img_ptr[y][x].r = clip(info.img_ptr[y][x].r * c_factor + zero_val * (1 - c_factor))
            info.img_ptr[y][x].g = clip(info.img_ptr[y][x].g * c_factor + zero_val * (1 - c_factor))
            info.img_ptr[y][x].b = clip(info.img_ptr[y][x].b * c_factor + zero_val * (1 - c_factor))

    return image


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sharpness(image, factor):
    cdef ImageInfo info
    cdef int x, y, i, j
    cdef double c_factor
    cdef float smooth_kernel[3][3]
    smooth_kernel[0][:] = [1 / 13., 1 / 13., 1 / 13.]
    smooth_kernel[1][:] = [1 / 13., 5 / 13., 1 / 13.]
    smooth_kernel[2][:] = [1 / 13., 1 / 13., 1 / 13.]
    cdef float r, g, b, div

    info = parse_img_info(image)
    c_factor = factor

    for y in range(1, info.height - 1):
        for x in range(1, info.width - 1):
            r = g = b = div = 0

            for i in range(-1, 2):
                for j in range(-1, 2):
                    r += smooth_kernel[i + 1][j + 1] * info.img_ptr[y + i][x + j].r
                    g += smooth_kernel[i + 1][j + 1] * info.img_ptr[y + i][x + j].g
                    b += smooth_kernel[i + 1][j + 1] * info.img_ptr[y + i][x + j].b

            info.img_ptr[y][x].r = clip(info.img_ptr[y][x].r * c_factor + r * (1 - c_factor))
            info.img_ptr[y][x].g = clip(info.img_ptr[y][x].g * c_factor + g * (1 - c_factor))
            info.img_ptr[y][x].b = clip(info.img_ptr[y][x].b * c_factor + b * (1 - c_factor))

    return image


@cython.boundscheck(False)
@cython.wraparound(False)
def to_numpy(image):
    cdef ImageInfo info
    info = parse_img_info(image)
    cdef np.ndarray[np.uint8_t, ndim=3] np_img = np.zeros([info.height, info.width, 3], dtype=np.uint8)

    for y in range(info.height):
        for x in range(info.width):
            np_img[y, x, 0] = info.img_ptr[y][x].r
            np_img[y, x, 1] = info.img_ptr[y][x].g
            np_img[y, x, 2] = info.img_ptr[y][x].b

    return np_img
