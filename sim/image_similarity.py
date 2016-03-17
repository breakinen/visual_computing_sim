import os, time, math, operator
from numpy import average, linalg, dot
from PIL import Image
import numpy
import functools
import logging

logging.basicConfig(level=logging.DEBUG, format=format)
logger = logging.getLogger(__name__)

img_cache_dict = {}
img_gray_cache_dict = {}
img_vector_cache_dict = {}
img_norm_cache_dict = {}


def main():
    begin_similarty_compare('photos')

# Test computing similarity of two images
def begin_similarty_compare(photo_directory):
    image_filepath1 = '/tue/Q3_VISUAL_COMPUTING/similarity/photos/GED0000.JPG'
    image_filepath2 = '/tue/Q3_VISUAL_COMPUTING/similarity/photos/GED0001.JPG'

    t1 = time.time()

    similarity = histogram_similarity(image_filepath1, image_filepath2)

    duration = "%0.1f" % ((time.time() - t1) * 1000)
    logger.debug("Histogram distance => %s took %s ms" % (similarity, duration))

    t1 = time.time()

    similarity = histogram_similarity(image_filepath1, image_filepath2)

    duration = "%0.1f" % ((time.time() - t1) * 1000)
    logger.debug("Histogram distance => %s took %s ms" % (similarity, duration))

    t1 = time.time()
    similarity = pixel_cosine_similarity(image_filepath1, image_filepath2)
    duration = "%0.1f" % ((time.time() - t1) * 1000)
    logger.debug("Cosine distance %s took %s ms" % (similarity, duration))

    t1 = time.time()
    similarity = pixel_cosine_similarity(image_filepath1, image_filepath2)
    duration = "%0.1f" % ((time.time() - t1) * 1000)
    logger.debug("Cosine distance %s took %s ms" % (similarity, duration))


def histogram_similarity(p1_path, p2_path):
    resize_size = (50, 50)

    if p1_path in img_cache_dict:
        imgA = img_cache_dict[p1_path]
    else:
        imgA = Image.open(p1_path)
        imgA = imgA.resize(resize_size)
        img_cache_dict[p1_path] = imgA
        print("adding to cache: " + p1_path)

    if p2_path in img_cache_dict:
        imgB = img_cache_dict[p2_path]
    else:
        imgB = Image.open(p2_path)
        imgB = imgB.resize(resize_size)
        img_cache_dict[p2_path] = imgB
        print("adding to cache: " + p2_path)

    h1 = imgA.histogram()
    h2 = imgB.histogram()

    rms = 1-math.sqrt(functools.reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, h1, h2))) / len(h1))
    return rms


def pixel_cosine_similarity(p1_path, p2_path):
    resize_size = (50, 50)

    if p1_path in img_vector_cache_dict:
        imgA_vector = img_vector_cache_dict[p1_path]
        imgA_norm = img_norm_cache_dict[p1_path]
    else:
        imgA = Image.open(p1_path)
        imgA = imgA.resize(resize_size)
        img_cache_dict[p1_path] = imgA
        imgA_vector = []
        for pixel_tuple in imgA.getdata():
            imgA_vector.append(pixel_tuple[0])
            imgA_vector.append(pixel_tuple[1])
            imgA_vector.append(pixel_tuple[2])
        img_vector_cache_dict[p1_path] = imgA_vector
        imgA_norm = linalg.norm(imgA_vector, 2)
        img_norm_cache_dict[p1_path] = imgA_norm
        print("adding to cache: " + p1_path)
        print(len(img_vector_cache_dict))

    if p2_path in img_vector_cache_dict:
        imgB_vector = img_vector_cache_dict[p2_path]
        imgB_norm = img_norm_cache_dict[p2_path]
    else:
        imgB = Image.open(p2_path)
        imgB = imgB.resize(resize_size)
        img_cache_dict[p2_path] = imgB
        imgB_vector = []
        for pixel_tuple in imgB.getdata():
            imgB_vector.append(pixel_tuple[0])
            imgB_vector.append(pixel_tuple[1])
            imgB_vector.append(pixel_tuple[2])
        img_vector_cache_dict[p2_path] = imgB_vector
        imgB_norm = linalg.norm(imgB_vector, 2)
        img_norm_cache_dict[p2_path] = imgB_norm
        print("adding to cache: " + p2_path)
        print(len(img_vector_cache_dict))

    res = dot(imgA_vector / imgA_norm, imgB_vector / imgB_norm)
    return res

def ghc_similarity(p1_path, p2_path):
    resize_size = (50, 50)

    if p1_path in img_gray_cache_dict:
        imgA = img_gray_cache_dict[p1_path]
    else:
        imgA = Image.open(p1_path)
        imgA = imgA.resize(resize_size)
        imgA = imgA.convert("L")
        img_gray_cache_dict[p1_path] = imgA
        print("adding to cache: " + p1_path)

    if p2_path in img_gray_cache_dict:
        imgB = img_gray_cache_dict[p2_path]
    else:
        imgB = Image.open(p2_path)
        imgB = imgB.resize(resize_size)
        imgB = imgB.convert("L")
        img_gray_cache_dict[p2_path] = imgB
        print("adding to cache: " + p2_path)

    code1 = img_hashcode(imgA)
    code2 = img_hashcode(imgB)
    res = hamming_distance(code1,code2)
    return res

def img_hashcode(image):
    pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join(map(lambda pixel: '1' if pixel < avg else '0', pixels))  # '00010100...'
    hexadecimal = int(bits, 2).__format__('016x').upper()
    return hexadecimal

def hamming_distance(s1, s2):
    len1, len2= len(s1),len(s2)
    if len1!=len2:
        "hamming distance works only for string of the same length, so i'll chop the longest sequence"
        if len1>len2:
            s1=s1[:-(len1-len2)]
        else:
            s2=s2[:-(len2-len1)]
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


if __name__ == "__main__":
    main()
