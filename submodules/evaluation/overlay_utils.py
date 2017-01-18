#!/usr/bin/env python

"""Utility functions for segmentation tasks."""

from PIL import Image
import scipy.ndimage
import numpy as np


def replace_colors(segmentation, color_changes):
    """
    Replace the values in segmentation to the values defined in color_changes.

    Parameters
    ----------
    segmentation : numpy array
        Two dimensional
    color_changes : dict
        The key is the original color, the value is the color to change to.
        The key 'default' is used when the color is not in the dict.
        If default is not defined, no replacement is done.
        Each color has to be a tuple (r, g, b) with r, g, b in {0, 1, ..., 255}
    Returns
    -------
    np.array
        The new colored segmentation
    """
    width, height = segmentation.shape
    output = scipy.misc.toimage(segmentation)
    output = output.convert('RGBA')
    for x in range(0, width):
        for y in range(0, height):
            if segmentation[x, y] in color_changes:
                output.putpixel((y, x), color_changes[segmentation[x, y]])
            elif 'default' in color_changes:
                output.putpixel((y, x), color_changes['default'])
    return output


def overlay_segmentation(image, segmentation, color_dict):
    """
    Overlay original_image with segmentation_image.

    Parameters
    ----------
    """
    width, height = segmentation.shape
    output = scipy.misc.toimage(segmentation)
    output = output.convert('RGBA')
    for x in range(0, width):
        for y in range(0, height):
            if segmentation[x, y] in color_dict:
                output.putpixel((y, x), color_dict[segmentation[x, y]])
            elif 'default' in color_dict:
                output.putpixel((y, x), color_dict['default'])

    background = scipy.misc.toimage(image)
    background.paste(output, box=None, mask=output)

    return np.array(background)
