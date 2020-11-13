# TITLE: Reformulate tiff images

## a. Functions

from PIL import Image, ImageFile, TiffImagePlugin
from matplotlib import pyplot as plt
import numpy as np
import math

TiffImagePlugin.READ_LIBTIFF = True

def ConvertUsableTiff(depthpath, stencilpath):
    DepthToUsableDepth(depthpath)                                                             # Depth
    StencilToUsableStencil(stencilpath)                                                       # Stencil

## b. Apply function on command line

import os

depth = os.sys.argv[1]
stencil = os.sys.argv[2]

ConvertUsableTiff(depth, stencil)

from pathlib import Path
rootdir = Path("D:/ImageDB/Collision")

[DepthToUsableDepth(d) for d in rootdir.resolve().glob('**/*/*depth.tif') if d.stat().st_size < 1000000]
[StencilToUsableStencil(s) for s in rootdir.resolve().glob('**/*/*stencil.tif') if s.stat().st_size < 1000000]


[TifToJpeg(str(d)) for d in rootdir.resolve().glob('**/*/*colour.tif')]

