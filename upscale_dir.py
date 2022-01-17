import numpy as np
import sys
import os
from PIL import Image
from ISR.models import RRDN

extensions = ('.png', '.PNG', '.jpg', '.JPG')

rrdn = RRDN(weights='gans')

dirName = sys.argv[1]
for root, _, files in os.walk(dirName):
    for fileName in files:
        if fileName.endswith(extensions):
            newFileName = fileName.replace(".", "_up.")
            img = Image.open(dirName+"\\"+fileName)
            lr_img = np.array(img)
            sr_img = rrdn.predict(lr_img)
            upscaledImage = Image.fromarray(sr_img)
            upscaledImage.save(dirName+"\\"+newFileName)