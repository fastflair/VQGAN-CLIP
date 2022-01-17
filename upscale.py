import numpy as np
import sys
from PIL import Image
from ISR.models import RRDN

fileName = sys.argv[1]
newFileName = fileName.replace(".", "_up.")

img = Image.open(fileName)
lr_img = np.array(img)

rrdn = RRDN(weights='gans')
sr_img = rrdn.predict(lr_img)
upscaledImage = Image.fromarray(sr_img)
upscaledImage.save(newFileName)