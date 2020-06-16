################################################################################################
#
# COMPUTER VISION
# 2018/2019
# SEAM CARVING
# FEDERICO GARCIA GARCIA
#
################################################################################################

from seamCarving import *

# Load images
img  = imLoadRGB("example images/butterfly.png")

# Increase by 50%
n = int(float(img.shape[1])*0.5)

print("-- Increasing with standard energy function --")
img2_sc = SM_inc(img, n, True,  False)

# Show
show([img, img2_sc], ["Original", "Standard energy"], 1, 2)