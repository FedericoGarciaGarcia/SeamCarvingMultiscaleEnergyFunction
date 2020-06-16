################################################################################################
#
# COMPUTER VISION
# 2018/2019
# SEAM CARVING
# FEDERICO GARCIA GARCIA
#
################################################################################################

from seamCarving import *

# Load image
img = imLoadRGB("example images/butterfly.png")

# Reduction of 50% in both axis with standard and new energy functions
nx = int(float(img.shape[1])*0.5)
ny = int(float(img.shape[0])*0.5)

print("-- 1/4: Standard energy function X reduction --")
img_sc = SM_redu(img,    nx, True, False)
print("-- 2/4: Standard energy function Y reduction --")
img_sc = SM_redu(img_sc, ny, True, True)

print("-- 3/4: New energy function X reduction --")
img_sc_alt = SM_redu(img,        nx, False, False)
print("-- 4/4: New energy function Y reduction --")
img_sc_alt = SM_redu(img_sc_alt, ny, False, True)

# Show
show([img, img_sc, img_sc_alt], ["Original", "Standard energy", "New energy"], 1, 3)
