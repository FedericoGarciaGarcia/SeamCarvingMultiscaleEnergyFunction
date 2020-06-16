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
img  = imLoadRGB("example images/me.png")
mask = cv2.imread("example images/me_mask.png", 0)

# Reduction of 50%
n = int(float(img.shape[1])*0.5)

print("-- 1/3: Reduce image --")
img_sc      = SM_redu(img, n, True, False)
print("-- 2/3: Reduce with mask protection --")
img_sc_mask1 = SM_redu(img, n, True, False, mask)
print("-- 3/3: Reduce with mask removal --")
img_sc_mask2 = SM_redu(img, n, True, False, mask, False)

# Show
show([img, mask, img_sc, img_sc_mask1, img_sc_mask2], ["Original", "Mask", "Reduction", "Protection", "Elimination"], 2, 3)