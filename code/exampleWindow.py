################################################################################################
#
# COMPUTER VISION
# 2018/2019
# SEAM CARVING
# FEDERICO GARCIA GARCIA
#
################################################################################################

from seamCarving import *

# Global variables
src = None
dst = None
seams = None
seams_pixeles = None
current = 0
title_window = ''

# Action when trackbar slider is changed
def on_trackbar(val):
    global dst
    global seams
    global seams_pixeles
    global current

    # Delta
    iterations = dst.shape[1] - val

    if(val > 0):
        # Moved to the left
        if(iterations > 0):
            # Run algorithm with the delta difference
            for i in range(iterations):
                dst = RemoveSeamImageRGB (dst, seams[current])
                current = current+1
        # Moved to the right
        else:
            # Run algorithm with the delta difference
            for i in range(-iterations):
                current = current-1
                dst = AddSeamImageRGB (dst, seams[current], seams_pixeles[current])

    # Draw borders
    border=cv2.copyMakeBorder(dst, top=0, bottom=0, left=int((src.shape[1]-dst.shape[1])/2), right=int(math.ceil((src.shape[1]-dst.shape[1])/2)), borderType=cv2.BORDER_CONSTANT, value=0)
    cv2.imshow(title_window, border)
    pass

print("-- Finding seams --")

# Variables globales
src = cv2.imread("example images/tower.png", 3)
dst = src.copy()
seams, seams_pixeles = SM_RealTime(src, True) # Obtener los seams
current = 0
title_window = 'Seam Carving - Federico Garcia Garcia'

# Crear trackbar y ventana
alpha_slider_max = src.shape[1]
cv2.namedWindow(title_window)
trackbar_name = 'Width (px)'
cv2.createTrackbar(trackbar_name, title_window, src.shape[1], alpha_slider_max, on_trackbar)
on_trackbar(src.shape[1])
cv2.waitKey()