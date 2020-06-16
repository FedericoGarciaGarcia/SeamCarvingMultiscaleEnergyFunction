################################################################################################
#
# COMPUTER VISION
# 2018/2019
# SEAM CARVING
# FEDERICO GARCIA GARCIA
#
################################################################################################

################################################################################################
# LIBRARIES

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

######################################
# AUXILIARY FUNCTIONS
######################################

# Convert gray floating point image to integer image.
# Negative values are allowed
#
#   img: floting point matrix image.
#
#   return: integer image in the range [0, 255]
#
def Norm(img):
    
    # Hallamos los valores maximos y minimos
    min_value = min(map(min,img))
    max_value = max(map(max,img))
    
    # Image size
    h = img.shape[0] # Height
    w = img.shape[1] # Width
    
    # Destination image
    img_new = np.zeros((h, w), np.uint8)
    
    # Loop where the conversion takes place
    for i in range (0, h):
        for j in range (0, w):
            img_new[i, j] = int(255.0*(img[i, j]-min_value)/(max_value-min_value));
    
    # Return new image
    return img_new
            
    pass

# Load RGB image
#
#   path: file path of image
#
#   return: RGB image
#
def imLoadRGB(path):
    img = cv2.imread(path, 3)
    return cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

# Draws images in a window, each with a title
#
#   vim: list of images
#   rows: number of rows
#   cols: number of columns
#   titles: list of strings
#
def show(vim, titles, rows, cols):

    # Number of images
    n = len(vim)

    # Show each image
    for i in range(0, n):
        plt.subplot(rows, cols, i+1)

        # Hide axis'
        plt.xticks([]), plt.yticks([])
        
        # If image has 3 dimensions, it's RGB
        if(len(vim[i].shape) == 3):
            # Show RGB image
            plt.imshow(vim[i])
        # Otherwise, it's gray
        else:
            # Convert gray image to RGB and show
            plt.imshow(cv2.cvtColor(src=vim[i], code=cv2.COLOR_GRAY2RGB))
            
        # Put title
        plt.title(titles[i])

    # Try and show in fullscreen
    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    except:
        # Do nothing
        pass

    # Show window
    plt.show();

    pass

# Standard energy function.
# Absolute sum of the partial derivatives of each color channel.
#
#   src: source image
#
#   return: energy image
#
def EnergyRGB(src):
    # X derivative
    kx, ky = cv2.getDerivKernels(dx=1, dy=0, ksize=3)
    img_dx = cv2.sepFilter2D(src=src*1.0, ddepth=-1, kernelX=kx, kernelY=ky)
    
    # Y derivative
    kx, ky = cv2.getDerivKernels(dx=0, dy=1, ksize=3)
    img_dy = cv2.sepFilter2D(src=src*1.0, ddepth=-1, kernelX=kx, kernelY=ky)
    
    # Absolute sum for each channel
    img_e =         abs(img_dx[:, :, 0]) + abs(img_dy[:, :, 0])
    img_e = img_e + abs(img_dx[:, :, 1]) + abs(img_dy[:, :, 1])
    img_e = img_e + abs(img_dx[:, :, 2]) + abs(img_dy[:, :, 2])
    
    return img_e

# New energy function.
# Absolut sum of the partial derivatives of each color channel in different scales;
# the smaller the scale, the less weight.
#
#   src: source image
#
#   return: energy image
#
def EnergyRGB_New(src):
    # Energies at different scales
    img_e1 = EnergyRGB(src)
    img_e2 = cv2.pyrUp(EnergyRGB(cv2.pyrDown(src)))
    img_e3 = cv2.pyrUp(cv2.pyrUp(EnergyRGB(cv2.pyrDown(cv2.pyrDown(src)))))
    img_e4 = cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(EnergyRGB(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(src)))))))

    # Resize images to original size
    img_e2 = cv2.resize(img_e2, dsize=(src.shape[1], src.shape[0]), interpolation = cv2.INTER_CUBIC)
    img_e3 = cv2.resize(img_e3, dsize=(src.shape[1], src.shape[0]), interpolation = cv2.INTER_CUBIC)
    img_e4 = cv2.resize(img_e3, dsize=(src.shape[1], src.shape[0]), interpolation = cv2.INTER_CUBIC)

    # Sum of images
    img_e = (img_e1+img_e2/2+img_e3/3+img_e4/4)/(1+1/2+1/3)
    
    return img_e
    
# Paint a seam over an image
#
#   src: source image
#   seam: list that defines the seam
#   color: color of the seam. Red by default
#
#   return: image with seam painted over it
#
def PaintSeam(src, seam, color=(255, 0, 0)):
    dst = src.copy()
    
    for i in range (1, src.shape[0]):
        dst[i, seam[i]] = color
        
    return dst

# Paint multiple seams over an image
#
#   src: source image
#   seams: list of seams
#   indices: the indices of the seams list to paint
#   color: color of the seam. Red by default
#
#   return: image with seams painted over it
#
def PaintSeams(src, seams, indices, color=(255, 0, 0)):
    dst = src.copy()
    
    for i in indices:
        for y in range (1, src.shape[0]):
            dst[y, seams[i][y]] = color
        
    return dst

# Given an minimum energy image and a column position where the minimum
# value is in the last row, finds the minimum energy seam
#
#   img_seams: minimum energy image
#   jj: column position where the seam starts
#
#   return: seam (list of positions)
#
def GetSeam(img_seams, jj):
    # A seam is a list with the size of the height/width.
    # It stores rows or columns.
    seam = [0] * img_seams.shape[0]
    
    j = jj

    for i_inv in range (0, img_seams.shape[0]):
        # Inverted index (we start from the bottom)
        i = img_seams.shape[0]-i_inv-1
        
        # Change of direction
        d = 0
        
        # Only from the second row
        if i > 0:
            # Min value is of the same column, for now
            min_val = img_seams[i-1, j]
        
            # Check if left value is smaller
            if j > 0: # Careful with borders
                if img_seams[i, j-1] < min_val:
                    d = -1
                    min_val = img_seams[i, j-1]
                    
            # Check if right value is smaller
            if j < img_seams.shape[1]-1: # Careful with borders
                if img_seams[i, j+1] < min_val:
                    d = 1
                    min_val = img_seams[i, j+1]
                    
            # Update change of direction
            j = j+d
                    
        # We reached the first column. Use that value
        else:
            seam[i] = j # Redundant, but just to be clear
                
        # Assign position
        seam[i] = j
        
    return seam

# Get the minimum energy image
#
#   img_e: energy image
#
#   return: minimum energy image
#
def GetMinimumEnergyImage(img_e):
    
    # Create empty image with the same size as the original
    img_seams = np.zeros(img_e.shape, dtype=float)

    # First row has the original values
    img_seams[0, :] = img_e[0, :]
        
    # From the second row to the last
    for i in range (1, img_e.shape[0]):
        for j in range (0, img_e.shape[1]):
            # Careful with borders
            if(j == 0):
                img_seams[i, j] = img_e[i, j] + min(
                        img_seams[i-1, j],
                        img_seams[i-1, j+1])
            elif(j == img_e.shape[1]-1):
                img_seams[i, j] = img_e[i, j] + min(
                        img_seams[i-1, j-1],
                        img_seams[i-1, j])
            else:
                img_seams[i, j] = img_e[i, j] + min(
                        img_seams[i-1, j-1],
                        img_seams[i-1, j],
                        img_seams[i-1, j+1])

    return img_seams

# Get desired number of seams
#
#   img_e: energy image
#   n_seams: number of seams to return
#
#   return: set of seams
#
def GetSeamMin(img_e, n_seams=1):
    
    # Get minimum energy image
    img_seams = GetMinimumEnergyImage(img_e)

    # Ordered indices
    indices = np.argsort(img_seams[img_seams.shape[0]-1, :])
    
    # Seams
    seams = [None] * n_seams

    for i in range(n_seams):

        # The minimum
        min_pos = indices[i]
        
        # Minimum seam
        seams[i] = GetSeam(img_seams, min_pos)

    return seams

# Remove seam from a single channel image
#
#   src: image to remove seam from
#   seam: seam to use for removal
#   t: image format type. uint8 by default
#
#   return: reduced image
#
def RemoveSeamImageGray(src, seam, t=np.uint8):
    
    # Image with one less column
    dst = np.zeros((src.shape[0], src.shape[1]-1), t)
    
    # Copy every row, excluding the seam's column
    for i in range (0, len(seam)):

        # Before the seam
        dst[i:i+1, 0:seam[i]] = src[i:i+1, 0:seam[i]]
 
        # After the seam
        dst[i:i+1, seam[i]:dst.shape[1]] = src[i:i+1, seam[i]+1:src.shape[1]]
    
    return dst

# Remove seam from an RGB image
#
#   src: image to remove seam from
#   seam: seam to use for removal
#
#   return: reduced image
#
def RemoveSeamImageRGB(src, seam, t=np.uint8):
    
    # Image with one less column
    dst = np.zeros((src.shape[0], src.shape[1]-1, src.shape[2]), t)
    
    # Run for every channel
    for i in range (0, 3):
        dst[:, :, i] = RemoveSeamImageGray(src[:, :, i], seam)
    
    return dst

# Insert seam in a single channel image. Pixels between the seam are averaged
#
#   src: image to insert seam to
#   seam: seam to use for removal
#
#   return: increased image
#
def AddSeamImageGray(src, seam, seam_pixel=None, t=np.uint8):
    
    # Image with an extra column
    dst = np.zeros((src.shape[0], src.shape[1]+1), t)
    
    # Copy every row
    for i in range (0, len(seam)):

        # In the seam (average of pixels)
        # Careful with borders
        if(seam_pixel == None):
            # Before the seam
            dst[i:i+1, 0:seam[i]] = src[i:i+1, 0:seam[i]]
     
            # After the seam
            dst[i:i+1, seam[i]+1:dst.shape[1]] = src[i:i+1, seam[i]:src.shape[1]]

            # The seam
            if(seam[i] == dst.shape[1]):
                dst[i, seam[i]] = np.uint8((np.uint32(src[i, seam[i]])+np.uint32(dst[i, seam[i]-1]))/2)
            elif(seam[i] == 0):
                dst[i, seam[i]] = np.uint8((np.uint32(src[i, seam[i]])+np.uint32(dst[i, seam[i]+1]))/2)
            else:
                dst[i, seam[i]] = np.uint8((np.uint32(src[i, seam[i]])+np.uint32(dst[i, seam[i]-1])+np.uint32(dst[i, seam[i]+1]))/3)
        else:
            # Before the seam
            dst[i:i+1, 0:seam[i]] = src[i:i+1, 0:seam[i]]
     
            # After the seam
            dst[i:i+1, seam[i]+1:dst.shape[1]] = src[i:i+1, seam[i]:src.shape[1]]

            # The seam
            dst[i, seam[i]] = seam_pixel[i]

    return dst

# Insert seam in a RGB image
#
#   src: image to insert seam to
#   seam: seam to use for removal
#
#   return: increased image
#
def AddSeamImageRGB(src, seam, seam_pixel=None, tipo=np.uint8):
    
    # Image with an extra column
    dst = np.zeros((src.shape[0], src.shape[1]+1, src.shape[2]), tipo)
    
    # Run for every channel
    for i in range (0, 3):
        if(seam_pixel == None):
            dst[:, :, i] = AddSeamImageGray(src[:, :, i], seam)
        else:
            dst[:, :, i] = AddSeamImageGray(src[:, :, i], seam, [j[i] for j in seam_pixel])
    
    return dst

# Seam Carving for reducing size
#
#   src: RGB image to remove seam from
#   n: minimum number of seams
#   classic: true =standard energy function
#            false=new energy function
#   horizontal: true =rotate to use horizontal seams
#               false=no rotar
#   mask: mask image to protect or remove pixels
#   mask_mode: true =protect pixeles
#              false=remove pixeles
#
#   return: reduced image
#
def SM_redu(src, n, classic, horizontal, mask=None, mask_mode=True):
    # Copy
    dst = src.copy()

    if(not(mask is None)):
        m = mask.copy()

    # Rotate image if requested
    if(horizontal):
        dst = np.rot90(dst)

        if(not(mask is None)):
            m = np.rot90(m)
    
    # Loop
    i=0
    while i < n:
        
        # Energy image
        if(classic):
            img_e = EnergyRGB(dst)
        else:
            img_e = EnergyRGB_New(dst)

        # Apply mask if requested
        if(not(mask is None)):
            img_e = GetEnergyByMask(img_e, m, mask_mode)

        # Minimum energy seam
        seam = GetSeamMin(img_e)[0]
        
        # Reduce image
        dst = RemoveSeamImageRGB (dst, seam)

        # Reduce mask
        if(not(mask is None)):
            m = RemoveSeamImageGray (m, seam)
        
        # Log
        print("Seam " +str(i+1)+ "/" +str(n))
        
        i = i+1

    # Rotate back if necessary
    if(horizontal):
        dst = np.rot90(dst, -1)
    
    return dst

# Seam Carving for increasing size
#
#   src: RGB image to insert seam to
#   n: minimum number of seams
#   classic: true =standard energy function
#            false=new energy function
#   horizontal: true =rotate to use horizontal seams
#               false=no rotar
#   mask: mask image to protect or remove pixels
#   mask_mode: true =protect pixeles
#              false=remove pixeles
#
#   return: reduced image
#
def SM_inc(src, n, classic, horizontal, mask=None, mask_modo=True):

    # Copy
    dst  = src.copy()
    src2 = src.copy()

    if(not(mask is None)):
        m = mask.copy()

    # Rotate image if requested
    if(horizontal):
        dst  = np.rot90(dst)
        src2 = np.rot90(src2)

        if(not(mask is None)):
            m = np.rot90(m)
    
    # Seams minimos
    seams = [None] * n

    # Loop to get the n seams
    i=0
    while i < n:
        
        # Energy image
        if(classic):
            img_e = EnergyRGB(src2)
        else:
            img_e = EnergyRGB_New(src2)

        # Apply mask if requested
        if(not(mask is None)):
            img_e = GetEnergyByMask(img_e, m, mask_modo)

        # Minimum energy seam
        seams[i] = GetSeamMin(img_e)[0]
        
        # Reduce image
        src2 = RemoveSeamImageRGB (src2, seams[i])

        # Reduce mask
        if(not(mask is None)):
            m = RemoveSeamImageGray (m, seams[i])
                
        print("Seam " +str(i+1)+ "/" +str(n))
        
        i = i+1

    # Now that we have the seams, insert
    i=0
    while i < n:
        
        # Increase image
        dst = AddSeamImageRGB (dst, seams[i])

        for j in range(n):
            for y in range(len(seams[i])):
                if j != i:
                    if seams[j][y] >= seams[i][y]:
                        seams[j][y] += 2
        
        i = i+1

    # Rotate back if necessary
    if(horizontal):
        dst = np.rot90(dst, -1)
    
    return dst

# Given an image and a seam, get the pixels
#
#   src: image source
#   seam: list of positions
#
#   return: list of pixels
#
def GetPixelsSeam(src, seam):
    seam_pixeles = [None] * len(seam)

    for i in range(len(seam)):
        seam_pixeles[i] = src[i, seam[i]]

    return seam_pixeles

# Seam Carving in real time
#
#   src: image source
#   classic: true =standard energy function
#            false=new energy function
#
#   return: list of seams and list of pixels
#
def SM_RealTime(src, clasico):

    # Copy
    dst = src.copy()
    
    # Number of seams
    n = src.shape[1]-1

    # Seams
    seams = [None] * n
    
    # Pixels
    seams_pixeles = [None] * n

    # Loop to get seams
    i=0
    while i < n:
        
        # Energy image
        if(clasico):
            img_e = EnergyRGB(dst)
        else:
            img_e = EnergyRGB_New(dst)

        # Minimum energy seam
        seams[i] = GetSeamMin(img_e)[0]

        # Pixels of the minimum energy seam
        seams_pixeles[i] = GetPixelsSeam(dst, seams[i])
        
        # Reduce image
        dst = RemoveSeamImageRGB (dst, seams[i])
                
        print("Seam " +str(i+1)+ "/" +str(n))

        i = i+1

    return seams, seams_pixeles

# Change energy image's energy given a mask
#
#   src: image source
#   mask: gray image
#   mode: true =protect pixels
#         false=remove pixels
#
#   return: energy image with modified energy
#
def GetEnergyByMask(src, mask, modo):

    if(modo == True):
        dst = np.where(mask==0, src, 100000)
    else:
        dst = np.where(mask==0, src, -100000)

    return dst