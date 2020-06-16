# Seam Carving with multiscale energy function
The famous Seam Carving algorithm implemented in Python and OpenCV. A new energy function is proposed, consisting on the sum of a standard energy function on multiple lower scales, to prevent larger object deformation. The standard energy function is the absolute sum of the partial derivatives of each color channel.

You can find the original paper [here](http://graphics.cs.cmu.edu/courses/15-463/2012_fall/hw/proj3-seamcarving/imret.pdf).

## Reduction

![Reduction](https://github.com/FedericoGarciaGarcia/SeamCarvingMultiscaleEnergyFunction/blob/master/images/reduce.png)

## Enlargement

![Enlargement](https://github.com/FedericoGarciaGarcia/SeamCarvingMultiscaleEnergyFunction/blob/master/images/enlargement.png)

## Mask protection and elimination

![Mask](https://github.com/FedericoGarciaGarcia/SeamCarvingMultiscaleEnergyFunction/blob/master/images/mask.png)

## Interactive window

![Window](https://github.com/FedericoGarciaGarcia/SeamCarvingMultiscaleEnergyFunction/blob/master/images/window.png)

## How to use

Install OpenCV:

```
pip install opencv-python
```

If you also want contrib modules, run this one instead (not necessary):

```
pip install opencv-contrib-python
```

The file *seamCarving.py* includes all the necessary functions for Seam Carving.

### Examples

Four example files are provided with a folder with images. They can be run directly without arguments:

```
python code\example.py
```

After each execution, a window is shown with the results.

* **exampleEnlarge.py**: Enlarges an image horizontally by 50% using the standard energy function.
* **exampleReduce.py**: Reduces an image horizontally by 50% using the standard energy function and the multiscale energy function.
* **exampleMask.py**: Reduces an image horizontally by 50% using the standard energy function. A mask is applied to remove and to protect pixels.
* **exampleWindow.py**: Finds all the horizontal seams of an image and opens up an interactive windows with a slider. Moving the slider to the left or right allows for image resizing.

## Authors

* **Federico Garcia Garcia**