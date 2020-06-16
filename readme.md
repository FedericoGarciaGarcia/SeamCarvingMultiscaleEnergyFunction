# Seam Carving with multiscale energy function
The famous Seam Carving algorithm implemented in Python and OpenCV. A new energy function is proposed, consisting on the sum of a standard energy function on multiple lower scales, to prevent larger object deformation. The standard energy function is the absolute sum of the partial derivatives of each color channel.

## Reduction

![Reduction](https://github.com/FedericoGarciaGarcia/SeamCarvingMultiscaleEnergyFunction/blob/master/images/reduce.png)

## Enlargement

![Enlargement](https://github.com/FedericoGarciaGarcia/SeamCarvingMultiscaleEnergyFunction/blob/master/images/enlargement.png)

## Mask protection and elimination

![Mask](https://github.com/FedericoGarciaGarcia/SeamCarvingMultiscaleEnergyFunction/blob/master/images/mask.png)

## Interactive window

![Window](https://github.com/FedericoGarciaGarcia/SeamCarvingMultiscaleEnergyFunction/blob/master/images/window.png)

## How to use

### Requirements

Install OpenCV:

```
pip install opencv-python
```

If you also want contrib modules, run this one instead (not necessary):

```
pip install opencv-contrib-python
```

### Examples

Four example files are provided, with a folder with images. They can be run directly without arguments:

```
python code\example.py
```

* **exampleIncrease.py**: 

## Authors

* **Federico Garcia Garcia**

## Acknowledgments

Textures and materials taken from:
* [3D Textures](https://3dtextures.me/)
* [Texture Haven](https://texturehaven.com/textures/)