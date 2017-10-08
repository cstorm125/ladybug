# Girl-Ladyboy Detection with VGG-Face

We perform transfer learning using the pre-trained VGG-Face model to classify girls and ladyboys of East Asian origins, with a lot of hints on the implementation side from [@m.zaradzki](https://github.com/mzaradzki) from [Above Intelligence](https://aboveintelligent.com/tagged/artificial-intelligence). 

The ladyboy examples are scraped from [My Lady Boy Date](https://myladyboydate.com/) and the girl examples are from [Date in Asia](https://www.dateinasia.com/). The reason for using different sources is that in practice general-purpose dating sites have very limited number of ladyboy profiles. After that, we ran a haarcascade classifier to detect frontal faces and ended up with the following datasets:

* Train: 3,151 girls / 2,210 ladyboys
* Valid: 900 girls / 631 ladyboys
* Test: 450 girls / 316 ladyboys

For more information see the [Medium article]().

## ledyba

Python package for girl-ladyboy detection using the model detailed in this project.

### Installation

```
pip install ledyba
```

### Example

```
import ledyba

#frontal face with dimension of (224,224,3)
crpim = detect_face(image_file)
ledyba.predict_gender(crpim)

#Output
#(0, 'girl', 0.92305273)
```

## notebook

### capture_face.ipynb

Capture faces from scraped profile pictures using [OpenCV](http://opencv.org)

### modeling.ipynb

Transfer learning using [VGG-Face](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf). The fully connected layers are replaced with two FC-BN-DO layers and a final sigmoid layer, which are trained for 10 epochs. It achieved the accuracy of 94.13%, F-score of 94% and AUC of 0.934.

## Rmd

Exploratory data analysis on the age and country of origin of the datasets

## Scraper

R scripts to scrape the girl and ladyboy pictures.

## R

R scripts to download the pictures from the scraped URLs

## References
* Deep face recognition, O. M. Parkhi and A. Vedaldi and A. Zisserman, Proceedings of the British Machine Vision Conference (BMVC), 2015 [paper](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf).
* Zaradzki, M. “Face Recognition with Keras and OpenCV – Above Intelligent (AI).” Above Intelligent (AI), Above Intelligent (AI), 6 Mar. 2017, [Medium post](aboveintelligent.com/face-recognition-with-keras-and-opencv-2baf2a83b799)
* [Practical Deep Learning For Coders, Part 1](http://course.fast.ai)

## What We Can Do Better

* More pictures, preferrably from the same data source with similar image quality
* Ensembling and training for more epochs
* An app to delivery the model

