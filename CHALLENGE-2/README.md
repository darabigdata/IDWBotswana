
# Google Image webscraping and Classification

This tutorial is a simpler version of https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/ and is divided into three parts:

1. Obtaining a text file of the image urls that will make up the training and test datasets
2. Using python to download and perfom simple checks on the images
3. Use Scikit learn to perform image classification

### Step 1: Image Webscraping
In this example I would like to find out if I can build a simple ML model to tell the difference between Images of Santa Claus vs Christmas images without Santa Claus. The first thing we will need to do is perfom webscraping using Google Chrome. 

1. Open Chrome and navigate to google image search. Now enter your search, e.g Santa Claus
2. Open the Developer console: either use CTRL+SHIFT+I or go to 'More tools', then 'Developer tools' 
3. The next step is to start scrolling! Keep scrolling until you have found all relevant images to your query
4. Next is to grab all the urls of the images in your scroll. In the console enter the following commands:

```javascript
// pull down jquery into the JavaScript console
var script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);
```
```javascript
// grab the URLs
var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });
```

```javascript
// write the URls to file (one per line)
var textToSave = urls.toArray().join('\n');
var hiddenElement = document.createElement('a');
hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
hiddenElement.target = '_blank';
hiddenElement.download = 'urls.txt';
hiddenElement.click();
```
The last step will download 'urls.txt'

### Step 2: Download the image with Python

I have provided a [script](https://github.com/hrampadarath/JBCA_Hack_Night_Dec/blob/master/google_images_webscraping/download_images.py) to download the images. It is a Frankenstein of the codes from the [pyimager webpage](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/) and my own that does not require opencv! You can write your oun or use the one provided.


### Step 3: Image classification with sklearn.

An example [classifier](https://github.com/hrampadarath/JBCA_Hack_Night_Dec/blob/master/google_images_webscraping/Classifying_Santa_Clause_Images.ipynb) has been added to this repository that uses a combination of Gabor filters and Support Vector Machines (SVMs). 
