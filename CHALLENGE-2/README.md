# Challenge 2: Google Image Web-scraping and Classification

In this challenge you will learn how to web-scrape images from Google and use them to train/test a Machine Leaning (ML) model. The aim is to come up with a image classification problem (cats vs dogs, people vs trees, Trump vs an orange cheeto etc), web-scrape the images and then use ML for the classification.

### What's in the repo?

* **Simple_Movie_Recommender.ipynb**
    * *A jupyter notebook that implements simple machine learn to recommend movies based on a model trained on an input database*
* **tmdb_5000_movies.csv**
    * *An input database of 5000 movies [from Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata)*
* **tmdb_movies_clean.csv**
    * *An output cleaned database of movies*
* **movie_indices.pkl**
    * *An output pickle file of the machine learning model*
* **app.py**
    * *Code to make a simple Flask application*
* **templates**
    * *A directory of Flask HTML templates*
* **Procfile**
    * *A Heroku Procfile to turn the Flask app into a web application*
* **requirements.txt**
    * *A Heroku requirements file to turn the Flask app into a web application*

### Dependencies

* [matplotlib](https://matplotlib.org/)
* [numpy](www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [sklearn](scikit-learn.org/)
* [sklearn-image](https://scikit-image.org/)

------

## Simple Web-scraping and Classification Tutorial

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
