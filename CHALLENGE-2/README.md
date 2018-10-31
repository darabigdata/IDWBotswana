<p align="center"><img width=30% src="https://github.com/darabigdata/IDWBotswana/blob/master/media/00000022.jpg"></p>


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
* [requests](docs.python-requests.org/en/master/)

------

## Simple Web-scraping and Classification Tutorial

This tutorial is a simpler version of  and is divided into three parts:

1. Obtaining a text file of the image urls that will make up the training and test datasets
2. Using python to download and perfom simple checks on the images
3. Use Scikit learn to perform image classification

### Step 1: Image Webscraping

For this challenge you're going to need to build your own library of training data. For image data one of the best places to get this kind of data is Google Images. So in this tutorial we'll use Google Images to web-scrape a database of images. The instructions for this step are a simplified version of the excellent blog post [here](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/). To follow this tutorial you'll need to use Google Chrome, but there are also many nice ways of scraping data from the web using the Python requests library and the BeautifulSoup library, e.g. [here](https://allofyourbases.com/2017/10/08/web-scraping-youtube-in-python/).

1. Open Chrome and navigate to google image search. Now enter your search, e.g. "Zebra"
2. Open the Developer console: either use CTRL+SHIFT+I or go to 'View' --> 'Developer' --> 'Javascript Console'. 
3. The next step is to start scrolling! Keep scrolling until you have found all relevant images to your query.
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
hiddenElement.download = 'zebra_urls.txt';
hiddenElement.click();
```
The last step will download a text file named: 'zebra_urls.txt'.

### Step 2: Download all the images with Python

This repo includes a [script](https://github.com/darabigdata/IDWBotswana/blob/master/CHALLENGE-2/download_images.py) to download all the images listed in the url file. You can also always write your own!

The script provided here takes two arguments: (1) the url list file, and (2) the location of the directory where you want the images to be stored. You can run it like this:

```bash
> python download_images.py -u zebra_urls.txt -o ./ZEBRA/
```

Inside the script, the first step is to grab a list of the urls from the input file:

```python
# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0
```

then to loop through each of the urls and download each one into the specified output folder:

```python
# loop the URLs and download the images
for url in rows:
	try:
		# try to download the image
		r = requests.get(url, timeout=60)
 
		# save the image to disk
		p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(8))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()
 
		# update the counter
		print("[INFO] downloaded: {}".format(p))
		total += 1
 
	# handle if any exceptions are thrown during the download process
	except:
		print("[INFO] error downloading {}...skipping".format(p))

```

Once you've got all of the images you probably want to check to see if any have been corrupted or aren't really images. So there's a check at the end that tries to open each image. If the check fails (i.e. the image can't be opened) then the image file is deleted:

```python
# open the images, if it returns an error delete the image.
images = glob.glob('{}/*.jpg'.format(args["output"]))

for image in images:
	delete = False
	try:
		im = plt.imread(image,format='jpeg')
		if im is None:
			delete = True
	except:
		print('Except')
		delete = True

	if delete:
		print('INFO deleting {}'.format(image))
		os.remove(image)		
```

For classification we're also going to need a set of images that don't contain our target class, i.e. images that are NOT of zebras. A good online database for random images is the [UKBench Dataset](https://archive.org/details/ukbench), it contains 10,000 images. You can build a "not zebra" dataset by randomly sampling images from there. Remember to randomly sample approximately the same number of "not zebra" images as "zebra" images, otherwise you'll end up with a [class imbalance problem](https://towardsdatascience.com/dealing-with-imbalanced-classes-in-machine-learning-d43d6fa19d2). 

### Step 3: Image classification with sklearn.

There are many different approaches to image classification. One heavily used method is Convolutional Neural Networks (CNNs) and there's a good example of how to implement a CNN using the [keras library](https://keras.io/) in this blog.

In this repo we've provided an example [classifier](https://github.com/darabigdata/IDWBotswana/blob/master/CHALLENGE-2/Classifying_Zebra_Images.ipynb) that uses a combination of [Gabor filters](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html) and [Support Vector Machines (SVMs)](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72). 

The purpose of the Gabor filter is to extract machine learning **features** 

```python
# first we will define a function that will use Gabor filters to reduce the images to a constant set of features
# define Gabor features
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        #feats[k, 0] = filtered.mean()
        #feats[k, 1] = filtered.var()
        feats[k, 0] = kurtosis(np.reshape(filtered,-1))
        feats[k, 1] = skew(np.reshape(filtered,-1))
    return feats
```

```python
# prepare Gabor filter bank kernels
kernels = []
for sigma in (1,4):
    theta = np.pi
    for frequency in (0.05, 0.25):
        print('theta = {}, sigma = {} frequency = {}'.format(theta, sigma, frequency) )
        kernel = np.real(gabor_kernel(frequency,theta=theta,sigma_x=sigma, sigma_y=sigma))
        kernels.append(kernel)
                         
np.shape(kernels)
```

```python
zebra_feats = np.zeros((len(zebra_images),9))
for i, image in enumerate(zebra_images):
    im = plt.imread(image,format='jpeg')
    if len(im.shape) > 2:
        imean = im.mean(axis=2)
    else:
        imean = im
    imfeats = compute_feats(imean,kernels).reshape(-1)
    zebra_feats[i,:-1] = imfeats 
    zebra_feats[i,-1] = 1
```

```python
nozebra_feats = np.zeros((len(nozebra_images),9))
for i, image in enumerate(nozebra_images):
    im = plt.imread(image,format='jpeg')
    imfeats = compute_feats(im.mean(axis=2),kernels).reshape(-1)
    nozebra_feats[i,:-1] = imfeats 
    nozebra_feats[i,-1] = 0
```

```python
#combine the datasets
ds = np.concatenate((nosanta_feats,santa_feats), axis=0)
features = ds[:,:-1]
```

```python
features = MaxAbsScaler().fit_transform(features)
```

```python
target = ds[:,-1]
```

```python
x_train, x_test, y_train, y_test = train_test_split(features,target)

print('Training data and target sizes: \n{}, {}'.format(x_train.shape,y_train.shape))
print('Test data and target sizes: \n{}, {}'.format(x_test.shape,y_test.shape))
```

```python
# Create a classifier: a support vector classifier
classifier = svm.SVC(C=1,kernel='rbf',gamma=1)

# fit to the training data
classifier.fit(x_train,y_train)
```

```python
# now predict the value of the digit on the test data
y_pred = classifier.predict(x_test)
```

```python
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
```

```python
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))
```
