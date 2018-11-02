<p align="center"><img width=30% src="https://github.com/darabigdata/IDWBotswana/blob/master/media/00000022.jpg"></p>


# Challenge 2: Google Image Web-scraping and Classification

In this challenge you will learn how to web-scrape images from Google and use them to train/test a Machine Leaning (ML) model. The aim is to come up with a image classification problem (cats vs dogs, people vs trees, Trump vs an orange cheeto etc), web-scrape the images and then use ML for the classification.

### What's in the repo?

* **Classify_Zebra_Images.ipynb**
    * *A jupyter notebook that implements simple machine learning to identify images of zebras*
* **zebra_urls.txt**
    * *A list of urls of pictures of zebras from Google Image Search*
* **download_images.py**
    * *Code to download images from a list of urls*
* **get_random_images.py**
    * *Code to randomly select pictures from the [Caltech-256 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)*
* **zebra**
    * *A directory of ~400 pictures that contain zebras*
* **nozebra**
    * *A directory of ~400 pictures that don't contain zebras*

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

For classification we're also going to need a set of images that don't contain our target class, i.e. images that are NOT of zebras. A good online database for random images is the [Caltech-256 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/), it contains about 30,000 images grouped into categories. You can build a "not zebra" dataset by randomly sampling images from there (make sure you avoid images from the zebra category!). Remember to randomly sample approximately the same number of "not zebra" images as "zebra" images, otherwise you'll end up with a [class imbalance problem](https://towardsdatascience.com/dealing-with-imbalanced-classes-in-machine-learning-d43d6fa19d2). 

### Step 3: Image classification with sklearn.

There are many different approaches to image classification. One heavily used method is Convolutional Neural Networks (CNNs) and there's a good example of how to implement a CNN using the [keras library](https://keras.io/) in [this blog](https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/) and [this blog](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/).

In this repo we've provided an example [classifier](https://github.com/darabigdata/IDWBotswana/blob/master/CHALLENGE-2/Classifying_Zebra_Images.ipynb) that uses a combination of [Gabor filters](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html) and [Support Vector Machines (SVMs)](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72). 

The purpose of the Gabor filter is to extract machine learning **features** on multiple scales from an image. By doing this it compresses the information in each image down to a small set of numbers. First we need to define the type of Gabor filters we want to use:

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

and then we need to define the number of **scales** we want to filter on:

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

Once we've done that we can apply the filters to our "zebra" images. We are using **2 filters** on **4 scales**, so we will get an output of **8 features** for each image.

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

We now need to do the same for our "not zebra" images:

```python
nozebra_feats = np.zeros((len(nozebra_images),9))
for i, image in enumerate(nozebra_images):
    im = plt.imread(image,format='jpeg')
    imfeats = compute_feats(im.mean(axis=2),kernels).reshape(-1)
    nozebra_feats[i,:-1] = imfeats 
    nozebra_feats[i,-1] = 0
```

We'll combine all of these features into a single dataset:

```python
#combine the datasets
ds = np.concatenate((nozebra_feats,zebra_feats), axis=0)
features = ds[:,:-1]
```

and then we need to normalise the feature values to lie between 0 and 1. We can do this using a library routine from the scikit-learn library:

```python
features = MaxAbsScaler().fit_transform(features)
```

We need to tell our machine learning classifier which column in the dataset corresponds to the target class, i.e. "zebra" or "not zebra":

```python
target = ds[:,-1]
```

and then we can split the full dataset into:

* a training data set (to train our classifier), and 
* a test dataset (to test our classifer).

If we wanted, we could also add in a *validation* dataset to test for over-fitting... we won't do that in this simple example, but you might want to think about it for your own classifier.

```python
x_train, x_test, y_train, y_test = train_test_split(features,target)

print('Training data and target sizes: \n{}, {}'.format(x_train.shape,y_train.shape))
print('Test data and target sizes: \n{}, {}'.format(x_test.shape,y_test.shape))
```

Now we have to choose a classifer. For this example we're going to use a [Support Vector Machine (SVM) from the scikit-learn library](http://scikit-learn.org/stable/modules/svm.html). There are various options for how to implement support vector machines in scikit-learn; here we're using the Support Vector Classifier (SVC) and you can find a description of the parameters [here](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC). 

To use it all we have to do is (1) call the algorithm, and (2) tell it to fit a machine learning model using our training data:

```python
# Create a classifier: a support vector machine classifier
classifier = svm.SVC(C=1, kernel='rbf', gamma=1)

# fit to the training data
classifier.fit(x_train,y_train)
```

Once we've trained the machine learning model we can test how well it works using our test data:

```python
# now predict the value of the digit on the test data
y_pred = classifier.predict(x_test)
```

To assess how well it performed there are a range of methods. A simple way to view the results is the [confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/):

```python
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
```

There are also the standard error metrics:

```python
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))
```

This nice [slide by Nikos Nikolaou](https://github.com/as595/4IR-ClassificationWorkshop/tree/master/NIKOS_NIKOLAOU) summarises some of the standard metrics for assessing how well machine learning algorithms perform.

<p align="center"><img width=80% src="https://github.com/darabigdata/IDWBotswana/blob/master/media/errors.png"></p>

Or you can read about it online, for example [here](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c) and [here](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall).



-----

This tutorial is based on a [JBCA hack challenge](https://github.com/hrampadarath/JBCA_Hack_Night_Dec/tree/master/google_images_webscraping)
