"""
Webscraping images from Google images. From 
https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/

This assumes that you have a file with list of urls for images to be downloaded.
To get this follow the instructions on the webpage above.

requires two inputs:
-u: path to file containing image URLs
-o: path to output directory of images


"""

# import the necessary packages
#from imutils import paths
import argparse
import requests
import os
import glob
import time
import matplotlib.pyplot as plt

start = time.time()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
	help="path to file containing image URLs")

ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())
 
# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0


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





print('Total time in minutes = {}'.format((time.time() - start) / 60.0))


