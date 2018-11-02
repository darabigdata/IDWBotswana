
<p align="center"><img width=30% src="https://github.com/darabigdata/IDWBotswana/blob/master/media/movie_clapper.jpg"></p>

# Challenge 1: Build a machine learning recommendation app

Machine Learning can be used to provide awesome applications and services. Examples are recommender systems on Netflix, Amazon and even Google reverse Image search. However, creating an application isn't limited to big tech firms. For this challenge you will build a web application of your choice. To help you get started we've provided a tutorial on how to build a machine learning movie recommendation application.

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
* **Flask_App.ipynb**
    * *A jupyter notebook that contains code (the same as app.py) to make a simple Flask application*
* **templates**
    * *A directory of Flask HTML templates*
* **Procfile**
    * *A Heroku Procfile to turn the Flask app into a web application*
* **requirements.txt**
    * *A Heroku requirements file to turn the Flask app into a web application*

### Dependencies

* [pandas](https://pandas.pydata.org/)
* [json](https://docs.python.org/3/library/json.html)
* [sklearn](scikit-learn.org/)
* [pickle](https://docs.python.org/3/library/pickle.html)
* [virtualenv](https://virtualenv.pypa.io/)
* [flask](flask.pocoo.org/)
* [random](https://docs.python.org/3/library/random.html)
* [gunicorn](https://gunicorn.org/)

------

## Simple Movie Recommender Tutorial

This is a simple tutorial that shows you how to convert a Machine Learning (ML) project - here a movie recommendation engine - into a web application using Flask and deploying to the web on Heroku. If you'd like more detail on either of these you can check out this [flask tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world) and/or this [heroku tutorial](https://xcitech.github.io/tutorials/heroku_tutorial/).

### Step 1. Understand what you want to create

This may seem as a silly step, but it is the most important. Having an idea of what you want to build will help you understand what 
you will need from the Machine Learning part of the project. In this tutorial we'll make a simple web application to provide 
movie recommendations, based on a movie the user liked. The input will be a name of the movie and the output will be recommendations based on that movie. For this to work we'll need: (1) a database of movies; and (2) a machine learning model for the recommendations. 

### Step 2. Build your model and save the important bits

This is where you work on your ML project and optimise your models. In this example we'll make a simple movie recommender using the [TMDB database from Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata). The method we'll use is implemented in the [Simple_Movie_Recommender jupyter notebook](https://github.com/darabigdata/IDWBotswana/blob/master/CHALLENGE-1/Simple_Movie_Recommender.ipynb) in this repository.

In most cases, you will not want to have the web application running the ML training and data pre-processing every time a user opens the  application (although in some cases you may, if the application uses data from the user). Doing the data pre-processing and ML heavy lifting off-line and exporting the model will make your application more efficient. The [example notebook](https://github.com/darabigdata/IDWBotswana/blob/master/CHALLENGE-1/Simple_Movie_Recommender.ipynb) exports two datasets:

1. A cleaned version of the original dataset: 
here, only the columns we're interested in ('original_title', 'genres', 'popularity') are exported as a cleaned dataset.
```python
#make new data frame
movies_new_df = movies[['original_title', 'genres','popularity']]
# save new file 
movies_new_df.to_csv('tmb_movies_clean.csv', index=False)
```
2. The ML model using pickle: recommender systems can either use content or collaborative filtering. Content is the more simple approach and here we're using genres and ratings/popularity. In this example the ML uses the unsupervised version of the [k-nearest-neighbours algorithm](https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/), and the model itself is exported using pickle:
```python
from sklearn.neighbors import NearestNeighbors
#build the model
nn_model = NearestNeighbors(n_neighbors=5,algorithm='auto').fit(features)

#Obtain the indices of and distances to the the nearest K neighbors of each point.
distances, indices = nn_model.kneighbors(features)

#Export model indices as a pickle file
import pickle
with open('movieindices.pkl', 'wb') as fid:
    pickle.dump(indices, fid,2)
```

### Step 3. Create a virtual environment

Before starting with Flask to build the web application, it's sensible to create and start a virtual environment (this will make things simpler when you push the app to Heroku later):

```bash
> virtualenv -p python3 env_flask
> source env_flask/bin/activate
```
Then install Flask and other prerequisites. (You will need to install all modules you plan to use in your Flask app, here we need only panda additionally)

```bash
>pip install flask
>pip install gunicorn
>pip install panda
```
### Step 4. Build the Flask app

The Flask app consists of 2 main components: (1) the main code 
[app.py](https://github.com/darabigdata/IDWBotswana/blob/master/CHALLENGE-1/app.py) and (2) the HTML templates, which are saved here in a folder called 
[templates](https://github.com/darabigdata/IDWBotswana/tree/master/CHALLENGE-1/templates). A simple app.py returns a rendered version of the html files in the templates folder. For example:

```python
from flask import Flask, request, render_template
import pickle
import pandas as pd

# initilise Flask
app = Flask(__name__)

@app.route('/') # the webpage link/extension
def main():
    return render_template('home.html') # call to the html template named "home.html"
```

The home.html file:

```html
{% extends "layout.html" %}
{% block body %}

<div class="container" style="width:100%; height:60%">
<h1>Movie Recommender App</h1>

    <div>
        <form action = "/similarByName" method = 'POST'>
	    <p> <input name="name" type ="text" placeholder="Search by Name" />
        <input type ="submit" value="submit" /> </p>
        </form>
    </div>

</div>
{% endblock %}

```
The above html file has three important features:

1. extends "layout.html" - the [layout.html](https://github.com/darabigdata/IDWBotswana/blob/master/CHALLENGE-1/templates/layout.html) is a general html file that dictates the look of the application. All the html files/pages in the application are an extention of this page.
2. <form action = "/similarByName" method = 'POST'> here a form is used to create a query (i.e. name of movie), that makes a call to a function "/similarByName" in the [app.py](https://github.com/darabigdata/IDWBotswana/blob/master/CHALLENGE-1/app.py). Once this query is made, the function "/similarByName" searches the database for a movie with a similiar name and returns a dictionary. 
3. This is all wrapped in an html block that is passed to layout.html, creating the webpage. 

The overall aim of this web app is that user will search for the movie in a form provided (a query that is sent to 
"/similarByName" via  "similar.html") and for each item in the list there will be an associated recommend button (that sends a query to another function /similarByContent also by  "similar.html"). The "/similarByContent" function is where the pickle file is used. 

To complete this application, we added a random movie generator and an about page.

You can either run the app.py script directly:

```bash
> python app.py
```

or you can follow the steps in the [Flask_App jupyter notebook]((https://github.com/darabigdata/IDWBotswana/blob/master/CHALLENGE-1/Simple_Movie_Recommender.ipynb).

**Note:** Before you link it to Heroku you can still see your app in the web-browser. Flask will make a webpage at http://127.0.0.1:5000/

### Step 5. Deploy to Heroku

You will need a [Heroku](https://www.heroku.com/) account and the [HerokuCLI](https://devcenter.heroku.com/articles/heroku-cli). For this tutorial, we can use the free version of Heroku.

Create the Procfile: A Procfile is a mechanism for declaring what commands are run by your application's dynos on the Heroku platform. Create a file called “Procfile” and put the following in it:

	web: gunicorn app:app

Create the python requirements file by running the following at the command prompt (within the virtual environment):

 	pip freeze > requirements.txt

Set up HerokuCLI using the instructions here.
Create a new app on the Heroku Website by logging into your account. You can ignore add to pipelines. I named my app: movie-recommender-example.

Login to Heroku through the command prompt:

    heroku login

Upload to Heroku (the instructions will be listed in your Heroku app page):

    > git init
    > heroku git:remote -a movie-recommender-example
    > git add .
    > git commit -am "make it better"
    > git push heroku master

Your app should be live

    use the "Open app" link on the Heroku app page
    or https://YOURAPP.herokuapp.com/
    
You can view the final web-app from this tutorial at https://movie-recommender-example.herokuapp.com/    

-----

This tutorial is based on a [JBCA hack challenge](https://github.com/hrampadarath/JBCA_Hack_Night_Dec/tree/master/web_app).
