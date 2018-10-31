
<p align="center"><img width=40% src="https://github.com/darabigdata/IDWBotswana/blob/master/media/music_notes.jpg"></p>

# Challenge 3: Build a machine learning music classification system

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
* **templates**
    * *A directory of Flask HTML templates*
* **Procfile**
    * *A Heroku Procfile to turn the Flask app into a web application*
* **requirements.txt**
    * *A Heroku requirements file to turn the Flask app into a web application*

### Dependencies

* [librosa](https://librosa.github.io/librosa/)
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

This is a simple tutorial, using the example provided in this repository on converting a Machine Learning (ML) project into a web application using Flask and deploying to the web on Heroku. For more extensive tutorials on Flask and Heroku please see:

1. https://xcitech.github.io/tutorials/heroku_tutorial/
2. https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world

### Step 1. Understand what you want to create

This may seem as a silly step, but it is the most important. Having an idea on what you want to build will help you understand what 
you will need from the Machine Learning part of the project. In this tutorial we'll make a simple web application to provide 
movie recommendations, based on a movie the user liked. The input will be a name of the movie and the output will be recommendations based on that movie. For this to work we'll need: (1) a database of movies; and (2) a machine learning model for the recommendations. 
