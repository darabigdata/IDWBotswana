from flask import Flask, request, render_template
import pickle
import pandas as pd
from random import randint

# initilise Flask
app = Flask(__name__)



@app.route('/') # the webpage link/extension
def main():
    return render_template('home.html') # call to the html template named "home.html"

@app.route('/about')
def about():
	return render_template('about.html')


@app.route('/similarByName',methods=['POST'])
def similar_by_name():
	# the main pythin code
	df = pd.read_csv('tmb_movies_clean.csv')
	if request.method == 'POST':
		result = request.form
	query = result['name']
	n = 0
	movie_list = []
	print('Movies with the words "{}" in the title:'.format(query))
	for i, name in enumerate(df.original_title):
		if query.lower() in name.lower():
			info = {
				"name": df['original_title'][i],
				"rating": df['popularity'][i],
				"genre": df['genres'][i]
			}
			movie_list.append(info)
			n+=1
	return render_template("similar.html",
						   title='Name',
						   name=query,
						   topmovies=movie_list)


@app.route('/similarByContent',methods=['POST'])
def similar_by_content():
	df = pd.read_csv('tmb_movies_clean.csv')
	if request.method == 'POST':
		result = request.form
	query = result['name']

	#load the model file
	pkl_file = open('movieindices.pkl', 'rb')
	indices = pickle.load(pkl_file)
	if query not in df['original_title']:
		N = df[df['original_title'] == query].index[0]
		movie_list = []
		for n in indices[N][1:]:
			info = {
				"name": df['original_title'][n],
				"rating": df['popularity'][n],
				"genre": df['genres'][n]
			}
			movie_list.append(info)

		return render_template("similar.html",
						   title='Content',
						   name=query,
						   topmovies=movie_list)

@app.route('/random',methods=['GET', 'POST'])
def random():
	df = pd.read_csv('tmb_movies_clean.csv')
	if request.method == 'POST':
		R = randint(0,len(df)-1)
		rand_movie = df.iloc[R]
	else:
		R = randint(0,len(df)-1)
		rand_movie = df.iloc[R]

	return render_template('ratings.html',
			name = rand_movie['original_title'],
			genre = rand_movie['genres'],
			ratings = rand_movie['popularity'])


if __name__ == '__main__':
    # you might need to change your port number
    # allowed values are 5000 - 5010
	app.run(host='0.0.0.0',port=5000)
