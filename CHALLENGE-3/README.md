
<p align="center"><img width=40% src="https://github.com/darabigdata/IDWBotswana/blob/master/media/music_notes.jpg"></p>

# Challenge 3: Build a machine learning music classification system

Music recommendation systems are all over the internet, from Spotify to iTunes. But how do they know what music you will like? For this challenge you build a machine learning application that classifies music using the content of the individual tracks. Your application could make recommendations for individuals, or it could suggest musical tracks that would be good in films, or it could automatically identify artists, or it could do something else! The choice is up to you. 

### What's in the repo?

* **Extract_Features.ipynb**
    * *A jupyter notebook that uses the librosa library to extract machine learning features from audio files*
* **Music_Classifier.ipynb**
    * *A jupyter notebook that implements simple machine learning to classify music*
* **data_5band**
    * *A folder containing files with machine learning features for different artists/bands*
* **data_5and25band**
    * *A folder containing files with machine learning features for different artists/bands*
    
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

## Simple Music Classification

We've provided dictionaries of features for a selection of different artists in the two folders [data_5band]() and [data_5and25band]() so if you just want to skip straight to the machine learning you can ignore Step 1 & Step 2 and go immediately to Step 3. However, if you want to extract features for your own favourite artists (using your own music library) we've also provided some example code to show you how. The code is set up to look for all the audio files (.mp3, .mp4, .m4a etc.) for a particular artist/band in the same directory, but you can alter that if you need to.

### Step 1. Load in some music

Extracting features from audio can be a time consuming process. If you want to speed things up you can parallelize your python! First you probably want to find out how many cores you can parallelize over:

```python
num_workers = multiprocessing.cpu_count() 
print('you have {0} cores available to do your bidding...'.format(num_workers))
```

You'll need a function to load the music in:

```python
import librosa as lb

def load_music(songname1,songpath1):
    try:
        print('loading the song: {0} ......... located here: {1} '.format(songname1, songpath1))
        songdata1, sr1 = lb.load(songpath1) # librosa library used to grab song data and sample rate
        print ('done........ '+songname1)
        return [songname1,songdata1,sr1]
    except: # the song could be corrupt or you could be trying to load something which isn't a song
        print('..............................FAILED...............................')
        print(songpath1)
        print('...................................................................')
        return
```

Then you can specify the directory where the audio files are and make a list of their titles:

```python
path='BandOfSkulls'

for song in os.listdir(path):
    print (song)
    songname.append(song)
    songpath.append(path+'/'+song)
```


 ...before loading them in. You can do this either in **parallel**:

```python
# Parallel version:
with multiprocessing.Pool(processes=num_workers) as pool:
    songdb=pool.starmap(load_music,zip(songname,songpath))
    pool.close()
    pool.join()
```
or as a normal **serial** operation:

```python
# Serial version:
songdb=[]
for i in range(0, len(songname)):
    songdb.append(load_music(songname[i], songpath[i]))
```

Finally just check that they all loaded!

```python
print ('>>> loaded {0} songs into memory'.format(len(songdb)))
```

### Step 2. Extract features from your data

We'll start by separating out the name of each song, the feature data for that song and the sampling rate for the song.

```python
for song in songdb: 
    song_name.append(song[0])
    song_data.append(song[1])
    song_sr.append(song[2])
```

To extract features from music data we can use the python library [librosa](https://librosa.github.io/librosa/), which extracts musical attributes from the time and frequency data in each audio file. It's very versatile, so there's no single set of features that covers everything. In the example script in this repo we're using the features suggested by [Alex Clarke]((https://informationcake.github.io/music-machine-learning/)). These are extracted by the function **get_features_mean()**, but you can use librosa to extract your own favourite set of features.

You can find a description of some of the musical features that librosa automatically extracts [here](https://librosa.github.io/librosa/feature.html).

Whatever you specify in your function, you'll need to run it over all your songs. As before, you can either do this in parallel:

```python
# Parallel version:
with multiprocessing.Pool(processes=num_workers,maxtasksperchild=1) as pool:
    res=pool.starmap(get_features_mean,zip(song_data,song_sr,itertools.repeat(hop_length1),itertools.repeat(n_fft1)))
    pool.close()
    pool.join()
```

or as a normal serial operation:

```python
# Serial version:
res=[]
for i in range(0,len(song_data)):
    res.append(get_features_mean(song_data[i], song_sr[i], hop_length1, n_fft1))
```

You can then concatenate all of these features into a single dictionary for that specific artist/band:

```python
data_dict_mean={}
for i in range(0,len(songdb)):
    data_dict_mean.update({song_name[i]:res[i]})
```

You can check what features are in your dictionary like this:

```python
print('>>> The features extracted from the songs are: ')
print(res[0].keys())
```

Then finally you probably want to write those features into a file because this process took quite a while and we don't want to have to keep repeating it...

```python
print('>>> Saving dictionary to disk...')
savefile=str(path)+'_data'
save_obj(data_dict_mean,savefile)
```


### Step 3. Run some machine learning

We'll need a function to extract the information from the dictionaries of features that we saved to disk. Something like this should work:

```python
def prepare_data(all_data_in):
    
    all_features=[]
    all_artists=[]
    
    # Create lists of song names and features for each artist:
    for artist in all_data_in: 
        
        # load in the feature dictionary for the artist:
        data=load_obj(artist.replace('.pkl',''))
        print('loading {0}'.format(artist))
        
        songname=[] # will be a list of song names
        songfeat=[] # will be a list of dictionaries containing the feature data
        artists=[]  # will be a list of artists
        
        # extract out the features, song name and artist into separate lists:
        for song in data: 
            songfeat.append(data[song]) 
            songname.append(song)
            artists.append(artist.replace('_data.pkl','').replace('all_','').replace(path,'').replace('_data_testsplit.pkl','').replace('_data_trainsplit.pkl',''))

        # make a list of the feature names:
        feature_names=list(songfeat[0].keys()) 
        
        # make a list all the feature values for this artist:
        features=[] 
        for i in range(len(songfeat)):
            features.append(list(songfeat[i].values())) 
            
        # append the feature values for this artist into a master list:
        all_features+=features
        
        # append the artist name for this artist into a master list:
        all_artists+=artists
        
    return all_features, all_artists, feature_names
```

We then need to find the data to run it on. So to start with we should tell the program where we put all of the dictionary files:

```python
path='./data_5band/'
```

Then we should make a list of them all:

```python
all_data=glob.glob(path+'/*_data.pkl')
```

Then we use the function we defined above to read in all the data:

```python
all_features, all_artists, feature_names = prepare_data(all_data) 
```

Whatever form of machine learning we end up using we're going to need to split our input data into:

* training data (to train our machine learning algorithm)
* test data (to test how well the training worked)

Let's start by taking 90% of the data for training:

```python
train_percent=0.9
test_percent=0.1
```

We can then use the [scikit-learn function train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to randomly divide the full dataset for us:

```python
features_train, features_test, artists_train, artists_test = train_test_split(all_features, all_artists, train_size=train_percent, test_size=test_percent, random_state=0, stratify=all_artists)
```

There are two examples of different machine learning approaches in the [MusicClassification.ipynb jupyter notebook](), here we'll just look at a simple random forest classifier.

First we need to build our forest:

```python
n_estimators=1000 # number of trees
forest = RandomForestClassifier(n_estimators=n_estimators, random_state=2, class_weight='balanced')
```

Then we need to train the machine learning model:

```python
forest.fit(features_train, artists_train)
```

Now we can test it:

```python
artists_pred = forest.predict(features_test)
```

...and look at the performance metrics to see how well we did:

```python
print(classification_report(artists_test, artists_important_pred,target_names=names))
```


-----

This tutorial is based on a [JBCA hack challenge](https://github.com/informationcake/music-machine-learning) and the work by [Alex Clarke](https://informationcake.github.io/music-machine-learning/).
