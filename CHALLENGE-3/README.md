
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


-----

This tutorial is based on a [JBCA hack challenge](https://github.com/informationcake/music-machine-learning) and the work by [Alex Clarke](https://informationcake.github.io/music-machine-learning/).
