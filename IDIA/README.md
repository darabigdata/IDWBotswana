## What if I want to...

### ...install a different python library?

If you want to install a python library (for example the pandas library) you will need to do a local install. You can do this on the command line in the terminal using

```python
pip install —user pandas
```

or directly in your Jupyter notebook using

```python
import sys
! {sys.executable} -m pip install —user pandas
```

Local libraries are installed in /home/USERNAME/.local/lib/python3.6/site-packages/ so you will need to add this to your system path. In a notebook cell type:

```python
import sys
sys.path.append('/home/USERNAME/.local/lib/python3.6/site-packages/')
sys.path.append('/opt/workshop/lib/python3.6/site-packages')
```

again, remembering to replace USERNAME with your own userid. You should then be able to import the library.

If you want to see the current path, use:

```python
print(sys.path)
```

### ...work from the terminal?

Before you can run python scripts on the command line in the terminal you will need to specify the PYTHONPATH environment variable. You can do this by using the following lines, replacing USERNAME with your own userid.

```bash
export PYTHONPATH="/opt/workshop/lib/python3.6/site-packages/:/home/USERNAME/.local/lib/python3.6/site-packages/"
```

### ...upload something to the IDIA cloud?

Either grab it from github or use the upload button on the top left of the user interface <img src="https://github.com/darabigdata/IDWBotswana/blob/master/media/upload.png" width=5%>

### ...fix my librosa dependencies?

```bash
cd
mkdir ffmpeg
cd ffmpeg
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-32bit-static.tar.xz
tar -xvf ffmpeg-git-32bit-static.tar.xz
export PATH="$PATH:/home/USERNAME/ffmpeg/ffmpeg-git-32bit-static"
```

remember to replace USERNAME with your own user name!

You can test it out by running the file [test_librosa.py]() in the terminal:

```bash
> python test_librosa.py
```
