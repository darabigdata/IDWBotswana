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


### ...write my presentation slides?

[link removed]()


### ...copy all my work off the IDIA cloud to github?

If you do not already have a github account you will need to [create one](https://github.com) (this is free). Make a note of the email address you used to register, your own github username and your github password. 

Once you've registered look for a button that says "New Repository". 

Click the button and fill in the following form like this:

<p align="center"><img width=80% src="https://github.com/darabigdata/IDWBotswana/blob/master/media/github1.png"></p>

The "Owner" field should contain your own github username.

**Click "Create Repository"**

Your github repo has now been created. Click on "idwdarahack" at the top to see the empty repository.

Now go to the IDIA cloud. The following instructions assume that you are in your home directory.

If you cloned the IDWBotswana github repo onto the cloud you should first do:

```bash
cd IDWBotswana
rm -rf .git
```

Then change directory back to your home directory and initialise this directory for github using the following commands:

```bash
cd
echo "# My IDWBotswana hackathon" >> README.md
git init
git add README.md
git add *
```

Then use the following sequence of commands, but remember to substitute:

* me@myemail.com - the email address you used to register your github account
* mygithubname - your own github user name (note this is used in the git config commandline **and** the git remote commandline!)

```bash
git config --global user.email "me@myemail.com"
git config --global user.name "mygithubname"
git commit -m "first commit"
git remote add origin https://github.com/mygithubname/idwdarahack.git
git push -u origin master
```

You will now be asked for your github user name and password.

And that's it - all your code should now be on github.

*Note: If you accidentally get the url wrong in the "git remote ..." command, you can reset it by typing:*

```bash
git remote set-url origin https://github.com/mygithubname/idwdarahack.git
```
