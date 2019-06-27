# DEMOGRAPHICS #

Age and gender estimation.

### Requirements ###

* tensorflow
* numpy
* facelib
* common

## Installation ##
```sh
sudo apt-get install libboost-all-dev libopenblas-dev liblapacke-dev cmake build-essential
sudo apt-get install python-dev python-pip python-opencv python-setuptools #python-opencv

pip install --user git+ssh://git@bitbucket.org/macherlabs/facelib.git
pip install --user git+https://github.com/ap193uee/common.git

pip install --user git+ssh://git@bitbucket.org/macherlabs/demographics.git
```
## Usage ##
    
    import cv2
    from demographics import Demography
    filename = 'test.jpg'
    dmg = Demography()
    img = cv2.imread(filename)
    if img is not None:
        print(dmg.run(img))