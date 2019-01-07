# Neural Style Transfer on Flask Server 🚀

Fast Neural Style Transfer that makes your pictures more beautiful.

```
$ virtualenv venv
$ source venv/bin/activate
$ python3 app.py
```

On Heroku use, to install use 

```
pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl

```

To run locally, 
```
python3 local.py eval --content-image <inputimage_path> --model <model_path> --output-image <outputimage_path> --cuda 0
```

Sample:
```
python3 local.py eval --content-image sample/input.jpg --model saved_models/mosaic.pth --output-image sample/output.jpg --cuda 0
```