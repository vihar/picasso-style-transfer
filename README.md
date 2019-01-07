# Neural Style Transfer on Flask Server 🚀



```
$ virtualenv venv
$ source venv/bin/activate
$ python app.py
```


On Heroku use, http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl

```
python3 local.py eval --content-image <inputimage_path> --model <model_path> --output-image <outputimage_path> --cuda 0
```

python3 local.py eval --content-image input.jpg --model saved_models/mosaic.pth --output-image output.jpg --cuda 0