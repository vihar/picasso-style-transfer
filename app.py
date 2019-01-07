from flask import Flask, render_template, request, redirect, url_for
import re
import torch
from torchvision import transforms
import utils
from transformer_net import TransformerNet
from flask_dropzone import Dropzone
import os
import random


app = Flask(__name__)

dropzone = Dropzone(app)
app.config.update(
    UPLOADED_PATH='uploads',
    DROPZONE_MAX_FILE_SIZE=10000,
    DROPZONE_INPUT_NAME='data',
    DROPZONE_MAX_FILES=1,
)

IMAGE_FOLDER = os.path.join('static', 'images')

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        input_img = request.form['imgurl']
        style = request.form['style']
        return redirect(url_for('stylize', input_img=input_img, style=style))

    return render_template('app.html')


@app.route("/model")
def stylize():
    device = torch.device("cpu")
    input_img = request.args.get('input_img')
    model_get = request.args.get('style')
    print(model_get)

    content_image = utils.load_image(input_img)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    model_get = str(model_get)
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model_get)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
        a = random.randint(1, 101)
        img_path = str("static/images/output_{}.jpg".format(a))
    utils.save_image(img_path, output[0])
    image_k = str("output_{}.jpg".format(a))

    get_image = os.path.join(app.config['UPLOAD_FOLDER'], image_k)

    print("Done")

    return render_template("index.html", get_image=get_image)


if __name__ == "__main__":
    app.run(debug=True)
