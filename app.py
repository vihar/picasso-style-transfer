from flask import Flask, render_template

import re

import torch

from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16


app = Flask(__name__)


@app.route("/")
def stylize():
    device = torch.device("cpu")

    content_image = utils.load_image("sample3.jpg")
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load("saved_models/mosaic.pth")
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
    utils.save_image("images/output-images/sample3.jpg", output[0])
    print("Done")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
