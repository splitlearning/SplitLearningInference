from flask import Flask, jsonify, request
from PIL import Image
from flask_cors import CORS, cross_origin
import json
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from imagenet import imagenet_classes

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy dog'
# app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/*": {"origins": "http://localhost:5000"}})

# todo: load only the server model weights and not full model weights on the server
full_model = models.resnet18(pretrained=True)

split_layer = 5
# need to add the flatten layer since it is usually called within ResNet18's .forward() and not included in .children()
full_model.fc = nn.Sequential(nn.Flatten(), list(model.children())[9])
server_model = nn.Sequential(*nn.ModuleList(model.children())[split_layer:])
server_model.eval()

@app.route('/')
def hello():
    return 'The Split Learning Inference ResNet18 Demo Server is up!'

@app.route('/inference', methods=['POST'])
@cross_origin(headers=['Access-Control-Allow-Origin'])
def inference():
    if request.method == 'POST':
        request_dict = request.get_json()
        split_activation = request_dict["data"]
        dims = request_dict["dims"]
        reshaped_split_activation = torch.Tensor(np.reshape(split_activation, dims))
        output_tensor = server_model(reshaped_split_activation)
        output_class = imagenet_classes[str(int(torch.argmax(output_tensor)))]
        print("Output Class: {}".format(output_class[1]))
        return jsonify({'class': output_class[1]})

if __name__ == "__main__":
    app.run(host="0.0.0.0")
