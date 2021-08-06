from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from flask_cors import CORS, cross_origin
import json
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from app.imagenet import imagenet_classes
import time

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy dog'
# app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/*": {"origins": "https://localhost:5000"}})

USE_CUDA = torch.cuda.is_available()
# todo: load only the server model weights and not full model weights on the server
full_model = models.resnet18(pretrained=True)

split_layer = 5
# need to add the flatten layer since it is usually called within ResNet18's .forward() and not included in .children()
full_model.fc = nn.Sequential(nn.Flatten(), list(full_model.children())[9])
server_model = nn.Sequential(*nn.ModuleList(full_model.children())[split_layer:])
server_model.eval()
if USE_CUDA:
    server_model.cuda() 
    print("Using CUDA")
else:
    print("Using CPU")

@app.route('/')
def hello():
    return 'The Split Learning Inference ResNet18 Demo Server is up!'

@app.route('/inference', methods=['POST'])
@cross_origin(headers=['Access-Control-Allow-Origin'])
def inference():
    if request.method == 'POST':
        start_time = time.time()
        request_dict = request.get_json()
        print("Parsing request time: {}".format(time.time()-start_time))
        split_activation = request_dict["data"]
        dims = request_dict["dims"]
        start_time = time.time()
        reshaped_split_activation = torch.Tensor(np.reshape(split_activation, dims))
        if USE_CUDA:
            reshaped_split_activation = reshaped_split_activation.cuda()
        output_tensor = server_model(reshaped_split_activation)
        output_class = imagenet_classes[str(int(torch.argmax(output_tensor)))]
        print("Output Class: {}, inference time: {}".format(output_class[1], time.time() - start_time))
        return jsonify({'class': output_class[1]})

@app.route('/static/<path:path>', methods=['GET'])
@cross_origin(headers=['Access-Control-Allow-Origin'])
def serve_static_files(path):
    return send_from_directory('static', path)

