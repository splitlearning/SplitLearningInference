import torch
import torch.nn as nn
import torchvision.models as models


model = models.resnet18(pretrained=True)

split_layer = 5
fc_features = model.fc.in_features
class_num = 1000
# note - the indexing 10 is specific to resnet18
model.fc = nn.Sequential(nn.Flatten(),
                         list(model.children())[9])
client_model = nn.Sequential(*nn.ModuleList(model.children())[:split_layer])
server_model = nn.Sequential(*nn.ModuleList(model.children())[split_layer:])
print("exporting now")
torch.onnx.export(client_model, torch.zeros(1,3,224,224) , 'resnet18_client.onnx', verbose=True)
torch.save(server_model.state_dict(), 'resnet18_server.pt')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

