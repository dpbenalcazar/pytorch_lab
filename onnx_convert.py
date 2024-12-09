import os
import json
import torch
import numpy as np
from PIL import Image
#from torchvision.transforms import v2
from torchvision import transforms as T
from models.classification import load_model
from models.siamese import siamese_embeddings
from argparse import ArgumentParser as argparse

parser = argparse()
parser.add_argument('-m', '--model_folder', required=True, 
                    help='Path to model folder')
args = parser.parse_args()

# Read model parameters
weights = os.path.join(args.model_folder, 'best_model.pth')
config = os.path.join(args.model_folder, 'log.json')
cfg_dict = json.load(open(config))
backbone = cfg_dict["backbone"]
model_type = cfg_dict["model_type"]
W, H = cfg_dict["image_size"]
NUM_CLASSES = cfg_dict["classes"]
RESOLUTION =  (W, H)
NUM_CHANNELS = 3

# Use cpu
device = 'cpu'

# Load model
if model_type == 'classification':
    model = load_model(backbone, weights, NUM_CLASSES)
elif model_type == 'siamese':
    model = siamese_embeddings(backbone, weights)
else:
    print('Model type "{}" not supported'.format(model_type))
    exit()
model.to(device)
model.eval()

# Count number of parameters
n_params = sum(p.numel() for p in model.parameters())
print('{}_{} successfully loaded in {}'.format(backbone, model_type, device))
print('Number of parameters: {:d}\n'.format(n_params))

# Generate Dummy input
#tform = v2.Compose([
#            v2.PILToTensor(),
#            v2.ToDtype(torch.float32, scale=True),
#            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#            ])
tform = T.Compose([
            T.ToTensor(),
        ])
dummy_input = 255*np.random.rand(128,128,3)
dummy_input = Image.fromarray(dummy_input.astype('uint8')).resize(RESOLUTION)
dummy_input = tform(dummy_input).unsqueeze(0)

# Save with onnx format
name = os.path.basename(weights).split('.')[0] + '.onnx'
opt_path = os.path.join(args.model_folder, name)
#with torch.no_grad:
torch.onnx.export(model, dummy_input, opt_path)
print('Model saved at:')
print(opt_path)

# Next step
print('\nTo serialize the model with OpenVINO run the following command:')
print('mo --framework=onnx --input_model={} --input_shape=[1,{},{},{}] --output_dir={}'.format(opt_path, NUM_CHANNELS, H, W, args.model_folder))