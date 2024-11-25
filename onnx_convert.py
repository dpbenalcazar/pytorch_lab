import os
import torch
from PIL import Image
#from torchvision.transforms import v2
from torchvision import transforms as T
from models.classification import load_model
from models.siamese import siamese_embeddings
from argparse import ArgumentParser as argparse

parser = argparse()
parser.add_argument('-b', '--backbone', required=True, 
                    help="Model´s backbone")
parser.add_argument('-t', '--model_type', default='clas', choices=['clas', 'siam'], 
                    help="Model type. Classiffication=clas, Siamese=siam")
parser.add_argument('-w', '--weights', required=True, 
                    help='Path to model weights file')
parser.add_argument('-c', '--classes', type=int, required=True,
                    help='Number of output classes')
parser.add_argument('-W', '--width', type=int, default=320,
                    help='Input image width')
parser.add_argument('-H', '--height', type=int, default=240,
                    help='Input image height')
parser.add_argument('-C', '--channels', type=int, default=3,
                    help='Number of channels in the input image')
args = parser.parse_args()

# Read Image parameters
RESOLUTION =  (args.width, args.height)
W, H =  (args.width, args.height)
NUM_CHANNELS = args.channels
NUM_CLASSES = args.classes
MODE = 'RGB' if NUM_CHANNELS==3 else 'L'

# Use cpu
device = 'cpu'

# Load model
if args.model_type == 'clas':
    model = load_model(args.backbone, args.weights, NUM_CLASSES)
else:
    model = siamese_embeddings(args.backbone, args.weights)
model.to(device)
model.eval()

# Count number of parameters
n_params = sum(p.numel() for p in model.parameters())
print('{}_{} successfully loaded in {}'.format(args.backbone, args.model_type, device))
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
dummy_input = Image.new(MODE, RESOLUTION, (128, 255, 0))
dummy_input = tform(dummy_input).unsqueeze(0)

# Save with onnx format
folder = os.path.dirname(args.weights)
name = os.path.basename(args.weights).split('.')[0] + '.onnx'
opt_path = os.path.join(folder, name)
torch.onnx.export(model, dummy_input, opt_path)
print('Model seved at:')
print(opt_path)

# Next step
print('\nTo serialize the model with OpenVINO run the following command:')
print('mo --framework=onnx --input_model={} --input_shape=[1,{},{},{}] --output_dir={}'.format(opt_path, NUM_CHANNELS, H, W, folder))