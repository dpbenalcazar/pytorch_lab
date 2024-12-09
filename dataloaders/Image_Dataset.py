import torch
import numpy as np
from PIL import Image
from utils.utils import read_list
from torch.utils.data import Dataset
#from torchvision.transforms import v2
from torchvision import transforms as T

class Image_Dataset(Dataset):
    def __init__(self, list_path, args=None, img_size=None, transform=None, limit=None):
        # Get inputs
        if transform:
            self.img_size = (args.img_size,args.img_size)
            self.da_library = args.da_library
            self.transform = transform
        else:
            self.img_size = img_size
            self.da_library = None
            self.transform = None

        # Make transformation to convert to tensor
        #self.to_tensor = v2.Compose([
        #            v2.PILToTensor(),
        #            v2.ToDtype(torch.float32, scale=True),
        #            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #            ])
        self.to_tensor = T.Compose([
            T.ToTensor(),
        ])
        # Read dataset from list
        img_paths, labels = read_list(list_path, limit)
        self.images = img_paths

        # Get labels and indexes
        self.labels = np.array(labels, dtype=int)
        self.index = np.array([*range(len(self.labels))], dtype=int)

        # Set dataset length
        self.len = len(self.images)

        # Get number of classes
        self.n_classes = len(np.unique(self.labels))

    def __len__(self):
        return self.len
    
    def apply_transform(self, image):
        if self.transform:
            if self.da_library in ['torchvision', 'pytorch', 'torch']:
                # Transform image using torchvision
                image = self.transform(image)
            else:
                # Convert to numpy and transform
                image = self.transform(image=np.array(image))
                # Obtain transformed image
                if self.da_library in ['albumentations', 'Albumentations']:
                    image = image['image']
                # Convert to pillow
                image = Image.fromarray(image)
        return image

    def load_image(self, i):
        pass
        # Read image
        img = Image.open(self.images[i]).convert('RGB').resize(self.img_size)
        # Apply transformation
        img = self.apply_transform(img)
        # Transform to tensor
        img = self.to_tensor(img)

        return img

    def __getitem__(self, item):
        # Get data
        label = self.labels[item]
        image = self.load_image(item)

        return image, label, item
