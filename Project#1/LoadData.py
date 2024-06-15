import os
import torch
from PIL import Image


# CustomDataset: load formatted dataset
class CustomDataset(torch.utils.data.Dataset):
    # __init__: initialization
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        self.images = self._make_dataset()


    # _find_classes: collect all classes along with their integer labels
    def _find_classes(self):
        # Find all class names
        classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
        classes.sort()
        
        # Map string to integer labels
        class_to_idx = {'Crayon_Shin': 0, 'Doraemon': 1, 'Hua_Family': 2, 'Ilu': 3, 'Maruko': 4}
        
        return classes, class_to_idx

    
    # _make_dataset: make images and their labels into a tuple, and return them as a dataset
    def _make_dataset(self):
        images = []
        
        # for every class
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            # for every image of the current class
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                item = (img_path, self.class_to_idx[class_name])
                images.append(item)
                
        return images


    # __len__: return the length of the dataset
    def __len__(self):
        return len(self.images)


    # __getitem__: get one single (image, label) item from the dataset
    def __getitem__(self, index):
        img_path, label = self.images[index]
        
        # image pre-processing
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label