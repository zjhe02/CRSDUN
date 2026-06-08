import os
import numpy as np
import torch
import pandas as pd
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Resize

class RandomHorizontalFlip():
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, sample):
        x = sample['image']
        y = sample['label']
        if np.random.rand(1) < self.prob:
            sample['image'] = torch.flip(x,dims=(-1,))
            sample['label'] = torch.flip(y,dims=(-1,))
            if 'rgb' in sample:
                sample['rgb'] = torch.flip(sample['rgb'],dims=(-1,))
        return sample
    
    
class SegIdentityTransform(object):
    # Hint: Note that our transforms work on dicts. This is an example of a transform that works
    # on a dict whose elements can be converted to np.arrays, and are then converted to torch.tensors
    # This performs the scaling of the RGB by division by 255, and puts channels first by performing the permute
    # for the label, we convert to long, datatype to let torch know that this is a discrete label.
    # You might want to change this or write different transforms depending on how you read data.
    def __call__(self, sample):
        output = {'image': torch.tensor(np.array(sample['image'])).permute(2,0,1),
                'label': torch.tensor(np.array(sample['label'])).long()}
        if 'rgb' in sample:
            output['rgb'] = torch.tensor(np.array(sample['rgb'])).permute(2,0,1)
        return output
    
    
class RandomCrop():
    def __init__(self, target_size=(256,256), edge=10):
        self.target_size = target_size
        self.edge = edge
    def __call__(self, sample):
        if min(sample['label'].shape) < max(self.target_size) + 2*self.edge:
            sample = Resize(self.target_size)(sample)
            return sample
        wx, wy = self.target_size
        wx0 , wy0 = sample['label'].shape
        try:
            center_x = np.random.randint(self.edge+wx//2,wx0 - self.edge - wx//2)
            center_y = np.random.randint(self.edge+wy//2,wy0 - self.edge - wy//2)
        except:
            print(sample['label'].shape)
        crop_x_0 = center_x - wx // 2
        crop_x_1 = center_x + wx // 2
        crop_y_0 = center_y - wy // 2
        crop_y_1 = center_y + wy // 2
        sample['image'] = sample['image'][crop_x_0:crop_x_1,crop_y_0:crop_y_1].astype(float)
        sample['label'] = sample['label'][crop_x_0:crop_x_1,crop_y_0:crop_y_1].astype(int)
        return sample
    
    
class RandomCropHoriz():
    def __init__(self, target_size=(256,256), edge=10, shift=170):
        self.target_size = target_size
        self.edge = edge
        self.shift = shift
    def __call__(self, sample):
        if min(sample['label'].shape) < max(self.target_size) + 2*self.edge:
            sample = Resize(self.target_size)(sample)
            return sample
        wx, wy = self.target_size
        wx0 , wy0 = sample['label'].shape
        try:
            center_x = np.random.randint(self.edge+wx//2,wx0 - self.edge - wx//2)
            center_y = np.random.randint(self.edge+wy//2,wy0 - self.edge - wy//2)
        except:
            print(sample['label'].shape)
        crop_x_0 = center_x - wx // 2
        crop_x_1 = center_x + wx // 2
        crop_y_0 = center_y - wy // 2
        crop_y_1 = center_y + wy // 2
        sample['image'] = sample['image'][self.shift:self.shift+self.target_size[0],crop_y_0:crop_y_1].astype(float)
        sample['label'] = sample['label'][self.shift:self.shift+self.target_size[0],crop_y_0:crop_y_1].astype(int)
        if 'rgb' in sample:
            sample['rgb'] = sample['rgb'][self.shift:self.shift+self.target_size[0],crop_y_0:crop_y_1].astype(float)
        return sample


def get_class_names():
    return ['bg,',
        'real potato',
        'fake potato',
        'real apple',
        'fake apple',
        'real orange',
        'fake orange',
        "real grape",
        "fake grape",
        "real lemon",
        "fake lemon",
        "real avocado",
        "fake avocado",
        "real pepper",
        "fake pepper",
        "real plant",
        "fake plant",
        "real banana",
        "fake banana",
        "real onion",
        "fake onion",
        "real unknown",
        "fake unknown"]  

def get_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (23, 3)
    """
    return np.asarray([[0, 0, 0],
            [128, 0, 0], # potato
            [0, 128, 0],
            [128, 128, 0], # apple
            [0, 0, 128],
            [128, 0, 128], # orange
            [0, 128, 128],
            [128, 128, 128], # grape
            [64, 0, 0],
            [192, 0, 0], # lemon
            [64, 128, 0],
            [192, 128, 0], # avocado
            [64, 0, 128],
            [192, 0, 128], # pepper
            [64, 128, 128],
            [192, 128, 128], # plant
            [0, 64, 0],
            [128, 64, 0], # banana
            [0, 192, 0],
            [128, 192, 0], # onion
            [0, 64, 128],
            [128, 64, 128], # unknown
            [0, 192, 128]])
    
def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """

    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
    for i, label in enumerate(get_labels()):
        label_mask[np.where(np.all(mask==label, axis=-1))[:2]] = i
    return label_mask

def decode_segmap(mask, unk_label=255):
    """Decode segmentation label prediction as RGB images
    Args:
        mask (torch.tensor): class map with dimensions (B, M,N), where the value at
        a given location is the integer denoting the class index.
    Returns:
        (np.ndarray): colored image of shape (BM, BN, 3)
    """
    mask[mask == unk_label] == 0
    mask = mask.numpy() # 1, 1, H, W
    cmap = get_labels() # 23, 3
    cmap_exp = cmap[..., None]
    colored = cmap[mask].squeeze() # B, C, H, W
    return colored

class HySpecSegmentation(Dataset):
    def __init__(self, root_dir, datafile, transform=None, selected=None):
        data = pd.read_csv(os.path.join(root_dir, datafile), index_col=0)
        self.data = data[data.masks == True]
        if selected is not None:
            mask = self.data.fruit.map(lambda x: np.any([fruit in x for fruit in selected]))
            self.data = self.data[mask]
        self.names = self.data.names.values
        self.names.sort()
        self.transform = transform 
        self.root_dir = root_dir
        self.class_names = get_class_names()

        self.num_classes = len(self.class_names)
    
        
    def __len__(self):
        return len(self.data)
    
    def read_image_label(self, idx):
        imagefile = self.root_dir + 'visible_28/' + self.names[idx] + '.npy'
        image = np.load(imagefile) # 28 ch
        labelfile = self.root_dir +  'labels/' + self.names[idx] + '.png'
        label = io.imread(labelfile)
        return image, label

    def __getitem__(self, idx):
        image, label_rgb = self.read_image_label(idx)
        label = encode_segmap(label_rgb)
        sample = {'image': image, 'label': label}    
        return self.transform(sample) if self.transform else sample
    
    
def prep_loaders(root_dir, batch_size=1, workers=1):
    # Load dataset
    train_dataset = HySpecSegmentation(
        root_dir=root_dir, 
        datafile='train_data.csv', 
        transform=transforms.Compose([RandomCropHoriz(),SegIdentityTransform(), RandomHorizontalFlip()])
    )
    test_dataset = HySpecSegmentation(
        root_dir=root_dir, 
        datafile='test_data.csv', 
        transform=transforms.Compose([SegIdentityTransform()])
    )

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=workers)
    print('Dataset size (num. batches)', len(train_loader), len(valid_loader))
    return train_loader, valid_loader

def prep_loaders_ddp(root_dir, batch_size=1, workers=1, rank=0, world_size=1):
    # Load dataset
    train_dataset = HySpecSegmentation(
        root_dir=root_dir, 
        datafile='train_data.csv', 
        transform=transforms.Compose([RandomCropHoriz(),SegIdentityTransform(), RandomHorizontalFlip()])
    )
    test_dataset = HySpecSegmentation(
        root_dir=root_dir,
        datafile='test_data.csv',
        transform=transforms.Compose([SegIdentityTransform()])
    )

    # Prepare data loaders
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, sampler=sampler)
    valid_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=workers)
    print('Dataset size (num. batches)', len(train_loader), len(valid_loader))
    return train_loader, valid_loader