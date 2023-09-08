
import datasets
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class BasicDataset(Dataset):
    """ A basic dataset class that implements automatic sharding behavior, for when system memory is a concern. """
    def __init__(self, dataset, length = None, shards: int = 1):
        if shards < 1:
            raise Exception(f"Shard number must be a positive integer, but got {shards} instead.")
        self.dataset = dataset if shards == 1 else datasets.concatenate_datasets([dataset.shard(shards, i, contiguous=True) for i in range(shards)])
        self.length = length if length is not None else len(self.dataset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.dataset[index]
    
class BasicImageDataset(BasicDataset):
    """ An image dataset class that includes a preprocessing method, to handle image transformation and preprocessing during training. """
    def __init__(self, dataset, length = None, shards: int = 1):
        super().__init__(dataset, length, shards)
        self.to_tensor = ToTensor()
        self.resize_dims = (224, 224)

    def preprocess(self, image):
        image = image.convert("RGB")
        image = image.resize(self.resize_dims, Image.Resampling.BILINEAR)
        image = self.to_tensor(image)
        return image
    
    def __getitem__(self, index):
        image_dict = self.dataset[index]
        image_dict["image"] = self.preprocess(image_dict["image"])
        return image_dict
