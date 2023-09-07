from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class BasicImageDataset(Dataset):
    """ A basic custom dataset class that includes a preprocessing function, to handle image manipulation during training. """
    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length
        self.to_tensor = transforms.ToTensor()
        self.resize_dims = (512, 512)

    def preprocess(self, image):
        image = image.convert("RGB")
        image = image.resize(self.resize_dims, Image.Resampling.BILINEAR)
        image = self.to_tensor(image)
        return image

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image_dict = self.dataset[index]
        image_dict["image"] = self.preprocess(image_dict["image"])
        return image_dict
