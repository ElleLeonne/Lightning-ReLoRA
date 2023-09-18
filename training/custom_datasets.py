
import datasets
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class BasicDataset(Dataset):
    """ A basic dataset class that implements automatic sharding behavior, for when system memory is a concern. """
    def __init__(self, dataset, length = None, shards: int = 1):
        if shards < 1:
            raise Exception(f"Shard number must be a positive integer, but got {shards} instead.")
        self.dataset = dataset if shards == 1 else datasets.concatenate_datasets([dataset.shard(shards, i, contiguous=True) for i in range(shards)])
        self.length = length if length is not None else len(self.dataset)
    
    @classmethod
    def load_dataset(cls, dataset, split: str = None, load_as_custom_dataset = True):
        """
        Returns this Dataset object from a url string, or a dictionary.

        Parameters:
        - dataset: A url string or dictionary to load the dataset as.
        - load_as_custom_dataset: Whether to wrap the dataset in this class, or to just return it as an HF Arrow Table.
        """
        if not isinstance(dataset, (str, dict)):
            raise Exception(f"Input 'dataset' must be a str, or dict object, got {type(dataset)}.")
        dataset = datasets.load_dataset(dataset, split=split) if isinstance(dataset, str) else datasets.Dataset.from_dict(dataset) if isinstance(dataset, dict) else dataset
        return cls(dataset) if load_as_custom_dataset is True else dataset
    
    @classmethod
    def create_splits(cls, dataset, length: float, split_to_load: str = None):
        """
        Automatically splits a dataset into train and eval sets.
        
        Parameters:
        - dataset: The dataset in question (or a string to the dataset on HuggingFace Hub, or a dictionary)
        - length: A floating point number between 0.0 and 1.0, that decides what percent of the dataset belongs in the first returned dataset.
        """
        if not isinstance(dataset, (datasets.Dataset, datasets.arrow_dataset.Dataset, Dataset, str, dict)):
            raise Exception(f"Input 'dataset' must be a Dataset, str, or dict object, got {type(dataset)}.")
        if not 0.0 < length < 1.0:
            raise Exception(f"Input 'length' must be between 0.0 and 1.0, got {length}.")
        
        if isinstance(dataset, (str, dict)): # Make sure we're working with a dataset object.
            dataset = cls.load_dataset(dataset, split_to_load, False)

        dset_len = len(dataset)
        train_len = int(dset_len * length)
        return cls(dataset.select(range(train_len))), cls(dataset.select(range(train_len, dset_len)))

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

class BasicLMDataset(BasicDataset):
    """ A basic Language Modeling dataset for tokenizing and preparing text inputs. """
    def __init__(self, dataset, tokenizer, length = None, shards: int = 1):
        super().__init__(dataset, length, shards)
        self.tokenizer = tokenizer
    
    def preprocess(self, input):
        pass

    def __getitem__(self, index):
        pass
