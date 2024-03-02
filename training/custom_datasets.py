import os
import datasets
import numpy as np
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
        """ Returns this Dataset object from a url string, or a dictionary.

        Parameters:
        - dataset: A url string or dictionary to load the dataset as.
        - load_as_custom_dataset: Whether to wrap the dataset in this class, or to just return it as an HF Arrow Table. """
        if not isinstance(dataset, (str, dict)):
            raise Exception(f"Input 'dataset' must be a str, or dict object, got {type(dataset)}.")
        dataset = datasets.load_dataset(dataset, split=split) if isinstance(dataset, str) else datasets.Dataset.from_dict(dataset) if isinstance(dataset, dict) else dataset
        return cls(dataset) if load_as_custom_dataset is True else dataset
    
    @classmethod
    def create_splits(cls, dataset, length: float, split_to_load: str = None):
        """ Automatically splits a single dataset into train and eval sets.
        
        Parameters:
        - dataset: The dataset in question (or a string to the dataset on HuggingFace Hub, or a dictionary)
        - length: A floating point number between 0.0 and 1.0, that decides what percent of the dataset belongs in the first returned dataset. """
        if not isinstance(dataset, (datasets.Dataset, datasets.arrow_dataset.Dataset, Dataset, str, dict)):
            raise Exception(f"Input 'dataset' must be a Dataset, str, or dict object, got {type(dataset)}.")
        if not 0.0 < length < 1.0:
            raise Exception(f"Input 'length' must be between 0.0 and 1.0, got {length}.")
        
        if isinstance(dataset, (str, dict)): # Make sure we're working with a dataset object.
            dataset = cls.load_dataset(dataset, split_to_load, False)

        dset_len = len(dataset)
        train_len = int(dset_len * length)
        return cls(dataset.select(range(train_len))), cls(dataset.select(range(train_len, dset_len)))

    def preprocess(self, text_dict):
        print("Please implement the 'preprocess' method in a subclass of this parent class.")
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        print("Please implement the '__getitem__' method in a subclass of this parent class.")
    def __getitems__(self, index):
        print("Please implement the '__getitems__' method in a subclass of this parent class.")
    
class BasicImageDataset(BasicDataset):
    """ An image dataset class that includes a preprocessing method, to handle image transformation and preprocessing during training. """
    def __init__(self, dataset, length = None, shards: int = 1):
        dset = [d["text"] for d in dataset]
        super().__init__(dataset, length, shards)
        self.to_tensor = ToTensor()
        self.resize_dims = (224, 224)

    def preprocess(self, image):
        return self.to_tensor(image.convert("RGB").resize(self.resize_dims, Image.Resampling.BILINEAR))
    def __getitem__(self, index):
        return self.preprocess(self.dataset[index]["image"])
    def __getitems__(self, indexes) -> list: # This accepts a list of samples, and returns a list of arrays.
        return [self.preprocess(self.dataset[index]["image"]) for index in indexes]

class BasicLMDataset(BasicDataset):
    def __init__(self, dataset,
                       tokenizer=None,
                       length = None,
                       shards: int = 1,
                       tokenize_list = False,
                       eos_default_index = 0):
        """ -args:
        
        -- dataset: A pytorch or huggingface dataset object.
        -- tokenizer: Pass a tokenizer if you want to tokenize during preprocessing, otherwise leave as None.
        -- length: We attempt to calculate the length automatically, but providing it can be much faster.
        -- shards: How many shards to split the dataset into, to avoid memory overflow.
        -- tokenize_list: Some tokenizers (those that break w/ multiprocessing) are designed to iterate over lists (batches) automatically.
        -- eos_default_index: _(Only) if you don't include labels_, we will first check tokenizer.eos_token for your end_of_sequence token before using this to produce them.
        """
        super().__init__(dataset, length, shards)
        self.tokenizer = tokenizer  # Provides further room for scaffolding, if other configurations are required.
        self.eos_token = tokenizer.eos_token if (self.tokenizer is not None and hasattr(tokenizer, "eos_token")) else eos_default_index 
        self.tokenize_list = tokenize_list

    def preprocess(self, txt: str):
        """ Tokenizes text, and returns a numpy array. Todo: Support returning pytorch tensors. """
        return self.tokenizer.tokenize(txt).astype(np.int32) if self.tokenizer is not None else txt
    def preprocess_list(self, input_list):
        return [[x.astype(np.int32) for x in x_list]
                for x_list in self.tokenizer.tokenize(input_list)] if self.tokenizer is not None else input_list
    def __getitem__(self, index) -> np.array:
        """ Tokenizes the text, and generates simple labels if they weren't provided. Operates on a single item. """
        input_dict = filter_dict(self.dataset[index])
        return_dict = {"text": self.preprocess(input_dict["text"])}
        return_dict["labels"] = ( np.concatenate((return_dict["text"][1:], [self.eos_token]))
                                  if "labels" not in input_dict else self.preprocess(input_dict["labels"]) )
        return return_dict
    def __getitems__(self, indexes) -> list: # This accepts a list of samples, and returns a list of numpy arrays.
        """ Tokenizes the text, and generates simple labels if they weren't provided. Operates on a batch of items.. """
        input_dict = filter_dict(self.dataset[indexes])
        text_list = self.preprocess_list(input_dict["text"]) if self.tokenize_list is True else [self.preprocess(item) for item in input_dict["text"]]
        labels_list = ( [np.concatenate((txt[1:], [self.eos_token])) for txt in text_list]
                        if "labels" not in input_dict else self.preprocess(input_dict["labels"]) )
        return [{"text": text, "labels": labels} for text, labels in zip(text_list, labels_list)]

# ------ Helper Functions ------
def filter_list(list_of_dicts: list, keys_to_keep: list = ["text", "labels"]) -> list:
    """ Drops anything not in the 'keys to keep' list from a list of dictionaries (a Dataset primitive). """
    return [{k: v for k, v in d.items() if k in keys_to_keep} for d in list_of_dicts]
def filter_dict(dict_of_lists: list, keys_to_keep: list = ["text", "labels"]) -> dict:
    """ Drops anything not in the 'keys to keep' list from a dictionary of lists (a Dataset primitive). """
    return {key: value for key, value in dict_of_lists.items() if key in keys_to_keep}

# ------ Assemble datasets from scratch ------
def dataset_from_chunks(dataset_path: str, split: str, num_chunks: int) -> datasets.Dataset:
    """ Returns a huggingface dataset object from a dataset of custom arrow tables. """
    def generate_paths(dataset_path: str, split: str, num_chunks: int) -> list:
        """ Generates a list of paths pointing to each chunk in a dataset. """
        chunk_list = []
        for i in range(1, num_chunks+1):    # length SHOULD be inferrable
            chunk_list.append(os.path.join(dataset_path, split, "chunk" + str(i)))
            print("Loading chunk" + str(i))
        return chunk_list
    def merge_arrows(path: str, desc=None):
        """ Merges arrow files into a single HF datasets object. """
        arrow_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".arrow")]
        table = []
        max_len = len(arrow_files)
        for i, arrow_file in enumerate(arrow_files):
            table.append(datasets.Dataset.from_file(os.path.join(path, arrow_file)))
            if i % 10 == 0: # Only print multiples of 10 for cleaner debugging.
                if desc != None:
                    print(desc + ", arrow table " + str(i+1)+ " of " + str(max_len) + " loaded.")
                else:
                    print("Arrow table " + str(i+1)+ " of " + str(max_len) + " loaded.")
        return datasets.concatenate_datasets(table)

    chunk_path_list = generate_paths(dataset_path, split, num_chunks)
    dataset_list = []
    for i, path in enumerate(chunk_path_list):
        dataset_list.append(merge_arrows(path, "Chunk"+str(i+1)))
    print("Merging fully loaded list.")
    return datasets.concatenate_datasets(dataset_list)
