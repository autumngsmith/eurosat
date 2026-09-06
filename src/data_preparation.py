"""Load EuroSAT dataset and prepare training, validation, and testing splits."""

from torchvision.datasets import EuroSAT
from torchvision.transforms import v2 as transforms
import matplotlib.pyplot as plt
import random
import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import configparser
from pathlib import Path
import logging
import torch
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torch import nn

ROOT = Path(__file__).resolve().parent

# read config file
cfg = configparser.ConfigParser()
cfg.read(ROOT / "config.cfg")

# set up logging
logger = logging.getLogger(__name__)

def setup_logging(level=logging.DEBUG, logfile=ROOT / "logs" / "train.log"):
    """Logging setup, debug"""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logfile,
        level=level,
        format="%(asctime)s %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

# Classes

class EuroSAT_dataset(Dataset):
    """
    Template taken directly from pytorch documentation on the Dataset custom class implementation
    We are not reading live data, so this custom dataset template is perfered.
    """
    def __init__(self, _df: pd.DataFrame,  _transform=None, _target_transform=None):
        """
        Args:
            _df: dataframe with image files and classes: `path to image, strata id, strata label`
            transform: transformations to apply to the images
            target_transform: apply transformations to the target var (class)
        """
        self.img_labels = _df
        self.transform = _transform
        self.target_transform = _target_transform

    def __len__(self):
        """
        Length of the dataset (num records)"""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Returns a pair of image, label as indiviudal variables    
        `return image, label`
        """
        img_path = self.img_labels.iloc[idx, 0]
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Methods

def load_data(dir: str): 
    "Loads EuroSAT data into the given directory; a folder `data/eurosat/2750` with all of the images will be created in said directory.\nReturns `ds` - the object that includes strata info"
    PATH = os.path.join(cfg['paths']['dir'], 'data')
    _ds = EuroSAT(root=PATH, download=True)
    logger.debug(f"data successfully read to: {PATH}")
    return _ds

def map_strata(_ds: EuroSAT):
    _strata = dict(enumerate(_ds.classes))
    _cols = ["file_path", "strata_id"]
    _data_indices = pd.DataFrame(_ds.samples, columns=_cols)
    _data_indices["strata_name"] = _data_indices["strata_id"].map(_strata)
    logger.debug("strata id to strata name mapping successful")
    return _data_indices

def training_splits(_df: pd.DataFrame):

    _x_train, _val_test = train_test_split(
        _df, 
        test_size=0.3, 
        stratify=_df['strata_id'], 
        random_state=seed
        )
    logger.debug("x_train and val_test created, creating x_val and x_test")

    _x_val, _x_test = train_test_split(
        _val_test, 
        test_size=0.5, 
        stratify=_val_test['strata_id'], 
        random_state=seed
        )
    logger.debug(f"The shape of \tx_train: {_x_train.shape}\n\t\t\t\t\t\t\t\t\t\tx_val:   {_x_val.shape}\n\t\t\t\t\t\t\t\t\t\t\t\tx_test:  {_x_test.shape}")

    return _x_train, _x_val, _x_test

def code_debug_sample(_df: pd.DataFrame):
    """
    makes a dataset for sake of debugging rather than always running a full training size dataset while building code
    """

    _x_train, remainder_1 = train_test_split(
        _df, 
        test_size=0.98, 
        stratify=_df['strata_id'], 
        random_state=seed
        )

    _x_val, remainder_2 = train_test_split(
        remainder_1, 
        test_size=0.99, 
        stratify=remainder_1['strata_id'], 
        random_state=seed
        )

    _x_test, remainder_3 = train_test_split(
            remainder_2, 
            test_size=0.99, 
            stratify=remainder_2['strata_id'], 
            random_state=seed
            )
 

    return _x_train, _x_val, _x_test 
    

def transformations(_is_training: bool, _mean: list, _std: list):
    """Create a transformation object

    Args: 
        _is_training: if the transformations should be for training images
        _mean: list of the means to normalize to (from ImageNet)
        _std: list of the standard devs to normalize to (from ImageNet)

    """
    logger.debug(f"training a training dataset: \t{_is_training}")
    _all_trans = [transforms.Resize(224), # takes an image or a tensor
            transforms.ToDtype(torch.float32, scale=True), # converts raw pixels [0, 255] to floats bound between 0 and 1
            transforms.Normalize(_mean, _std)] # operates on a tensor of floats)
    _train_trans = [
            transforms.RandomHorizontalFlip(p=0.5), # image or tensor input
            transforms.RandomVerticalFlip(p=0.5)]

    if _is_training:
        _trans = _all_trans + _train_trans
    else: 
        _trans = _all_trans

    return transforms.Compose(_trans)

def make_dataset(_data_indices, _trans):
    """
    Make datasets to be passed to dataloader.   
    returns type Dataset
    Args: 
        _data_indices: the indices belonging to a given dataset
        _trans: The composed transformations
    """
    logger.debug(f"making the dataset with the following transformations: {_trans}")
    _dataset = EuroSAT_dataset(_data_indices, _trans)
    logger.debug(f"Number of records: \t{len(_dataset)}")
    return _dataset

def make_dataloader(_dataset: EuroSAT_dataset, _shuffle: bool):
    """
    Create dataloader given a dataset
    """
    _dataloader = DataLoader(_dataset, batch_size=cfg["values"].getint("batch_size"), shuffle=_shuffle)
    logger.debug(f"DataLoader complete, length: {len(_dataloader)}")
    return _dataloader

def freeze_layers(_model, _freeze: bool):
    """
    Freeze or unfreeze mmodel layers (set requires_grad parameter to false for all model layers)
    Args:
        _freeze: True -> model will freeze; False -> model not frozen
    """
    # every layers weights are nn.parameter tensors
    # each one has a requires_grad = False attribute
    # loop over model.parameters() to freeze each layer

    for p in _model.parameters():
        p.requires_grad = not _freeze
    
    return _model

def train_epoch(_model, _dataloader, _is_frozen):
    """
    run training over one epoch to do:    
        * forward pass
        * loss
        * backward pass
        * update
    """
    _loss_fxn = nn.CrossEntropyLoss()

    if _is_frozen:
        _optimizer = torch.optim.Adam(_model.fc.parameters())
    else:
        _optimizer = torch.optim.Adam(_model.parameters(), lr=1e-4)

    for _images, _labels in _dataloader:
        # clears the gradients
        _optimizer.zero_grad()

        # per batch in epoch
        _logits = _model(_images)
        
        # loss fxn
        _loss = _loss_fxn(_logits, _labels)

        # backward pass
        _loss.backward()

        # update
        _optimizer.step()

def validate_eopch(_model, _dataloader):
    """
    Validate the training results from an epoch
    Returns: 
        - pct accuracy
        - avg loss
    """
    _loss_fxn = nn.CrossEntropyLoss()
    _model.eval() # enter evaluation mode 

    with torch.no_grad():
        _num_correct = 0
        _total_loss = 0
        for _images, _labels in _dataloader:
            _logits = _model(_images)
            _total_loss += _loss_fxn(_logits, _labels).item()    
            _preds = _logits.argmax(dim=1)
            _num_correct += (_preds == _labels).sum().item() # .item() helps return a number and not a tensor

        _pct_accuracy = _num_correct / len(_dataloader.dataset)
        _avg_loss = _total_loss / len(_dataloader)
        logger.info(f"\taccuracy of epoch: {round(_pct_accuracy, 2)}")
        logger.info(f"\tavg loss of epoch: {round(_avg_loss, 2)}")

    _model.train() # exit evaluation mode

    return _pct_accuracy, _avg_loss

if __name__ == "__main__":
    setup_logging()
    logger.debug("START: Logging set up")
    logger.debug(f"The root dir for this script is:\t{ROOT}")

    seed = cfg["values"].getint("rseed")
    mode = cfg["modes"]["mode"]

    logger.debug("Beginning data load.")
    ds = load_data(cfg['paths']['dir'])

    logger.debug("Mapping strata labels to file paths")
    data_indices = map_strata(ds)

    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    rnet_mean, rnet_std = preprocess.mean, preprocess.std

    logger.debug(f"mode: {mode} Randomly assign training, val, test indices")

    if mode == 'debug':
        # debug datasets
        x_train, x_val, x_test = code_debug_sample(data_indices)

    else: 
        x_train, x_val, x_test = training_splits(data_indices)

    ## Training dataset
    logger.debug(f"create training transformations for ResNet~~~~~~~~~~~~~~~~~~~~~~~~")
    train_trans = transformations(_is_training=True, _mean=rnet_mean, _std=rnet_std)

    # make training dataset and dataloader
    ds_train = make_dataset(x_train, train_trans)
    train_dataloader = make_dataloader(ds_train, True)

    ## Validation dataset
    logger.debug(f"\n\ncreate validation transformations for ResNet~~~~~~~~~~~~~~~~~~~~~~~~")
    val_trans = transformations(_is_training=False, _mean=rnet_mean, _std=rnet_std)

    # make validation dataset & data loader
    ds_val = make_dataset(x_val, val_trans)
    val_dataloader = make_dataloader(_dataset=ds_val, _shuffle=False)

    # Load Resnet18 Model
    logger.debug(f"removing last layer of `ResNet18` model")
    model18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # freeze backbone by freezing entire model
    frozen_model = freeze_layers(_model=model18, _freeze=True)

    # replace final layer with 10 catorgies
    frozen_model.fc = nn.Linear(model18.fc.in_features,10)

    ## Debug Training
    logger.debug(f"begin training process")
    # 3 epochs with frozen backbone
    logger.info(f"num records being processed in training loop: {len(x_train)}")

    results = {}
    for epoch in range(3):
        logger.debug(f"frozen; started epoch:\t{epoch+1}")
        train_epoch(_model=frozen_model, _dataloader=train_dataloader, _is_frozen=True)
        logger.debug(f"finished training epoch:\t{epoch+1}")
        pct_accuracy, avg_loss = validate_eopch(_model=frozen_model, _dataloader=val_dataloader)
        results[f"frozen epoch {epoch+1}"] = (pct_accuracy, avg_loss)

    # 5 epochs with unfrozen backbone
    unfrozen_model = freeze_layers(_model=frozen_model, _freeze=False)
    for epoch in range(5):
        logger.debug(f"unfrozen; started epoch:\t{epoch+1}")
        train_epoch(_model=unfrozen_model, _dataloader=train_dataloader, _is_frozen=False)
        logger.debug(f"finished training epoch:\t{epoch+1}")
        pct_accuracy, avg_loss = validate_eopch(_model=unfrozen_model, _dataloader=val_dataloader)
        results[f"unfrozen epoch {epoch+1}"] = (pct_accuracy, avg_loss)

    # ## Test dataset
    # logger.debug(f"\n\ncreate test transformations for ResNet~~~~~~~~~~~~~~~~~~~~~~~~")
    # test_trans = transformations(_is_training=False, _mean=rnet_mean, _std=rnet_std)

    # # make test dataset and dataloader
    # ds_test = make_dataset(x_test, test_trans)
    # test_dataloader = make_dataloader(_dataset=ds_test, _shuffle=False)

    logger.debug(f"End process.\n\n")
    
