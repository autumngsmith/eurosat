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
from torchvision.models import ResNet18_Weights


# set up logging
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

def setup_logging(level=logging.DEBUG, logfile=ROOT / "logs" / "train.log"):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logfile,
        level=level,
        format="%(asctime)s %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

# read config file
cfg = configparser.ConfigParser()
cfg.read(ROOT / "config.cfg")

# define methods
def get_classes(): 
    return True

def load_data(dir: str): 
    "Loads EuroSAT data into the given directory; a folder `data/eurosat/2750` with all of the images will be created in said directory.\nReturns `ds` - the object that includes strata info"
    try:
        PATH = os.path.join(cfg['paths']['dir'], 'data')
        _ds = EuroSAT(root=PATH, download=True)
        logging.debug(f"data successfully read to: {PATH}")
        return _ds
    except Exception as e:
        logger.debug(f"Data could not be read, the error is as follows:\n\t{e}")

def map_strata(_ds: EuroSAT):
    try:
        _strata = dict(enumerate(_ds.classes))
        _cols = ["file_path", "strata_id"]
        _data_indices = pd.DataFrame(_ds.samples, columns=_cols)
        _data_indices["strata_name"] = _data_indices["strata_id"].map(_strata)
        logger.debug("strata id to strata name mapping successful")
        return _data_indices
    except Exception as e:
        logger.debug(f"Strata could not be mapped, the error is as follows:\n\t{e}")
        return None

def training_splits(_df: pd.DataFrame):
    try:
        x_train, val_test = train_test_split(
            _df, 
            test_size=0.3, 
            stratify=_df['strata_id'], 
            random_state=seed
            )
        logging.debug("x_train and val_test created, creating x_val and x_test")

        x_val, x_test = val_test = train_test_split(
            val_test, 
            test_size=0.5, 
            stratify=val_test['strata_id'], 
            random_state=seed
            )
        logging.debug(f"The shape of \tx_train: {x_train.shape}\n\t\t\t\t\t\t\t\t\t\tx_val:   {x_val.shape}\n\t\t\t\t\t\t\t\t\t\t\t\tx_test:  {x_test.shape}")

        return x_train, x_val, x_test

    except Exception as e: 
        logging.debug(f"x_train, x_val, and x_test were not created: \n\t{e}")
        return None

def transformations(_is_training: bool, _mean: list, _std: list):
    """Create a transformation object

    Args: 
        _is_training: if the transformations should be for training images
        _mean: list of the means to normalize to (from ImageNet)
        _std: list of the standard devs to normalize to (from ImageNet)

    """
    logging.debug(f"training a training dataset: \t{_is_training}")
    _all_trans = [transforms.Resize(224), # takes an image
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(_mean, _std)] # operates on a tensor of floats)
    _train_trans = [
            transforms.RandomHorizontalFlip(p=0.5), # image or tensor input
            transforms.RandomVerticalFlip(p=0.5)]

    if _is_training:
        _trans = _all_trans + _train_trans
    else: 
        _trans = _all_trans

    logging.debug(f"the transformations are:\t\t\t\t\t{_trans}")
    return transforms.Compose(_trans)

if __name__ == "__main__":
    setup_logging()
    logger.debug("START: Logging set up")
    logger.debug(f"The root dir for this script is:\t{ROOT}")

    seed = cfg["values"].getint("rseed")

    logger.debug("Beginning data load.")
    ds = load_data(cfg['paths']['dir'])

    logger.debug("Mapping strata labels to file paths")
    data_indices = map_strata(ds)

    logger.debug(f"Randomly assign training, val, test indices")
    x_train, x_val, x_test = training_splits(data_indices)

    logger.debug(f"create training transformations for ResNet")
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    rnet_mean, rnet_std = preprocess.mean, preprocess.std
    trans = transformations(True, rnet_mean, rnet_std)

    logger.debug("run transformations on image")
    
    tst_img = ds.loader(ds.samples[3][0])
    tst_out = trans(tst_img)
    logger.debug(f"shape of output:\t\t\t\t\t\t\t{tst_out.shape}")

    logging.debug(f"End process.\n\n")
    


# def load_dataset(root: str, train: bool = False):
#     """Return the EuroSAT dataset, with train or eval transforms."""
#     transform = _build_transform(train=train)
#     return EuroSAT(root=root, transform=transform, download=False)


# def _build_transform(train: bool):
#     ...


# def main(epochs: int, lr: float):
#     ...


# if __name__ == "__main__":
#     main(epochs=5, lr=1e-4)
