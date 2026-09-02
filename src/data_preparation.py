"""Load EuroSAT dataset and prepare training, validation, and testing splits."""

from torchvision.datasets import EuroSAT
import matplotlib.pyplot as plt
import random
import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import configparser
from pathlib import Path
import logging

# set up logging
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

def setup_logging(level=logging.INFO, logfile=ROOT / "logs" / "train.log"):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logfile,
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
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
        logging.info(f"data successfully read to: {PATH}")
        return _ds
    except Exception as e:
        logger.info(f"Data could not be read, the error is as follows:\n\t{e}")

def map_strata(_ds: EuroSAT):
    try:
        _strata = dict(enumerate(_ds.classes))
        _cols = ["file_path", "strata_id"]
        _data_indices = pd.DataFrame(_ds.samples, columns=_cols)
        _data_indices["strata_name"] = _data_indices["strata_id"].map(_strata)
        logger.info("strata id to strata name mapping successful")
        return _data_indices
    except Exception as e:
        logger.info(f"Strata could not be mapped, the error is as follows:\n\t{e}")
        return None
    

def training_splits(_df: pd.DataFrame):
    try:
        x_train, val_test = train_test_split(
            _df, 
            test_size=0.3, 
            stratify=_df['strata_id'], 
            random_state=seed
            )
        logging.info("x_train and val_test created, creating x_val and x_test")

        x_val, x_test = val_test = train_test_split(
            val_test, 
            test_size=0.5, 
            stratify=val_test['strata_id'], 
            random_state=seed
            )
        logging.info(f"The shape of \tx_train: {x_train.shape}\n\t\t\t\t\t\t\t\t\t\tx_val:   {x_val.shape}\n\t\t\t\t\t\t\t\t\t\tx_test:  {x_test.shape}")

        return x_train, x_val, x_test

    except Exception as e: 
        logging.info(f"x_train, x_val, and x_test were not created: \n\t{e}")
        return None

if __name__ == "__main__":
    setup_logging()
    logger.info("START: Logging set up")
    logger.info(f"The root dir for this script is:\t{ROOT}")

    seed = cfg["values"].getint("rseed")

    logger.info("Beginning data load.")
    ds = load_data(cfg['paths']['dir'])

    logger.info("Mapping strata labels to file paths")
    data_indices = map_strata(ds)

    logger.info(f"Randomly assign training, val, test indices")
    x_train, x_val, x_test = training_splits(data_indices)


    logging.info(f"End process.\n\n")
    


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
