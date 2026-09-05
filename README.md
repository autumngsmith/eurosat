# Fine-tune a CNN on EuroSAT Images
#### _An implementation of a deep learning approach for land use classification from satellite imagery, aiming to reproduce current best-practice methods while building a foundation for further sensor modeling work. Built on a PyTorch ResNet18 backbone and trained on the EuroSAT Sentinel-2 dataset. The training pipeline was implemented independently to gain a deeper working understanding of the full workflow._

## Project Layout

``` bash
eurosat/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data.py       split + transforms + dataloaders
│   ├── model.py      resnet18 setup
│   └── train.py      the training loop
├── notebooks/
└── data/             gitignored 
```