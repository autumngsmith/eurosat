# Fine tune a CNN on Eurosat

#### Data pipeline - move to src folder

- [x] Train/val/test split — 70/15/15, stratified by class. Fix the random seed.
- [x] Check class balance. Note any imbalance now so you're not confused by it later.
- [x] `transforms`: 
    - [x] resize to 224 -- in progress
    - [x] normalize with ImageNet mean/std (required for pretrained weights)
    - [x] random horizontal + vertical flip on train only
        - Vertical flip is fine here in a way it isn't for natural images — overhead imagery has no canonical "up." Worth knowing why.
- [ ] `DataLoader` for each split. Batch size 32, `shuffle=True` on train only.


#### Claude setup
 - [ ] Hooks: claude can read the raw data file, but may not write to that file
 - [ ] Claude must edit the TODO.md file before 
 - [ ] have claude set up a study guide about CNN's and the workflow


## Complete

#### Setup 

- [x] New repo, `eurosat`
- [x] Local venv - easier setup and device fine for it ~~or Colab — pick one and don't relitigate it. Colab if you want a free GPU.~~
- [x] `pip install torch torchvision matplotlib scikit-learn`
- [x] Download EuroSAT RGB (27,000 images, 64×64, 10 land-use classes, ~2 GB). Available through `torchvision.datasets.EuroSAT` or from the original release.
- [x] Verify it loads and display a 3×3 grid of images with class labels. **Look at the data before modeling it.**
