import data_preparation

from data_preparation import (
    load_data, map_strata, training_splits, transformations,
    make_dataset, make_dataloader, validate_eopch, plot_loss_curves,
    resnet18, ResNet18_Weights, nn, torch,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    ROOT, cfg
)
from torch.utils.data import DataLoader

# Reconstructed from the completed training run (2026-09-06 17:37:38-18:00:09).
# Val loss/accuracy are exact, pulled from src/logs/train.log. Train loss was
# never logged during training, so these values are read off the existing
# loss_curves.png chart by eye -- close, but an approximation, not exact.
_TRAINING_RESULTS = {
    "frozen epoch 1 (pct ac, train avg loss, val avg loss)":   (0.90, 0.595, 0.32),
    "frozen epoch 2 (pct ac, train avg loss, val avg loss)":   (0.92, 0.335, 0.24),
    "frozen epoch 3 (pct ac, train avg loss, val avg loss)":   (0.92, 0.295, 0.23),
    "unfrozen epoch 1 (pct ac, train avg loss, val avg loss)": (0.94, 0.205, 0.20),
    "unfrozen epoch 2 (pct ac, train avg loss, val avg loss)": (0.97, 0.115, 0.08),
    "unfrozen epoch 3 (pct ac, train avg loss, val avg loss)": (0.97, 0.085, 0.09),
    "unfrozen epoch 4 (pct ac, train avg loss, val avg loss)": (0.97, 0.065, 0.07),
    "unfrozen epoch 5 (pct ac, train avg loss, val avg loss)": (0.97, 0.055, 0.10),
}

if __name__ == "__main__":

    data_preparation.seed = cfg["values"].getint("rseed")

    ds = load_data(cfg['paths']['dir'])
    data_indices = map_strata(ds)
    x_train, x_val, x_test = training_splits(data_indices)

    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    rnet_mean, rnet_std = preprocess.mean, preprocess.std
    test_trans = transformations(_is_training=False, _mean=rnet_mean, _std=rnet_std)
    ds_test = make_dataset(x_test, test_trans)
    test_dataloader = DataLoader(ds_test, batch_size=cfg["values"].getint("batch_size"), shuffle=False, num_workers=0)


    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to('mps')
    model.fc = nn.Linear(model.fc.in_features, 10).to('mps')
    model.load_state_dict(torch.load("best_model.pt"))

    test_pct_accuracy, test_avg_loss, test_predictions, test_labels = validate_eopch(
        _model=model, _dataloader=test_dataloader, _return_preds=True
    )

    c_matrix = confusion_matrix(test_labels.cpu().numpy(), test_predictions.cpu().numpy(), normalize='true')
    import matplotlib.pyplot as plt
    disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=ds.classes)
    disp.plot(values_format='.2f', xticks_rotation='vertical')
    results_dir = ROOT.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "confusion_matrix.png", bbox_inches='tight')
    plt.close()

    plot_loss_curves(_TRAINING_RESULTS, _freeze_epoch=3, _highlight_epochs=[7])
