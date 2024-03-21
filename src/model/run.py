import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim


from deepSynergy import DeepSynergyModel
from trainer import Trainer
from lib.vis import plot_performance
from lib.dataloader import get_dataloader


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    X_tr, train_loader, val_loader, test_loader, final_train_loader = get_dataloader(
        cfg, device
    )

    model = DeepSynergyModel(
        input_size=X_tr.shape[1],
        layers=cfg.model.layers,
        input_dropout=cfg.model.input_dropout,
        dropout=cfg.model.dropout,
        act_func=torch.nn.functional.relu,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.model.learning_rate, momentum=0.5)
    criterion = nn.MSELoss()

    # Train and Validate for hyperparameter choosing
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        train_loader,
        device,
        cfg,
        val_loader,
    )
    epo, val_losses, smooth_val_loss = trainer.train()

    # Retrain with tain+val data
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        final_train_loader,
        device,
        cfg,
        val_loader=None,
        test_loader=test_loader,
        firstTrain=False,
        epochs=epo,
    )
    trainer.train()
    test_loss = trainer.test()

    # Plot performance
    plot_performance(val_losses, smooth_val_loss, test_loss)


if __name__ == "__main__":
    main()
