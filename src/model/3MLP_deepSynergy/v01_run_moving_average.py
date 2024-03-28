import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np


from ThreeMLPdrugSynergyModel import ThreeMLPdrugSynergyModel
from v11_trainer_moving_average import Trainer
from lib.dataloader import get_dataloader
from lib.utils import set_seed
from lib.utils import moving_average


@hydra.main(version_base="1.1", config_path="../../conf/model", config_name="3mlp")
def main(cfg: DictConfig):
    # Set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # Load data
    (
        train_loader,
        val_loader,
        test_loader,
        final_train_loader,
        drug_A_feature_shape,
        drug_B_feature_shape,
        cell_line_feature,
    ) = get_dataloader(cfg, device)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("3MLPdeepSynergy")

    model = ThreeMLPdrugSynergyModel(
        cfg.model.dsn1_layers,
        cfg.model.dsn2_layers,
        cfg.model.cln_layers,
        cfg.model.spn_layers,
        drug_A_feature_shape,
        drug_B_feature_shape,
        cell_line_feature,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.model.learning_rate, momentum=0.5)
    criterion = nn.MSELoss()

    epo = cfg.training.epochs
    train_flag = "first_train"
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        logger,
        cfg,
        train_loader,
        val_loader,
        test_loader,
        epo,
        train_flag,
    )
    if cfg.model.best_path is not None:
        trainer.test()
    else:
        val_losses = trainer.train()
        average_over = cfg.training.early_stopping_average_over
        mov_av = moving_average(np.array(val_losses), average_over)
        smooth_val_loss = np.pad(
            mov_av, (average_over // 2, average_over // 2), mode="edge"
        )
        epo = np.argmin(smooth_val_loss)
        train_flag = "second_train"
        trainer = Trainer(
            model,
            criterion,
            optimizer,
            logger,
            cfg,
            final_train_loader,
            val_loader,
            test_loader,
            epo,
            train_flag,
        )
        trainer.train()


if __name__ == "__main__":
    main()
