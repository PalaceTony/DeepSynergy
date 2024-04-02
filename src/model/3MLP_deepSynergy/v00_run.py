import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from ThreeMLPdrugSynergyModel import ThreeMLPdrugSynergyModel
from v10_trainer import Trainer
from lib.dataloader import get_dataloader
from lib.utils import set_seed
from lib.utils import configure_logging

parser = argparse.ArgumentParser(description="Parser for DC prediction.")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--data_file",
    type=str,
    default="/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/3mlp_data_test_fold1_tanh.p",
    help="Path to the data",
)
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument(
    "--dsn1_layers",
    type=int,
    default=[2048, 4096, 2048],
    help="DSN1 layer sizes",
)
parser.add_argument(
    "--dsn2_layers",
    type=int,
    default=[2048, 4096, 2048],
    help="DSN2 layer sizes",
)
parser.add_argument(
    "--cln_layers",
    type=int,
    default=[1024, 2048, 1024],
    help="CLN layer sizes",
)
parser.add_argument(
    "--spn_layers", nargs="+", type=int, default=[2048, 1024], help="SPN layer sizes"
)
parser.add_argument("--best_path", type=str, help="Saved model")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
parser.add_argument(
    "--early_stop_patience", type=int, default=100, help="Early stopping"
)
parser.add_argument("--output_dir", type=str, default="outputs", help="outputs")


def main():
    args = parser.parse_args()
    configure_logging(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    (
        train_loader,
        val_loader,
        test_loader,
        _,
        drug_A_feature_shape,
        drug_B_feature_shape,
        cell_line_feature,
    ) = get_dataloader(args, device)

    model = ThreeMLPdrugSynergyModel(
        args.dsn1_layers,
        args.dsn2_layers,
        args.cln_layers,
        args.spn_layers,
        drug_A_feature_shape,
        drug_B_feature_shape,
        cell_line_feature,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.5)
    criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        logging.getLogger("3MLPdeepSynergy"),
        args,
        train_loader,
        val_loader,
        test_loader,
    )
    if args.best_path:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
