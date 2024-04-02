import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import logging


from deepSynergy import DeepSynergyModel
from trainer import Trainer
from lib.dataloader import get_dataloader
from lib.utils import set_seed
from lib.utils import configure_logging


parser = argparse.ArgumentParser(description="Parser for DC prediction.")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--data_file",
    type=str,
    default="/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/data_test_fold1_tanh.p",
    help="Path to the data",
)
parser.add_argument(
    "--learning_rate", type=float, default=0.00001, help="Learning rate"
)
parser.add_argument(
    "--layers",
    type=int,
    default=[8182, 4096, 1],
    help="all layers",
)
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
parser.add_argument("--input_dropout", type=float, default=0.2, help="Input dropout")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
parser.add_argument(
    "--early_stop_patience", type=int, default=100, help="Early stopping"
)
parser.add_argument("--best_path", type=str, help="Saved model")
parser.add_argument("--output_dir", type=str, default="outputs", help="base outputs")


def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    X_tr, train_loader, val_loader, test_loader, _ = get_dataloader(args, device)
    model = DeepSynergyModel(
        input_size=X_tr.shape[1],
        layers=args.layers,
        input_dropout=args.input_dropout,
        dropout=args.dropout,
        act_func=torch.nn.functional.relu,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.5)
    criterion = nn.MSELoss()

    # Train and Validate
    logger = configure_logging(args)
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        logger,
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
