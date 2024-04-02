import argparse
import torch
import torch.nn as nn
import torch.optim as optim


from m_deepSynergy import DeepSynergyModel
from m_threeMLP import ThreeMLPdrugSynergyModel
from m_matchMaker import MatchMakerModel
from trainer import Trainer
from lib.dataloader import get_dataloader
from lib.utils import set_seed, configure_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for all models")
    # shared ####################################################################################################
    parser.add_argument("--epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--early_stop_patience", type=int, default=100, help="Early stopping"
    )
    parser.add_argument(
        "--input_dropout", type=float, default=0.2, help="Input dropout"
    )
    parser.add_argument(
        "--hidden_dropout", type=float, default=0.5, help="Hidden dropout"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepSynergy",
        help="which to run: deepSynergy, 3MLP, matchMaker",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/data_test_fold1_tanh.p",
        help="data path",
    )
    parser.add_argument("--best_path", type=str, help="best model")
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="base outputs"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")

    # deepSynergy ####################################################################################################
    parser.add_argument(
        "--deepSynergy_learning_rate", type=float, default=0.00001, help="Learning rate"
    )
    parser.add_argument(
        "--deepSynergy_layers",
        type=int,
        default=[8182, 4096, 1],
        help="all layers",
    )

    # 3MLP ####################################################################################################
    parser.add_argument(
        "--3MLP_learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--3MLP_dsn1_layers",
        type=int,
        default=[2048, 4096, 2048],
        help="drug a layer",
    )
    parser.add_argument(
        "--3MLP_dsn2_layers",
        type=int,
        default=[2048, 4096, 2048],
        help="drug b layer",
    )
    parser.add_argument(
        "--3MLP_cln_layers",
        type=int,
        default=[2048, 4096, 2048],
        help="cell line layer",
    )
    parser.add_argument(
        "--3MLP_spn_layers", type=int, default=[2048, 1024], help="Prediction Layer"
    )

    # matchMaker ###################################################################################################
    parser.add_argument(
        "--matchMaker_learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--matchMaker_dsn1_layers",
        type=int,
        default=[2048, 4096, 2048],
        help="drug a + cell layer",
    )
    parser.add_argument(
        "--matchMaker_dsn2_layers",
        type=int,
        default=[2048, 4096, 2048],
        help="drug b + cell layer",
    )
    parser.add_argument(
        "--matchMaker_spn_layers",
        type=int,
        default=[2048, 1024],
        help="Prediction layer",
    )

    #####################################################################################################
    args = parser.parse_args()

    # Model-specific parameters extraction
    shared_params = {
        k: v
        for k, v in vars(args).items()
        if not k.startswith("deepSynergy_") and not k.startswith("3MLP_")
    }

    args_deepSynergy = argparse.Namespace(
        **{
            k.replace("deepSynergy_", ""): v
            for k, v in vars(args).items()
            if k.startswith("deepSynergy_")
        }
    )
    args_deepSynergy.__dict__.update(shared_params)

    args_3MLP = argparse.Namespace(
        **{
            k.replace("3MLP_", ""): v
            for k, v in vars(args).items()
            if k.startswith("3MLP_")
        }
    )
    args_3MLP.__dict__.update(shared_params)

    args_matchMaker = argparse.Namespace(
        **{
            k.replace("matchMaker_", ""): v
            for k, v in vars(args).items()
            if k.startswith("matchMaker_")
        }
    )
    args_matchMaker.__dict__.update(shared_params)

    return args, args_deepSynergy, args_3MLP, args_matchMaker


def main():
    args, args_deepSynergy, args_3MLP, args_matchMaker = parse_args()

    if args.model == "deepSynergy":
        args = args_deepSynergy
    elif args.model == "3MLP":
        args = args_3MLP
    elif args.model == "matchMaker":
        args = args_matchMaker
    else:
        raise ValueError("Unsupported")

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataloader
    if args.model == "deepSynergy":
        train_loader, val_loader, test_loader, drugs_cell_shape = get_dataloader(
            args, device
        )
    elif args.model == "3MLP":
        (
            train_loader,
            val_loader,
            test_loader,
            drug_A_feature_shape,
            drug_B_feature_shape,
            cell_line_feature_shape,
        ) = get_dataloader(args, device)
    elif args.model == "matchMaker":
        (
            train_loader,
            val_loader,
            test_loader,
            drug_A_feature_cell_shape,
            drug_B_feature_cell_shape,
        ) = get_dataloader(args, device)

    # Model
    if args.model == "deepSynergy":
        model = DeepSynergyModel(layers=args.layers, input_size=drugs_cell_shape).to(
            device
        )
    elif args.model == "3MLP":
        model = ThreeMLPdrugSynergyModel(
            args.dsn1_layers,
            args.dsn2_layers,
            args.cln_layers,
            args.spn_layers,
            drug_A_feature_shape,
            drug_B_feature_shape,
            cell_line_feature_shape,
        ).to(device)
    elif args.model == "matchMaker":
        model = MatchMakerModel(
            args.dsn1_layers,
            args.dsn2_layers,
            args.spn_layers,
            drug_A_feature_cell_shape,
            drug_B_feature_cell_shape,
        ).to(device)

    # Optimizer and Criterion
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.5)
    criterion = nn.MSELoss()

    # Train
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
