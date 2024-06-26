import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval


from m_deepSynergy import DeepSynergyModel
from m_threeMLP import ThreeMLPdrugSynergyModel
from m_matchMaker import MatchMakerModel
from trainer import Trainer
from lib.dataloader import get_dataloader
from lib.utils import set_seed, configure_logging, int_list, print_hopt_space


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for all models")
    # shared ####################################################################################################
    parser.add_argument("--epochs", type=int, default=1000, help="epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--input_dropout", type=float, default=0.2, help="Input dropout"
    )
    parser.add_argument(
        "--hidden_dropout", type=float, default=0.5, help="Hidden dropout"
    )
    parser.add_argument("--hyperopt", type=bool, default=False, help="Hyperopt")
    parser.add_argument("--max_evals", type=int, default=150, help="trial number")
    parser.add_argument(
        "--model",
        type=str,
        default="3MLP",
        help="which to run: deepSynergy, 3MLP, matchMaker",
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=100, help="Early stopping"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/data_test_fold3_tanh.p",
        help="data path",
    )
    parser.add_argument("--best_path", type=str, help="best model")
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="base outputs"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--note", type=str, default="", help="note")

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
        type=int_list,
        default=[2048, 4096, 2048],
        help="drug a layer",
    )
    parser.add_argument(
        "--3MLP_dsn2_layers",
        type=int_list,
        default=[2048, 4096, 2048],
        help="drug b layer",
    )
    parser.add_argument(
        "--3MLP_cln_layers",
        type=int_list,
        default=[1024, 2048, 1024],
        help="cell line layer",
    )
    parser.add_argument(
        "--3MLP_spn_layers",
        type=int_list,
        default=[2048, 1024],
        help="Prediction Layer",
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
        if not k.startswith("deepSynergy_")
        and not k.startswith("3MLP_")
        and not k.startswith("matchMaker_")
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


def run_model(hyperparams):
    data_files = [
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/data_test_fold1_tanh.p",
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/data_test_fold2_tanh.p",
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/data_test_fold3_tanh.p",
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/data_test_fold4_tanh.p",
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/data_test_fold0_tanh.p",
    ]
    losses = []
    for data_file in data_files:
        set_seed(args.seed)
        args.data_file = data_file
        logger.info(f"Data file: {args.data_file}")
        if args.hyperopt:
            logger.info(
                "Hyperopt is updating the hyperparameters based on the space..."
            )
            learning_rate = hyperparams["learning_rate"]
            batch_size = hyperparams["batch_size"]
            dsn1_layers = hyperparams["dsn1_layers"]
            cln_layers = hyperparams["cln_layers"]
            spn_layers = hyperparams["spn_layers"]

            args.learning_rate = learning_rate
            args.batch_size = batch_size
            args.dsn1_layers = dsn1_layers
            args.cln_layers = cln_layers
            args.spn_layers = spn_layers
            logger.info("Done updating hyperparameters based on the space...")

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
            model = DeepSynergyModel(
                layers=args.layers, input_size=drugs_cell_shape
            ).to(device)
        elif args.model == "3MLP":
            args.dsn2_layers = (
                args.dsn1_layers
            )  # 3MLP has the same structure for drug A and drug B

            logger.info(
                f"Learning rate: {args.learning_rate}, Batch size: {args.batch_size}, DSN1: {args.dsn1_layers}, DSN2: {args.dsn2_layers}, CLN: {args.cln_layers}, SPN: {args.spn_layers}"
            )
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
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=50, verbose=True
        )

        # Train
        trainer = Trainer(
            model,
            criterion,
            optimizer,
            logger,
            args,
            train_loader,
            val_loader,
            test_loader,
            scheduler,
        )
        try:
            best_val_loss = trainer.train()
        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
            best_val_loss = 99999999
        logger.info(f"Dataset {args.data_file} completed with loss: {best_val_loss}")
        losses.append(best_val_loss.cpu().numpy())
    logger.info(f"All losses: {losses}")
    best_val_loss = np.mean(losses)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    return {"loss": best_val_loss, "status": STATUS_OK} if args.hyperopt else None


if __name__ == "__main__":
    args, args_deepSynergy, args_3MLP, args_matchMaker = parse_args()
    if args.model == "deepSynergy":
        args = args_deepSynergy
    elif args.model == "3MLP":
        args = args_3MLP
    elif args.model == "matchMaker":
        args = args_matchMaker
    else:
        raise ValueError("Unsupported")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = configure_logging(args)

    if args.hyperopt:
        logger.info("Tuning 3MLP")
        space = {
            "learning_rate": hp.choice(
                "learning_rate",
                [
                    0.01,
                    0.001,
                    0.0001,
                    0.00001,
                    0.03,
                    0.003,
                    0.0003,
                    0.00003,
                    0.05,
                    0.005,
                    0.0005,
                    0.00005,
                ],
            ),
            "batch_size": hp.choice("batch_size", [128, 256, 512, 1024]),
            "dsn1_layers": hp.choice(
                "dsn1_layers",
                [
                    [2048, 4096, 2048],
                    [4096, 8192, 4096],
                ],
            ),
            "cln_layers": hp.choice(
                "cln_layers",
                [
                    [2048, 4096, 2048],
                    [4096, 8192, 4096],
                ],
            ),
            "spn_layers": hp.choice(
                "spn_layers",
                [
                    [1024, 512],
                    [2048, 1024],
                    [4096, 2048],
                ],
            ),
        }

        # Save space
        print_hopt_space(logger, space)

        trials = Trials()
        best_indices = fmin(
            fn=run_model,
            space=space,
            algo=tpe.suggest,
            max_evals=args.max_evals,
            trials=trials,
        )

        for i, trial in enumerate(trials.trials):
            logger.info(
                f"Trial {i+1}: Loss: {trial['result']['loss']}, Params: {trial['misc']['vals']}"
            )
        best_params = space_eval(space, best_indices)
        best_loss = min(trial["result"]["loss"] for trial in trials.trials)
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best loss: {best_loss}")

    else:
        best_val_loss = run_model(args)
