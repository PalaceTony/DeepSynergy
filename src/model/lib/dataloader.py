from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import torch


def get_dataloader(args, device):
    with open(args.data_file, "rb") as file:
        (
            X_tr_drug_A,
            X_tr_drug_B,
            X_tr_cell_line,
            X_val_drug_A,
            X_val_drug_B,
            X_val_cell_line,
            X_test_drug_A,
            X_test_drug_B,
            X_test_cell_line,
            y_tr,
            y_val,
            y_test,
        ) = pickle.load(file)

    if args.model == "deepSynergy":
        X_tr_concatenated = np.concatenate(
            (X_tr_drug_A, X_tr_drug_B, X_tr_cell_line), axis=1
        )
        X_val_concatenated = np.concatenate(
            (X_val_drug_A, X_val_drug_B, X_val_cell_line), axis=1
        )
        X_test_concatenated = np.concatenate(
            (X_test_drug_A, X_test_drug_B, X_test_cell_line), axis=1
        )

        X_tr = torch.tensor(X_tr_concatenated, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val_concatenated, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test_concatenated, dtype=torch.float32).to(device)

    elif args.model == "3MLP":
        X_tr_drug_A = torch.tensor(X_tr_drug_A, dtype=torch.float32).to(device)
        X_tr_drug_B = torch.tensor(X_tr_drug_B, dtype=torch.float32).to(device)
        X_tr_cell_line = torch.tensor(X_tr_cell_line, dtype=torch.float32).to(device)

        X_val_drug_A = torch.tensor(X_val_drug_A, dtype=torch.float32).to(device)
        X_val_drug_B = torch.tensor(X_val_drug_B, dtype=torch.float32).to(device)
        X_val_cell_line = torch.tensor(X_val_cell_line, dtype=torch.float32).to(device)

        X_test_drug_A = torch.tensor(X_test_drug_A, dtype=torch.float32).to(device)
        X_test_drug_B = torch.tensor(X_test_drug_B, dtype=torch.float32).to(device)
        X_test_cell_line = torch.tensor(X_test_cell_line, dtype=torch.float32).to(
            device
        )

    elif args.model == "matchMaker":
        X_tr_drug_A_concatenated = np.concatenate((X_tr_drug_A, X_tr_cell_line), axis=1)
        X_val_drug_A_concatenated = np.concatenate(
            (X_val_drug_A, X_val_cell_line), axis=1
        )
        X_test_drug_A_concatenated = np.concatenate(
            (X_test_drug_A, X_test_cell_line), axis=1
        )

        X_tr_drug_B_concatenated = np.concatenate((X_tr_drug_B, X_tr_cell_line), axis=1)
        X_val_drug_B_concatenated = np.concatenate(
            (X_val_drug_B, X_val_cell_line), axis=1
        )
        X_test_drug_B_concatenated = np.concatenate(
            (X_test_drug_B, X_test_cell_line), axis=1
        )

        X_tr_drug_A_cell = torch.tensor(
            X_tr_drug_A_concatenated, dtype=torch.float32
        ).to(device)
        X_val_drug_A_cell = torch.tensor(
            X_val_drug_A_concatenated, dtype=torch.float32
        ).to(device)
        X_test_drug_A_cell = torch.tensor(
            X_test_drug_A_concatenated, dtype=torch.float32
        ).to(device)

        X_tr_drug_B_cell = torch.tensor(
            X_tr_drug_B_concatenated, dtype=torch.float32
        ).to(device)
        X_val_drug_B_cell = torch.tensor(
            X_val_drug_B_concatenated, dtype=torch.float32
        ).to(device)
        X_test_drug_B_cell = torch.tensor(
            X_test_drug_B_concatenated, dtype=torch.float32
        ).to(device)

    y_tr = torch.tensor(y_tr, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    if args.model == "deepSynergy":
        train_dataset = TensorDataset(X_tr, y_tr)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
    elif args.model == "3MLP":
        train_dataset = TensorDataset(X_tr_drug_A, X_tr_drug_B, X_tr_cell_line, y_tr)
        val_dataset = TensorDataset(X_val_drug_A, X_val_drug_B, X_val_cell_line, y_val)
        test_dataset = TensorDataset(
            X_test_drug_A, X_test_drug_B, X_test_cell_line, y_test
        )
    elif args.model == "matchMaker":
        train_dataset = TensorDataset(X_tr_drug_A_cell, X_tr_drug_B_cell, y_tr)
        val_dataset = TensorDataset(X_val_drug_A_cell, X_val_drug_B_cell, y_val)
        test_dataset = TensorDataset(X_test_drug_A_cell, X_test_drug_B_cell, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == "deepSynergy":
        return train_loader, val_loader, test_loader, X_tr.shape[1]
    elif args.model == "3MLP":
        return (
            train_loader,
            val_loader,
            test_loader,
            X_tr_drug_A.shape[1],
            X_tr_drug_B.shape[1],
            X_tr_cell_line.shape[1],
        )
    elif args.model == "matchMaker":
        return (
            train_loader,
            val_loader,
            test_loader,
            X_tr_drug_A_cell.shape[1],
            X_tr_drug_B_cell.shape[1],
        )
