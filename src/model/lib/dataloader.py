from torch.utils.data import DataLoader, TensorDataset
import gzip
import pickle
import torch


def get_dataloader(cfg, device):
    # Load data
    with gzip.open(cfg.data.file, "rb") as file:
        X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)

    # Convert numpy arrays to torch tensors and move them to the selected device
    X_tr, y_tr = torch.tensor(X_tr, dtype=torch.float32).to(device), torch.tensor(
        y_tr, dtype=torch.float32
    ).to(device)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(
        y_val, dtype=torch.float32
    ).to(device)
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(
        device
    ), torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(
        y_test, dtype=torch.float32
    ).to(device)

    # Define the data loaders
    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )
    final_train_dataset = TensorDataset(X_train, y_train)
    final_train_loader = DataLoader(
        final_train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )

    return X_tr, train_loader, val_loader, test_loader, final_train_loader
