import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle


def get_dataloader(args, device):
    # Load data
    with open(args.data_file, "rb") as file:
        (
            X_tr_drug_A,
            X_tr_drug_B,
            X_tr_cell_line,
            X_val_drug_A,
            X_val_drug_B,
            X_val_cell_line,
            X_train_drug_A,
            X_train_drug_B,
            X_train_cell_line,
            X_test_drug_A,
            X_test_drug_B,
            X_test_cell_line,
            y_tr,
            y_val,
            y_train,
            y_test,
        ) = pickle.load(file)

    # Convert numpy arrays to torch tensors and move them to the selected device
    def to_tensor_and_device(arrays):
        return [torch.tensor(array, dtype=torch.float32).to(device) for array in arrays]

    X_tr_drug_A, X_tr_drug_B, X_tr_cell_line = to_tensor_and_device(
        [X_tr_drug_A, X_tr_drug_B, X_tr_cell_line]
    )
    X_val_drug_A, X_val_drug_B, X_val_cell_line = to_tensor_and_device(
        [X_val_drug_A, X_val_drug_B, X_val_cell_line]
    )
    X_train_drug_A, X_train_drug_B, X_train_cell_line = to_tensor_and_device(
        [X_train_drug_A, X_train_drug_B, X_train_cell_line]
    )
    X_test_drug_A, X_test_drug_B, X_test_cell_line = to_tensor_and_device(
        [X_test_drug_A, X_test_drug_B, X_test_cell_line]
    )
    y_tr, y_val, y_train, y_test = to_tensor_and_device([y_tr, y_val, y_train, y_test])

    # Define the data loaders
    def get_loader(X1, X2, X3, Y, batch_size, shuffle):
        dataset = TensorDataset(X1, X2, X3, Y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = get_loader(
        X_tr_drug_A, X_tr_drug_B, X_tr_cell_line, y_tr, args.batch_size, True
    )
    val_loader = get_loader(
        X_val_drug_A,
        X_val_drug_B,
        X_val_cell_line,
        y_val,
        args.batch_size,
        False,
    )
    test_loader = get_loader(
        X_test_drug_A,
        X_test_drug_B,
        X_test_cell_line,
        y_test,
        args.batch_size,
        False,
    )
    final_train_loader = get_loader(
        X_train_drug_A,
        X_train_drug_B,
        X_train_cell_line,
        y_train,
        args.batch_size,
        True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        final_train_loader,
        X_tr_drug_A.shape[1],
        X_tr_drug_B.shape[1],
        X_tr_cell_line.shape[1],
    )
