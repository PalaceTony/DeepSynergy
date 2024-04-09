import numpy as np
import pandas as pd
import pickle
import gzip
from sklearn.model_selection import train_test_split


def normalize(X, means1=None, std1=None, norm="tanh_norm"):
    X = np.ascontiguousarray(X)
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if means1 is None:
        means1 = np.nanmean(X, axis=0)
    std1_corrected = np.where(std1 == 0, np.finfo(float).eps, std1)
    X_normalized = (X - means1) / std1_corrected
    return X_normalized, means1, std1


def process_data(split_ratios=(0.7, 0.15, 0.15), norm="tanh"):
    file = gzip.open(
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/raw/X.p.gz", "rb"
    )
    X = pickle.load(file)
    file.close()

    labels = pd.read_csv(
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/raw/labels.csv", index_col=0
    )
    labels = pd.concat([labels, labels])

    # Calculate feature counts and indices for slicing, similar to the original function
    drug_A_feature_count = 1309 + 802 + 2276
    drug_B_feature_count = drug_A_feature_count
    cell_line_feature_count = 3984
    drug_A_end = drug_A_feature_count
    drug_B_start = drug_A_end
    drug_B_end = drug_B_start + drug_B_feature_count
    cell_line_start = drug_B_end

    # Extract features
    y = labels["synergy"].values

    # Randomly split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=split_ratios[1] + split_ratios[2], random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
        random_state=42,
    )

    # Normalize each set using training set statistics
    X_train_drug_A, mean_drug_A, std_drug_A = normalize(
        X_train[:, 0:drug_A_end], norm=norm
    )
    X_train_drug_B, mean_drug_B, std_drug_B = normalize(
        X_train[:, drug_B_start:drug_B_end], norm=norm
    )
    X_train_cell_line, mean_cell_line, std_cell_line = normalize(
        X_train[:, cell_line_start:], norm=norm
    )

    # Normalize validation and test sets using the means and stds from the training set
    X_val_drug_A, _, _ = normalize(
        X_val[:, 0:drug_A_end], mean_drug_A, std_drug_A, norm=norm
    )
    X_val_drug_B, _, _ = normalize(
        X_val[:, drug_B_start:drug_B_end], mean_drug_B, std_drug_B, norm=norm
    )
    X_val_cell_line, _, _ = normalize(
        X_val[:, cell_line_start:], mean_cell_line, std_cell_line, norm=norm
    )

    X_test_drug_A, _, _ = normalize(
        X_test[:, 0:drug_A_end], mean_drug_A, std_drug_A, norm=norm
    )
    X_test_drug_B, _, _ = normalize(
        X_test[:, drug_B_start:drug_B_end], mean_drug_B, std_drug_B, norm=norm
    )
    X_test_cell_line, _, _ = normalize(
        X_test[:, cell_line_start:], mean_cell_line, std_cell_line, norm=norm
    )

    # Save
    save_path = (
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/data_random_split_%s.p" % norm
    )
    with open(save_path, "wb") as f:
        pickle.dump(
            (
                X_train_drug_A,
                X_train_drug_B,
                X_train_cell_line,
                X_val_drug_A,
                X_val_drug_B,
                X_val_cell_line,
                X_test_drug_A,
                X_test_drug_B,
                X_test_cell_line,
                y_train,
                y_val,
                y_test,
            ),
            f,
        )

    print(f"Data processed with random split and saved to {save_path}")


# Process the data with random split
process_data(norm="tanh")
