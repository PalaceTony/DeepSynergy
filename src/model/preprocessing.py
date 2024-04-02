import numpy as np
import pandas as pd
import pickle
import gzip


# Wrap the main process in a function
def process_data_for_folds(test_fold, val_fold, norm="tanh"):
    file = gzip.open(
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/raw_data/X.p.gz", "rb"
    )
    X = pickle.load(file)
    file.close()

    labels = pd.read_csv(
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/raw_data/labels.csv", index_col=0
    )
    labels = pd.concat([labels, labels])

    # Define feature counts
    drug_A_feature_count = 1309 + 802 + 2276
    drug_B_feature_count = 1309 + 802 + 2276
    cell_line_feature_count = 3984

    # Calculate indices for slicing
    drug_A_end = drug_A_feature_count
    drug_B_start = drug_A_end
    drug_B_end = drug_B_start + drug_B_feature_count
    cell_line_start = drug_B_end

    # Define indices for splitting based on the folds
    idx_tr = np.where(
        np.logical_and(labels["fold"] != test_fold, labels["fold"] != val_fold)
    )
    idx_val = np.where(labels["fold"] == val_fold)
    idx_test = np.where(labels["fold"] == test_fold)

    # Split data
    X_tr = X[idx_tr]
    X_val = X[idx_val]
    X_test = X[idx_test]
    y_tr = labels.iloc[idx_tr]["synergy"].values
    y_val = labels.iloc[idx_val]["synergy"].values
    y_test = labels.iloc[idx_test]["synergy"].values

    X_tr_drug_A = X_tr[:, 0:drug_A_end]
    X_tr_drug_B = X_tr[:, drug_B_start:drug_B_end]
    X_tr_cell_line = X_tr[:, cell_line_start:]

    X_val_drug_A = X_val[:, 0:drug_A_end]
    X_val_drug_B = X_val[:, drug_B_start:drug_B_end]
    X_val_cell_line = X_val[:, cell_line_start:]

    X_test_drug_A = X_test[:, 0:drug_A_end]
    X_test_drug_B = X_test[:, drug_B_start:drug_B_end]
    X_test_cell_line = X_test[:, cell_line_start:]

    def normalize(X, means1=None, std1=None, means2=None, std2=None, norm="tanh_norm"):
        # Ensure X is contiguous and handle NaN values in X
        X = np.ascontiguousarray(X)
        if std1 is None:
            std1 = np.nanstd(X, axis=0)
        if means1 is None:
            means1 = np.nanmean(X, axis=0)
        std1_corrected = np.where(std1 == 0, np.finfo(float).eps, std1)
        X_normalized = (X - means1) / std1_corrected
        if norm == "norm":
            return X_normalized, means1, std1, None, None
        elif norm == "tanh":
            return np.tanh(X_normalized), means1, std1, None, None
        elif norm == "tanh_norm":
            X_normalized = np.tanh(X_normalized)
            if means2 is None:
                means2 = np.mean(X_normalized, axis=0)
            if std2 is None:
                std2 = np.std(X_normalized, axis=0)
            # Again ensure no division by zero for std2
            std2_corrected = np.where(std2 == 0, np.finfo(float).eps, std2)
            X_normalized = (X_normalized - means2) / std2_corrected
            return X_normalized, means1, std1, means2, std2

    #### Normalize

    # Normalize drug A/B and cell line without variance-based feature filtering
    # Tr
    X_tr_drug_A, mean_drug_A, std_drug_A, _, _ = normalize(X_tr_drug_A, norm=norm)
    X_tr_drug_B, mean_drug_B, std_drug_B, _, _ = normalize(X_tr_drug_B, norm=norm)
    X_tr_cell_line, mean_cell_line, std_cell_line, _, _ = normalize(
        X_tr_cell_line, norm=norm
    )

    # Val
    X_val_drug_A, _, _, _, _ = normalize(
        X_val_drug_A, mean_drug_A, std_drug_A, norm=norm
    )
    X_val_drug_B, _, _, _, _ = normalize(
        X_val_drug_B, mean_drug_B, std_drug_B, norm=norm
    )
    X_val_cell_line, _, _, _, _ = normalize(
        X_val_cell_line, mean_cell_line, std_cell_line, norm=norm
    )

    # Test
    X_test_drug_A, _, _, _, _ = normalize(
        X_test_drug_A, mean_drug_A, std_drug_A, norm=norm
    )
    X_test_drug_B, _, _, _, _ = normalize(
        X_test_drug_B, mean_drug_B, std_drug_B, norm=norm
    )
    X_test_cell_line, _, _, _, _ = normalize(
        X_test_cell_line, mean_cell_line, std_cell_line, norm=norm
    )

    # Save processed data
    save_path = (
        "/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/data_test_fold%d_%s.p"
        % (test_fold, norm)
    )
    with open(save_path, "wb") as f:
        pickle.dump(
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
            ),
            f,
        )

    print(
        f"Data for test_fold={test_fold} and val_fold={val_fold} processed and saved to {save_path}"
    )


# Define the fold configurations
fold_configurations = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

# Loop over the configurations
for test_fold, val_fold in fold_configurations:
    process_data_for_folds(test_fold, val_fold, norm="tanh")
