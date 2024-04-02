import numpy as np
import pandas as pd
import pickle
import gzip

##### Define the parameters for data generation: folds for testing and validation and normalization strategy

# in this example tanh normalization is used
# fold 0 is used for testing and fold 1 for validation (hyperparamter selection)
norm = "tanh"
test_fold = 0
val_fold = 1
#### Define nomalization function


def normalize(
    X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm="tanh_norm"
):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1 != 0
    X = X[:, feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X - means1) / std1[feat_filt]
    if norm == "norm":
        return (X, means1, std1, feat_filt)
    elif norm == "tanh":
        return (np.tanh(X), means1, std1, feat_filt)
    elif norm == "tanh_norm":
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X - means2) / std2
        X[:, std2 == 0] = 0
        return (X, means1, std1, means2, std2, feat_filt)


#### Load features and labels

# contains the data in both feature ordering ways (drug A - drug B - cell line and drug B - drug A - cell line)
# in the first half of the data the features are ordered (drug A - drug B - cell line)
# in the second half of the data the features are ordered (drug B - drug A - cell line)

file = gzip.open("/hpc2hdd/home/mgong081/Projects/DeepSynergy/raw_data/X.p.gz", "rb")
X = pickle.load(file)
file.close()
# Split X into the two feature orderings
X1 = X[: len(X) // 2]
X2 = X[len(X) // 2 :]
X1.shape
X1
X2


def find_consecutive_identical_columns(X1, X2):
    # Number of columns in X1 and X2
    num_cols = X1.shape[1]

    # Initialize the index of the first non-identical column (from the end)
    first_non_identical_idx = None

    # Iterate over columns from the end to the beginning
    for i in range(num_cols - 1, -1, -1):
        # Compare the ith column in both matrices
        if not np.allclose(X1[:, i], X2[:, i]):
            first_non_identical_idx = i + 1
            break

    # If all columns are identical, then first_non_identical_idx will be None
    if first_non_identical_idx is None:
        return X1, X2  # All columns are identical
    else:
        # Return the identical columns from the end to the first non-identical index
        return X1[:, first_non_identical_idx:], X2[:, first_non_identical_idx:]


identical_columns_X1, identical_columns_X2 = find_consecutive_identical_columns(X1, X2)

# Due to the large size of the arrays, running this example might not be feasible in this environment.
identical_columns_X1.shape
identical_columns_X2.shape
# cell_line_feature_indices = np.all(X1 == X2, axis=0)
# cell_line_features = X1[:, 8773:]
# 12758 - 8774
# (1309 + 802 + 2276) * 2
# cell_line_feature_indices
# cell_line_features.shape
(12758 - 7904) / 2
# contains synergy values and fold split (numbers 0-4)
labels = pd.read_csv(
    "/hpc2hdd/home/mgong081/Projects/DeepSynergy/raw_data/labels.csv", index_col=0
)
# labels are duplicated for the two different ways of ordering in the data
labels = pd.concat([labels, labels])
labels
# Split labels to half
labels1 = labels.iloc[: len(labels) // 2]
labels2 = labels.iloc[len(labels) // 2 :]
labels1
labels2
#### Define indices for splitting

# indices of training data for hyperparameter selection: fold 2, 3, 4
idx_tr = np.where(
    np.logical_and(labels["fold"] != test_fold, labels["fold"] != val_fold)
)
# indices of validation data for hyperparameter selection: fold 1
idx_val = np.where(labels["fold"] == val_fold)
# indices of training data for model testing: fold 1, 2, 3, 4
idx_train = np.where(labels["fold"] != test_fold)
# indices of test data for model testing: fold 0
idx_test = np.where(labels["fold"] == test_fold)
#### Split data

X_tr = X[idx_tr]
X_val = X[idx_val]
X_train = X[idx_train]
X_test = X[idx_test]
y_tr = labels.iloc[idx_tr]["synergy"].values
y_val = labels.iloc[idx_val]["synergy"].values
y_train = labels.iloc[idx_train]["synergy"].values
y_test = labels.iloc[idx_test]["synergy"].values
print(X_tr.shape, X_val.shape, X_train.shape, X_test.shape)
#### Normalize training and validation data for hyperparameter selection

if norm == "tanh_norm":
    X_tr, mean, std, mean2, std2, feat_filt = normalize(X_tr, norm=norm)
    X_val, mean, std, mean2, std2, feat_filt = normalize(
        X_val, mean, std, mean2, std2, feat_filt=feat_filt, norm=norm
    )
else:
    X_tr, mean, std, feat_filt = normalize(X_tr, norm=norm)
    X_val, mean, std, feat_filt = normalize(
        X_val, mean, std, feat_filt=feat_filt, norm=norm
    )
#### Normalize training and test data for methods comparison

if norm == "tanh_norm":
    X_train, mean, std, mean2, std2, feat_filt = normalize(X_train, norm=norm)
    X_test, mean, std, mean2, std2, feat_filt = normalize(
        X_test, mean, std, mean2, std2, feat_filt=feat_filt, norm=norm
    )
else:
    X_train, mean, std, feat_filt = normalize(X_train, norm=norm)
    X_test, mean, std, feat_filt = normalize(
        X_test, mean, std, feat_filt=feat_filt, norm=norm
    )
print(X_tr.shape, X_val.shape, X_train.shape, X_test.shape)
#### Save data as pickle file

pickle.dump(
    (X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test),
    open("data_test_fold%d_%s.p" % (test_fold, norm), "wb"),
)
