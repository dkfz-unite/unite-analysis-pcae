import re
import pandas as pd
import json
from typing import Callable, Protocol, List, Tuple
import sys
from enum import StrEnum
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

Filter = Callable[[np.ndarray], np.ndarray]


class Imputer(Protocol):
    # define interface for imputers - mirror sklearn's various imputers
    def fit(self, X: np.ndarray) -> None: ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...

class NullImputer:
    # a null imputer that does nothing
    def fit(self, X: np.ndarray) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return X


class HandleMissingValuesOptions(StrEnum):
    GENEWISE_DELETION = "gene wise deletion"
    IMPUTE_GENE_MEAN_EXPRESSION = "impute gene mean expression"
    IMPUTE_GENE_MEDIAN_EXPRESSION = "impute gene median expression"


class Options(BaseModel):
    # pydantic validator object for options.json
    treat_zeros_as_missing: bool = Field(
        default=False, description="If true, zeros are treated as missing values (NaN)."
    )
    handle_missing_values: HandleMissingValuesOptions = Field(
        default=HandleMissingValuesOptions.GENEWISE_DELETION,
        description="Method to handle missing genes.",
    )
    scale_features: bool = Field(
        default=True,
        description="If true, data is scaled to have mean 0 and variance 1.",
    )


def set_zeros_to_nan(ar: np.ndarray) -> np.ndarray:
    ar[ar == 0] = np.nan
    return ar


def no_filter(ar: np.ndarray) -> np.ndarray:
    # a null filter that does nothing
    return ar


def scale_cols(ar: np.ndarray) -> np.ndarray:
    # Scale each column to have mean 0 and variance 1
    means = np.mean(ar, axis=0)
    stds = np.std(ar, axis=0)
    # check if standard deviation is nonzero
    if any(stds == 0.0):
        raise ValueError(
            "Data could not be scaled due to zero variances being found. "
            "This indicates that all expression values of at least one gene are equal"
        )
    return (ar - means) / stds


def gene_wise_deletion(ar: np.ndarray) -> np.ndarray:
    # Remove cols with any NaN values
    return ar[:, ~np.isnan(ar).any(axis=0)]


def get_treat_zeros_filter(opts: Options) -> Filter:
    if opts.treat_zeros_as_missing:
        return set_zeros_to_nan
    return no_filter


def get_missing_values_filter(opts: Options) -> Filter:
    if opts.handle_missing_values == HandleMissingValuesOptions.GENEWISE_DELETION:
        return gene_wise_deletion
    return no_filter


def get_imputer(opts: Options) -> Imputer:
    if (
        opts.handle_missing_values
        == HandleMissingValuesOptions.IMPUTE_GENE_MEAN_EXPRESSION
    ):
        return SimpleImputer(strategy="mean")
    elif (
        opts.handle_missing_values
        == HandleMissingValuesOptions.IMPUTE_GENE_MEDIAN_EXPRESSION
    ):
        return SimpleImputer(strategy="median")
    else:
        return NullImputer()


def get_scale_filter(opts: Options) -> Filter:
    if opts.scale_features:
        return scale_cols
    return no_filter


def read_input_file(file_path):
    df = pd.read_csv(file_path, delimiter="\t", index_col=0, usecols=["tpm", "gene_id"])
    return df


def read_all_input_files(root_path: Path) -> Tuple[np.ndarray, List[str]]:
    data_frames = []
    donor_names = []
    pattern = re.compile(r"^(Donor\d+)-.*\.tsv$")
    for file_path in root_path.glob("*.tsv"):
        match = pattern.match(file_path.name)
        if not match:
            Warning(
                f"File {file_path.name} does not match expected pattern and will be skipped."
            )
            continue
        df = read_input_file(file_path)
        data_frames.append(df)
        donor_names.append(match.group(1))

    if len(data_frames) == 0:
        raise ValueError(f"No files found matching {pattern}")
    # merge all the data frames so that the index is the union of all gene_ids
    merged_df = pd.concat(data_frames, axis=1, join="outer")
    # convert to numpy array and transpose so that rows are samples and columns are genes
    return merged_df.to_numpy().T, donor_names


def read_options(file_path: Path) -> Options:
    with open(file_path, "r") as f:
        config = json.load(f)
    return Options(**config)


def prepare_data(ar: np.ndarray, opts: Options) -> np.ndarray:
    treat_zeros_filter = get_treat_zeros_filter(opts)
    missing_values_filter = get_missing_values_filter(opts)
    scale_filter = get_scale_filter(opts)
    imputer = get_imputer(opts)

    # Apply filters and imputer
    ar = treat_zeros_filter(ar)
    ar = missing_values_filter(ar)
    # log transform before imputing
    ar = np.log1p(ar)
    # impute missing values if requested
    ar = imputer.fit_transform(ar)

    # if any missing values remain this indicates a bug
    assert not np.isnan(ar).any(), "Missing values at this stage indicates a bug"
    # scale features if requested
    ar = scale_filter(ar)
    return ar


def do_pca(ar: np.ndarray) -> np.ndarray:
    pca = PCA()
    transformed = pca.fit_transform(ar)
    return transformed


def main(root_path):
    x, donor_names = read_all_input_files(Path(root_path))
    opts = read_options(Path(root_path) / "options.json")
    x_prepared = prepare_data(x, opts)
    x_pca = do_pca(x_prepared)
    # get the components into a dataframe
    df_pca = pd.DataFrame(
        x_pca, index=donor_names, columns=[f"PC{i+1}" for i in range(x_pca.shape[1])]
    )
    df_pca.index.name = "Sample"
    # save to tsv
    df_pca.to_csv(Path(root_path) / "results.tsv", sep="\t")


if __name__ == "__main__":
    root_path = sys.argv[1]
    main(root_path)
