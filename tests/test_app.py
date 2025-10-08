import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
from unittest.mock import patch, MagicMock

from app import (
    Options,
    HandleMissingValuesOptions,
    NullImputer,
    set_zeros_to_nan,
    no_filter,
    scale_cols,
    gene_wise_deletion,
    get_treat_zeros_filter,
    get_missing_values_filter,
    get_imputer,
    read_input_file,
    read_all_input_files,
    read_options,
    prepare_data,
    do_pca,
    main,
)


class TestNullImputer:
    def test_null_imputer_does_nothing(self):
        imputer = NullImputer()
        data = np.array([[1, 2], [3, 4]], dtype=float)

        imputer.fit(data)
        result = imputer.transform(data)
        assert np.array_equal(result, data)

        result = imputer.fit_transform(data)
        assert np.array_equal(result, data)


class TestOptions:
    def test_default_options(self):
        opts = Options()
        assert opts.treat_zeros_as_missing == False
        assert (
            opts.handle_missing_values == HandleMissingValuesOptions.GENEWISE_DELETION
        )
        assert opts.scale_features == True

    def test_options_from_dict(self):
        data = {
            "treat_zeros_as_missing": True,
            "handle_missing_values": "impute gene mean expression",
            "scale_features": False,
        }
        opts = Options(**data)
        assert opts.treat_zeros_as_missing == True
        assert (
            opts.handle_missing_values
            == HandleMissingValuesOptions.IMPUTE_GENE_MEAN_EXPRESSION
        )
        assert opts.scale_features == False


class TestFilterFunctions:
    def test_set_zeros_to_nan(self):
        data = np.array([[0, 1, 2], [3, 0, 5]], dtype=float)
        result = set_zeros_to_nan(data.copy())
        expected = np.array([[np.nan, 1, 2], [3, np.nan, 5]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_no_filter(self):
        data = np.array([[1, 2], [3, 4]], dtype=float)
        result = no_filter(data)
        assert np.array_equal(result, data)

    def test_scale_cols(self):
        data = np.array([[1, 2], [3, 4]], dtype=float)
        result = scale_cols(data)

        # Check that each column has mean ~0 and std ~1
        assert np.allclose(np.mean(result, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(result, axis=0), 1, atol=1e-10)

    def test_gene_wise_deletion(self):
        data = np.array([[1, np.nan, 3], [4, 5, 6]])
        result = gene_wise_deletion(data)
        expected = np.array([[1, 3], [4, 6]])
        np.testing.assert_array_equal(result, expected)


class TestGetterFunctions:
    def test_get_treat_zeros_filter(self):
        opts_true = Options(treat_zeros_as_missing=True)
        opts_false = Options(treat_zeros_as_missing=False)

        filter_true = get_treat_zeros_filter(opts_true)
        filter_false = get_treat_zeros_filter(opts_false)

        data = np.array([[0, 1]], dtype=float)
        result_true = filter_true(data.copy())
        result_false = filter_false(data.copy())

        assert np.isnan(result_true[0, 0])
        assert result_false[0, 0] == 0

    def test_get_missing_values_filter(self):
        opts_deletion = Options(
            handle_missing_values=HandleMissingValuesOptions.GENEWISE_DELETION
        )
        opts_no_deletion = Options(
            handle_missing_values=HandleMissingValuesOptions.IMPUTE_GENE_MEAN_EXPRESSION
        )

        filter_deletion = get_missing_values_filter(opts_deletion)
        filter_no_deletion = get_missing_values_filter(opts_no_deletion)

        data = np.array([[1, np.nan], [2, 3]])
        result_deletion = filter_deletion(data)
        result_no_deletion = filter_no_deletion(data)

        assert result_deletion.shape[1] == 1  # Column with NaN removed
        assert result_no_deletion.shape[1] == 2  # No columns removed

    def test_get_imputer(self):
        opts_mean = Options(
            handle_missing_values=HandleMissingValuesOptions.IMPUTE_GENE_MEAN_EXPRESSION
        )
        opts_median = Options(
            handle_missing_values=HandleMissingValuesOptions.IMPUTE_GENE_MEDIAN_EXPRESSION
        )
        opts_none = Options(
            handle_missing_values=HandleMissingValuesOptions.GENEWISE_DELETION
        )

        imputer_mean = get_imputer(opts_mean)
        imputer_median = get_imputer(opts_median)
        imputer_none = get_imputer(opts_none)

        assert hasattr(imputer_mean, "strategy")
        assert hasattr(imputer_median, "strategy")
        assert isinstance(imputer_none, NullImputer)


class TestFileOperations:
    def create_test_tsv(self, path: Path, data: dict):
        """Helper to create test TSV files"""
        df = pd.DataFrame(data)
        df.index.name = "gene_id"
        df.to_csv(path, sep="\t")

    def test_read_input_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.tsv"

            # Create test data
            data = {"tpm": [1.5, 2.0, 3.5], "other_col": [10, 20, 30]}
            self.create_test_tsv(test_file, data)

            result = read_input_file(test_file)
            assert isinstance(result, pd.DataFrame)
            assert "tpm" in result.columns
            assert len(result) == 3

    def test_read_all_input_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files with proper naming
            files = [
                ("Donor1-sample.tsv", {"tpm": [1, 2, 3]}),
                ("Donor2-sample.tsv", {"tpm": [4, 5, 6]}),
                ("invalid-file.tsv", {"tpm": [7, 8, 9]}),  # Should be skipped
            ]

            for filename, data in files:
                self.create_test_tsv(temp_path / filename, data)

            with patch("app.Warning") as mock_warning:
                result_data, donor_names = read_all_input_files(temp_path)

            assert result_data.shape[0] == 2  # 2 valid files
            assert result_data.shape[1] == 3  # 3 genes
            assert len(donor_names) == 2
            assert "Donor1" in donor_names
            assert "Donor2" in donor_names
            mock_warning.assert_called_once()

    def test_read_all_files_no_files(self):
        """Edge case where no files are found"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with pytest.raises(ValueError, match="No files found matching"):
                result_data, donor_names = read_all_input_files(temp_path)

    def test_read_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            options_file = temp_path / "options.json"

            options_data = {
                "treat_zeros_as_missing": True,
                "handle_missing_values": "impute gene mean expression",
                "scale_features": False,
            }

            with open(options_file, "w") as f:
                json.dump(options_data, f)

            result = read_options(options_file)
            assert isinstance(result, Options)
            assert result.treat_zeros_as_missing == True
            assert (
                result.handle_missing_values
                == HandleMissingValuesOptions.IMPUTE_GENE_MEAN_EXPRESSION
            )
            assert result.scale_features == False


class TestDataProcessing:
    def test_prepare_data_with_defaults(self):
        # Create test data with some zeros and NaNs
        data = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, np.nan, 6.0]])

        opts = Options(
            handle_missing_values=HandleMissingValuesOptions.GENEWISE_DELETION
        )
        result = prepare_data(data, opts)

        # Should have removed columns with NaN/0 after zero treatment
        assert result.shape[1] < data.shape[1]
        assert not np.isnan(result).any()

    def test_prepare_data_with_imputation(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])

        opts = Options(
            handle_missing_values=HandleMissingValuesOptions.IMPUTE_GENE_MEAN_EXPRESSION,
            scale_features=False,
        )
        result = prepare_data(data, opts)

        # Should have same shape (no gene deletion) and no NaNs
        assert result.shape[1] == data.shape[1]
        assert not np.isnan(result).any()

    @pytest.mark.parametrize(
        "data",
        [np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]]), np.array([[1.0, 2.0, 3.0]])],
    )
    @pytest.mark.parametrize(
        "handle_missing_values",
        [
            option.value
            for option in HandleMissingValuesOptions
            if option != HandleMissingValuesOptions.GENEWISE_DELETION
        ],
    )
    @pytest.mark.parametrize("scale_features", [True])
    def test_prepare_data_with_very_few_unique_data_points_with_scaling(
        self, data, handle_missing_values, scale_features
    ):
        opts = Options(
            handle_missing_values=handle_missing_values,
            scale_features=scale_features,
        )

        with pytest.raises(ValueError, match="Data could not be scaled"):
            result = prepare_data(data, opts)

    def test_do_pca(self):
        # Create test data
        data = np.random.rand(10, 5)  # 10 samples, 5 features
        result = do_pca(data)

        assert result.shape[0] == data.shape[0]  # Same number of samples
        assert result.shape[1] <= data.shape[1]  # PCs <= features


# Fixtures for common test data
@pytest.fixture
def sample_data():
    return np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 0.0], [9.0, 10.0, 11.0, 12.0]]
    )


# test end to end with different combinations of options
class TestEndToEnd:
    @pytest.mark.parametrize("treat_zeros", [True, False])
    @pytest.mark.parametrize(
        "missing_value_strategy",
        [
            HandleMissingValuesOptions.GENEWISE_DELETION,
            HandleMissingValuesOptions.IMPUTE_GENE_MEAN_EXPRESSION,
            HandleMissingValuesOptions.IMPUTE_GENE_MEDIAN_EXPRESSION,
        ],
    )
    @pytest.mark.parametrize("scale_features", [True, False])
    def test_end_to_end(
        self, sample_data, treat_zeros, missing_value_strategy, scale_features
    ):

        # write sample data to three files beginning Donor1-sample.tsv, Donor2-sample.tsv, Donor3-sample.tsv
        with tempfile.TemporaryDirectory() as root_dir:
            root_dir = Path(root_dir)
            for i in range(0, 3):
                print(sample_data[i, np.newaxis].shape)
                file_path = root_dir / f"Donor{i}-sample.tsv"
                df = pd.DataFrame(
                    sample_data[i, np.newaxis].T,
                    index=["geneA", "geneB", "geneC", "geneD"],
                    columns=["tpm"],
                )
                df.index.name = "gene_id"
                df.to_csv(file_path, sep="\t", index=True)
            # Create options file
            options = {
                "treat_zeros_as_missing": treat_zeros,
                "handle_missing_values": missing_value_strategy.value,
                "scale_features": scale_features,
            }
            options_path = root_dir / "options.json"
            with open(options_path, "w") as f:
                json.dump(options, f)
            # Run main
            main(root_dir)

            # check the output files exist
            pca_output = root_dir / "results.tsv"
            # load the file and check it has the right shape
            assert pca_output.exists()
            df = pd.read_csv(pca_output, sep="\t", index_col=0)
            assert df.shape[0] == 3  # 3 donors
            # number of PCs should be 1 < the number of genes plus the  (4)
            assert df.shape[1] == 3

            # assert the column names are PC1, PC2, PC3
            assert list(df.columns) == ["PC1", "PC2", "PC3"]

            # assert the Index is named Sample
            assert df.index.name == "Sample"
            assert list(df.index) == ["Donor1", "Donor2", "Donor0"]
