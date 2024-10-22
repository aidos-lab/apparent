import pytest
import networkx as nx
import numpy as np
import pandas as pd
import tempfile

from apparent.features import add_curvature_column
from tests.test_curvature import _line_graph, _triangle_graph, _tree_graph


# Test Arguments
class test_args:
    def __init__(self, curvature, alpha=0, prob_fn=None, sample_size=3):
        self.curvature = curvature
        self.alpha = alpha
        self.prob_fn = prob_fn
        self.sample_size = sample_size


# Fixtures
@pytest.fixture
def empty_feature_dataframe():
    df = pd.DataFrame({"G": [_triangle_graph(), _line_graph(), _tree_graph()]})
    return df


@pytest.fixture
def incomplete_feature_dataframe():
    tri_orc = np.array([0.5, 0.5, 0.5])
    df = pd.DataFrame(
        {
            "G": [_triangle_graph(), _line_graph(), _tree_graph()],
            "OR_0": [tri_orc, np.nan, np.nan],
        }
    )
    return df


class TestFeatureComputations:

    def test_add_curvature_column_OR(self, empty_feature_dataframe):
        args = test_args("OR")
        df, label = add_curvature_column(empty_feature_dataframe, args)
        assert label == "OR_0"
        assert label in df.columns
        for curvature in df[label].values:
            print(curvature)
            assert isinstance(curvature, np.ndarray)

    def test_add_curvature_column_Forman(self, empty_feature_dataframe):
        args = test_args("Forman")
        df, label = add_curvature_column(empty_feature_dataframe, args)
        assert label == "Forman"
        assert label in df.columns
        for curvature in df[label].values:
            assert isinstance(curvature, np.ndarray)

    def test_add_curvature_column_Resistance(self, empty_feature_dataframe):
        args = test_args("Resistance")
        df, label = add_curvature_column(empty_feature_dataframe, args)
        assert label == "Resistance"
        assert label in df.columns
        for curvature in df[label].values:
            assert isinstance(curvature, np.ndarray)

    def test_add_curvature_column_unsupported(self, empty_feature_dataframe):
        args = test_args("Unsupported")
        with pytest.raises(ValueError):
            df, label = add_curvature_column(empty_feature_dataframe, args)

    def test_overwrite_curvature_column(self, incomplete_feature_dataframe):
        test_sample_size = 1

        args = test_args("OR", sample_size=test_sample_size)
        df, label = add_curvature_column(incomplete_feature_dataframe, args)

        original = incomplete_feature_dataframe[label][0]
        curvatures = df[label]
        np.testing.assert_array_equal(curvatures[0], original)

        original_length = incomplete_feature_dataframe[label].notna().sum()
        assert original_length == 1
        assert df[label].notna().sum() == original_length + test_sample_size
