import pytest
import polars as pl
import pandas as pd
from  aggregation import AggregationManager

@pytest.fixture
def manager():
    return AggregationManager(
        index_cols=['time', 'region_id'],
        target_cols=['conflict_prob', 'death_count']
    )

@pytest.fixture
def valid_df():
    return pl.DataFrame({
        "time": [1, 2],
        "region_id": [10, 20],
        "conflict_prob": pl.Series(
            "conflict_prob",
            [[0.1, 0.2], [0.3, 0.4]],
            dtype=pl.List(pl.Float64)
        ),
        "death_count": pl.Series(
            "death_count",
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=pl.List(pl.Float64)
        )
    })

def test_add_valid_model(manager, valid_df):
    manager.add_model(valid_df)
    assert manager.n_models == 1
    assert len(manager.models) == 1
    assert isinstance(manager.models[0], pl.DataFrame)

def test_missing_index_column(manager, valid_df):
    bad_df = valid_df.drop("region_id")
    with pytest.raises(ValueError, match="Missing required index columns"):
        manager.add_model(bad_df)

def test_missing_target_column(manager, valid_df):
    bad_df = valid_df.drop("conflict_prob")
    with pytest.raises(ValueError, match="Missing target columns"):
        manager.add_model(bad_df)

def test_index_column_wrong_type(manager, valid_df):
    # Make 'time' a string instead of int
    bad_df = valid_df.with_columns(pl.Series("time", ["a", "b"]))
    with pytest.raises(TypeError, match="Index column 'time' must be integer"):
        manager.add_model(bad_df)

def test_target_column_wrong_type(manager, valid_df):
    # Make 'conflict_prob' a float instead of a list
    bad_df = valid_df.with_columns(pl.Series("conflict_prob", [0.1, 0.2]))
    with pytest.raises(TypeError, match="Target column 'conflict_prob' must be a list"):
        manager.add_model(bad_df)

def test_add_from_pandas(manager, valid_df):
    # Convert valid_df to pandas
    pdf = valid_df.to_pandas()
    manager.add_model(pdf)
    assert manager.n_models == 1

