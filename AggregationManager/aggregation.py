import polars as pl
import pandas as pd
import numpy as np
from typing import List, Union, Optional, Dict, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AggregationManager:
    """
    Advanced distribution aggregation manager for ensemble forecasting.

    Supports weighted aggregation of both point predictions and distribution samples
    from multiple models using Polars for efficient operations.

    Parameters:
        weights: Optional weights for each model (default: equal weights)
        index_cols: List of index column names (default: ['time', 'entity_id'])
        target_cols: List of target variable names
    """

    def __init__(
            self,
            weights: Optional[List[float]] = None,
            index_cols: List[str] = ['time', 'entity_id'],
            target_cols: Optional[List[str]] = None
    ):
        self.models: List[pl.DataFrame] = []
        self.weights = weights
        self.index_cols = index_cols
        self.target_cols = target_cols
        self.n_models = 0

    def add_model(self, data: Union[pl.DataFrame, pd.DataFrame, str, Path]) -> None:
        """
        Add a model's predictions to the aggregation pool.

        Parameters:
            data: Polars DataFrame, Pandas DataFrame or path to parquet/csv file containing predictions
        """
        # Read in dataframe as polars dataframe
        if isinstance(data, pl.DataFrame):
            df = data
        elif isinstance(data, pd.DataFrame):
            df = pl.from_pandas(data)
        elif isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix == ".parquet":
                try:
                    df = pl.read_parquet(path)
                except Exception as e:
                    raise ValueError(f"Failed to read parquet file {path}: {e}")
            elif path.suffix == ".csv":
                try:
                    df = pl.read_csv(path)
                except Exception as e:
                    raise ValueError(f"Failed to read csv file {path}: {e}")
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. File must be either .parquet or .csv")
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                "Type must be either Polars DataFrame, Pandas DataFrame or path to parquet/csv file"
            )


        # Validate that index columns are in dataframe
        missing_index_cols = [c for c in self.index_cols if c not in df.columns]
        if missing_index_cols:
            raise ValueError(f"Missing required index columns: {missing_index_cols}")

        # Validate that target columns are in dataframe
        if self.target_cols:
            missing_targets = [c for c in self.target_cols if c not in df.columns]
            if missing_targets:
                raise ValueError(f"Missing target columns: {missing_targets}")

        # Validate target column types
        for col in self.index_cols:
            if not isinstance(df[col].dtype, pl.datatypes.IntegerType):
                raise TypeError(f"Index column '{col}' must be integer, got {df[col].dtype}")
        for col in self.target_cols:
            if not isinstance(df[col].dtype, pl.datatypes.List):
                raise TypeError(f"Target column '{col}' must be a list, got {df[col].dtype}")

        # TODO drop irrelevant columns

        # Append model, increase model count
        self.models.append(df)
        self.n_models += 1


    def aggregate_distributions(
            self,
            method: str = "concat",
            n_samples: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Aggregate distributions from all models using specified method.

        Parameters:
            method: Aggregation method - "concat", "average", or "weighted"
            n_samples: Number of samples to generate for weighted method

        Returns:
            Polars DataFrame with aggregated distributions
        """




        pass

    def aggregate_point_predictions(
            self,
            aggregation_func: str = "mean",
            use_weights: bool = True
    ) -> pl.DataFrame:
        """
        Aggregate point predictions from all models.

        Parameters:
            aggregation_func: Aggregation function ("mean", "median", "min", "max")
            use_weights: Whether to use model weights

        Returns:
            Polars DataFrame with aggregated point predictions
        """
        # Implementation here
        pass

    def calculate_ensemble_statistics(self) -> pl.DataFrame:
        """
        Calculate comprehensive statistics for the ensemble distribution.

        Returns:
            Polars DataFrame with ensemble statistics including mean, std, and quantiles
        """
        # Implementation here
        pass