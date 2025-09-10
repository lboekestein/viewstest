import polars as pl
import pandas as pd
import numpy as np
from functools import reduce
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

        # select only index columns and target columns
        df = df.select(self.index_cols + self.target_cols)

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

        # join dataframes
        joined = self._inner_join_model_predictions()

        # Weighted aggregation
        if method == "weighted":
            if self.weights:
                # verify weights format, length
                if isinstance(self.weights, list) and all(isinstance(w, float) for w in self.weights):
                    if len(self.weights) == self.n_models:
                        weights = self._normalize_weights()

                        if not n_samples:
                            #TODO should this default to the largest sample size in the models? Or a set value?
                            # n_samples == max_sample_size(joined)
                            pass

                        #TODO
                        # pooled_df = self._linear_pool_resample(joined, weights, n_samples)
                        # return pooled_df

                        pass
                    else:
                        raise ValueError(f"Got {self.n_models} models but {self.weights} were given."
                                         f" Number of weights must match number of models")
                else:
                    raise ValueError(f"Weights must be specified as list of floats, got {type(self.weights)}")
            else:
                raise ValueError("Weights must be specified for weighted aggregation")

                # TODO this could be changed to defaulting to averaging instead

        # Average aggregation
        elif method == "average":

            # TODO

            pass

        # Concat aggregation
        elif method == "concat":

            # TODO check sample-size consistency
            # TODO implement linear pooling

            pass


        else:
            raise ValueError(f"Unsupported aggregation method: {method}. Must be one of 'weighted', 'average' or 'concat'")

        pass

    def aggregate_point_predictions(
            self,
            aggregation_func: Union[str, Callable[[pl.Series], float]] = "mean",
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

        if aggregation_func == "mean":
            aggregation_func = pl.Series.mean
        elif aggregation_func == "median":
            aggregation_func = pl.Series.median
        elif aggregation_func == "min":
            aggregation_func = pl.Series.min
        elif aggregation_func == "max":
            aggregation_func = pl.Series.max
        elif callable(aggregation_func):
            aggregation_func = aggregation_func
        else:
            raise ValueError(f"Unsupported aggregation function: \"{aggregation_func}\", must be one of 'mean', 'median', "
                             f"'min', 'max' or custum aggregation function of form Callable[[pl.Series], float]")


        # TODO Implementation here

        pass

    def calculate_ensemble_statistics(self) -> pl.DataFrame:
        """
        Calculate comprehensive statistics for the ensemble distribution.

        Returns:
            Polars DataFrame with ensemble statistics including mean, std, and quantiles
        """
        # Implementation here
        pass


    def _normalize_weights(self) -> List[float]:
        """
        Normalize model weights

        Returns:
            list of normalized weights that sum to 1
        """
        total = sum(self.weights)
        return [w / total for w in self.weights]

    def _inner_join_model_predictions(self) -> pl.DataFrame:
        """
        Inner join model predictions based in index columns.

        Returns:
            Polars DataFrame with combined model predictions
        """

        # join dataframes on index columns
        if self.models:
            joined = reduce(
                lambda left, right: left.join(right, on=self.index_cols, how="inner"),
                self.models,
            )

            # Optional; could raise or print statement with warning if, how many rows are dropped

            return joined
        else:
            raise ValueError("No models to join. Add at least one model using add_model()")

