from sktime.performance_metrics.forecasting import make_forecasting_scorer
from sktime.performance_metrics.forecasting import (
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

mse = MeanSquaredError()
rmse = MeanSquaredError(square_root=True)
mape = MeanAbsolutePercentageError(multioutput="uniform_average", symmetric=False)


def custom_gain_function(y, y_pred, multioutput="uniform_average", **kwargs):

    # true and pred arrays to dataframe
    my_df = pd.DataFrame()
    my_df["y"] = pd.Series(y)
    my_df["y_pred"] = pd.Series(y_pred)

    # calculate loss
    my_df.loc[(my_df["y"] > 0) & (my_df["y_pred"] > 0), "cash"] = my_df["y"]
    my_df.loc[(my_df["y"] > 0) & (my_df["y_pred"] < 0), "cash"] = my_df["y"].apply(
        lambda y: -y if y > -0.05 else -0.05
    )
    my_df.loc[(my_df["y"] < 0) & (my_df["y_pred"] < 0), "cash"] = -(my_df["y"])
    my_df.loc[(my_df["y"] < 0) & (my_df["y_pred"] > 0), "cash"] = my_df["y"].apply(
        lambda y: y if y > -0.05 else -0.05
    )

    # loss column to array
    loss = my_df.cash.to_numpy()

    # loss square
    # squared_difference = tf.square(loss)

    return tf.reduce_sum(loss, axis=-1).numpy()


def direction_score(y, y_pred, multioutput="uniform_average", **kwargs):
    y = y > 0
    y_pred = y_pred > 0
    return f1_score(y, y_pred)


forecast_profit_score = make_forecasting_scorer(
    func=custom_gain_function, multioutput="uniform_average", greater_is_better=True
)
