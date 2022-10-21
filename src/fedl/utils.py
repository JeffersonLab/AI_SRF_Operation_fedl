import io

import mlflow
import numpy as np
import onnx
import onnxruntime
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import List, Union, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch import nn
from torch.utils.data import DataLoader

sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def score_model(y_pred: pd.DataFrame, y_test: pd.DataFrame, multioutput='raw_values') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r2 = r2_score(y_test, y_pred, multioutput=multioutput)
    mse = mean_squared_error(y_test, y_pred, multioutput=multioutput)
    mae = mean_absolute_error(y_test, y_pred, multioutput=multioutput)

    return r2, mse, mae


def print_model_scores(r2: Union[float, List[float]], mse: Union[float, List[float]], mae: Union[float, List[float]],
                       set_name: str) -> None:
    """Print out the scores and loss values in a standard fashion"""
    print(f"Avg. {set_name} R2: {r2}")
    print(f"Avg. {set_name} MSE: {mse}")
    print(f"Avg. {set_name} MAE: {mae}")


def plot_percent_error(df, title, show=False):
    plt.figure(figsize=(10, 10))

    g = sns.FacetGrid(data=df.loc[df.type == '%error', :], col='variable', aspect=1.5, height=1.5)
    g.map_dataframe(sns.scatterplot, x='EGAIN', y='value', alpha=0.1)
    g.set_axis_labels("EGain of Modeled CMs", "% Error")
    # for (row_key, col_key), ax in g.axes_dict.items():
    #     ax.set_title(f"\n{col_key[3:7]} {row_key}", loc='center', pad=-20)
    #     # ax.invert_yaxis()

    g.fig.subplots_adjust(top=0.7)
    g.fig.suptitle(title)
    mlflow.log_figure(g.fig, artifact_file=f"{title}.png".replace(" ", "_"))
    if show:
        plt.show()
    return g


def plot_predictions_over_time(df, title="", show=False):
    g = sns.FacetGrid(df[df.type.isin(['observed', 'predicted'])], col='variable', hue='type')
    g.map_dataframe(sns.scatterplot, x='dtime', y='value', alpha=0.2)
    g.add_legend()
    g.set_axis_labels("Time", "rem/h")
    g.fig.subplots_adjust(top=0.7)
    g.fig.suptitle(f"Obs. and Pred. Over Time {title}")

    # for (row_key, col_key), ax in g.axes_dict.items():
    # noinspection PyUnusedLocal
    for (col_key), ax in g.axes_dict.items():
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # ax.xaxis.set_major_locator(mdates.MonthLocator())
        # ax.xaxis.set_major_locator(mdates.DateLocator())
    plt.gcf().autofmt_xdate()

    mlflow.log_figure(g.fig, artifact_file=f"{title}.png".replace(" ", "_"))

    if show:
        plt.show()

    return g


def get_sensor_plot_data(y_true: pd.DataFrame, y_pred: pd.DataFrame, dtime, egain):
    """Generates the DataFrame needed by the show_sensor_data function"""

    # Construct basic DataFrames that we will update and concatenate
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    y_err = y_pred - y_true
    y_err_perc = 100 * (y_pred - y_true) / y_true


# Generate the different DataFrames that will then be combined and melted
    # Observed data
    y_true['type'] = ['observed'] * len(y_true)
    y_true['dtime'] = dtime
    y_true['EGAIN'] = egain

    # Predicted
    y_pred['type'] = ['predicted'] * len(y_pred)
    y_pred['dtime'] = dtime
    y_pred['EGAIN'] = egain

    # Error
    y_err['type'] = ['error'] * len(y_err)
    y_err['dtime'] = dtime
    y_err['EGAIN'] = egain
    y_err.columns = y_true.columns

    # Percent Error
    y_err_perc = pd.DataFrame(y_err_perc)
    y_err_perc['type'] = ['%error'] * len(y_err_perc)
    y_err_perc['dtime'] = dtime
    y_err_perc['EGAIN'] = egain
    y_err_perc.columns = y_true.columns

    # Combine and melt the detector DataFrames
    df = pd.concat([y_true, y_pred, y_err, y_err_perc])
    df_melt = df.melt(id_vars=['type', 'EGAIN', 'dtime'])
    df_melt.value = df_melt.value.astype('float64')

    # Split out gamma and neutron data to be plotted separately
    g_df = df_melt[df_melt['variable'].str.contains('gDsRt')]
    n_df = df_melt[df_melt['variable'].str.contains('nDsRt')]

    diag_df = y_true.melt(id_vars=['type', 'EGAIN', 'dtime'], value_name='observed')
    diag_df['predicted'] = y_pred.melt(id_vars=['type', 'EGAIN', 'dtime'], value_name='predicted')['predicted']
    g_diag_df = diag_df.loc[diag_df.variable.str.contains("_gDsRt"), :]
    n_diag_df = diag_df.loc[diag_df.variable.str.contains("_nDsRt"), :]

    return g_df, n_df, g_diag_df, n_diag_df


# noinspection PyUnusedLocal
def plot_diagonal(*args, **kwargs):
    """This plots a black, diagonal line from the bottom-left to top-right corners."""
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    top = max(ylim + xlim)
    x = [0, top]
    y = x
    plt.plot(x, y, color='k')


def show_pred_vs_obs_plot(df, title, alpha=0.25, show=False):
    plt.figure(figsize=(10, 10))

    g = sns.FacetGrid(data=df, col='variable', hue='variable', aspect=1, height=3, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x='observed', y='predicted', alpha=alpha)
    g.map(plot_diagonal)
    g.set_axis_labels("Observed (rem/h)", "Predicted (rem/h)")
    for col_key, ax in g.axes_dict.items():
        ax.set_title(f"\n{col_key[3:7]}", loc='center', pad=-20)
        # ax.invert_yaxis()

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title)
    mlflow.log_figure(g.fig, artifact_file=f"{title}.png".replace(" ", "_"))
    if show:
        plt.show()
    return g


def show_sensor_plot(df: pd.DataFrame, title, show=False):
    plt.figure(figsize=(10, 10))

    g = sns.FacetGrid(data=df[df.type != '%error'], col='variable', row='type', hue='type', aspect=1.5, height=1.5)
    g.map_dataframe(sns.scatterplot, x='EGAIN', y='value', alpha=0.1)
    g.set_axis_labels("Total EGain of Modeled CMs", "rem/h")
    for (row_key, col_key), ax in g.axes_dict.items():
        ax.set_title(f"\n{col_key[3:7]} {row_key}", loc='center', pad=-20)
        # ax.invert_yaxis()

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title)
    mlflow.log_figure(g.fig, artifact_file=f"{title}.png".replace(" ", "_"))
    if show:
        plt.show()
    return g


def plot_data(g_df: pd.DataFrame, n_df: pd.DataFrame, g_diag_df: pd.DataFrame, n_diag_df: pd.DataFrame,
              title: str = "", show=False) -> None:
    show_sensor_plot(g_df, f'Gamma {title}', show=show)
    show_sensor_plot(n_df, f'Neutron {title}', show=show)
    show_pred_vs_obs_plot(g_diag_df, f'Gamma Obs. Vs. Pred {title}', show=show)
    show_pred_vs_obs_plot(n_diag_df, f'Neutron Obs. Vs. Pred {title}', show=show)
    plot_percent_error(g_df, f'gamma %Err {title}', show=show)
    plot_percent_error(n_df, f'neutron %Err {title}', show=show)
    plot_predictions_over_time(g_df, title=f'Gamma Obs and Pred Over Time {title}', show=show)
    plot_predictions_over_time(n_df, title=f'Neutron Obs and Pred Over Time {title}', show=show)


def plot_results(df, title):
    """Plots the performance results.  df.columns are 'date' 'R2_INX1L22_nDsRt', ... 'R2_INX1L27_gDsRt', "MSE_INX...'
    """
    df_melt = df.melt(id_vars=['date'], var_name='signal_metric', value_name='value')
    df_melt['metric'] = df_melt.signal_metric.apply(lambda x: x.split('_')[0])
    df_melt['detector'] = df_melt.signal_metric.apply(lambda x: x.split('_')[1])
    df_melt['radiation'] = df_melt.signal_metric.apply(lambda x: x.split('_')[2])

    g = sns.FacetGrid(data=df_melt, row='radiation', col='metric', sharex=True, sharey=False, hue='detector')
    g.map(sns.lineplot, 'date', 'value')
    #    g.set(ylim=(-5, 5))
    g.fig.suptitle(f"{title}: Scores by detectors across Rad Type")
    g.fig.subplots_adjust(top=0.95)
    g.add_legend()
    plt.show()

    g = sns.FacetGrid(data=df_melt[df_melt.radiation == "nDsRt"], row='detector', col='metric', sharex=True,
                      sharey=False, hue='metric')
    g.map(sns.lineplot, 'date', 'value')
    #    g.set(ylim=(-5, 5))
    g.fig.suptitle(f"{title}: Neutron scores across detectors")
    g.fig.subplots_adjust(top=0.95)
    g.add_legend()
    plt.show()

    g = sns.FacetGrid(data=df_melt[df_melt.radiation == "gDsRt"], row='detector', col='metric', sharex=True,
                      sharey=False, hue='metric')
    g.map(sns.lineplot, 'date', 'value')
    #    g.set(ylim=(-5, 5))
    g.fig.suptitle(f"{title}: Gamma scores across detectors")
    g.fig.subplots_adjust(top=0.95)
    g.add_legend()
    plt.show()

    r2_cols = df.columns[df.columns.str.startswith('R2')]
    mse_cols = df.columns[df.columns.str.startswith('MSE')]
    mae_cols = df.columns[df.columns.str.startswith('MAE')]

    plt.plot(df.date, df[r2_cols].mean(axis=1), 'k', label="avg R-Sq")
    plt.legend()
    plt.title(f"{title}: Average R-Sq across all signals")
    plt.show()

    plt.plot(df.date, df[mse_cols].mean(axis=1), 'k', label='avg MSE')
    plt.legend()
    plt.title(f"{title}: Average MSE across all signals")
    plt.show()

    plt.plot(df.date, df[mae_cols].mean(axis=1), 'k', label='avg MAE')
    plt.legend()
    plt.title(f"{title}: Average MAE across all signals")
    plt.show()


def make_predictions(model: nn.Module, data_loader: DataLoader, device, y_cols: List[str]):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            pred_list.append(model(inputs))

    y_pred = pd.DataFrame(torch.concat(pred_list).cpu().numpy(), columns=y_cols)

    return y_pred

def convert_to_onnx(model, x, input_names, output_names):
    """Convert a model to ONNX format.
    Args:
        model: A pytorch model
        x: An input to model.  requires_grad should be trun for this
        input_name: The name of the input to the model

    """
    torch_out = model(x)
    onnx_model_save = io.BytesIO()
    torch.onnx.export(model, x, onnx_model_save, export_params=True, do_constant_folding=True, input_names=input_names,
                      output_names=output_names, dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    # Check that the model is well-formed
    onnx_model = onnx.load_model_from_string(onnx_model_save.getvalue())
    onnx.checker.check_model(onnx_model)

    # Check that the ONNX model makes the same prediction as the pytorch model
    ort_session = onnxruntime.InferenceSession(onnx_model_save.getvalue())
    ort_inputs = {ort_session.get_inputs()[0].name: x.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(torch_out.detach().cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)

    return onnx_model
