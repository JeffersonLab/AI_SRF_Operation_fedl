import copy
from typing import List, Optional

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, mean_squared_error


def train(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
          num_epochs: int, optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.Module, save_file: str,
          tb_log_dir: str, id_string: str = None,
          lr_scheduler: Optional[optim.lr_scheduler.StepLR] = None) -> torch.nn.modules.Module:
    """Train the given module on the given data and other parameters.

    Note: criterion is a standard pytorch _Loss object.
    """
    best_val_r2 = -math.inf
    if save_file is None:
        save_file = f"{model.__class__.__name__}.{datetime.now().strftime('%Y-%m-%d_%H%M%S')}-state-dict"

    print(f"in train(): device = {device}")
    tb = SummaryWriter(log_dir=tb_log_dir)
    num_batches = len(train_loader)

    for epoch in range(num_epochs):
        print("\n")
        print("=" * 20, "Starting epoch %d" % (epoch + 1), "=" * 20)

        model.train()  # This will cause models with dropout, etc., layers to be activated during training

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % math.ceil(num_batches / 10.) == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss={loss.item():.4f}")

        # Step the learning rate scheduler along
        if lr_scheduler is not None:
            lr_scheduler.step()

        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f"{name}.grad", weight.grad, epoch)

        # If I don't do this after the epoch, then validation loss is consistently lower than train loss.
        output = evaluate_model(model, train_loader, device, criterion, multioutput=True)
        train2_loss, train2_r2, train_mmse, train_mr2 = output['loss'], output['r2'], output['multi_mse'], output[
            'multi_r2']

        output = evaluate_model(model, val_loader, device, criterion, multioutput=True)
        val_loss, val_r2, val_mmse, val_mr2 = output['loss'], output['r2'], output['multi_mse'], output['multi_r2']

        # Track all of these metrics in tensorboard.
        loss_scalars = {'Train_Loss': train2_loss, 'Val Loss': val_loss}
        r2_scalars = {'Train R2': train2_r2, 'Val R2': val_r2}
        tb.add_scalars("Loss", loss_scalars, epoch)
        tb.add_scalars("R2", r2_scalars, epoch)
        tb.add_scalars("Train_Individual_R2", train_mr2, epoch)
        tb.add_scalars("Val_Individual_R2", val_mr2, epoch)
        tb.add_scalars("Train_Individual_MSE", train_mmse, epoch)
        tb.add_scalars("Val_Individual_MSE", val_mmse, epoch)

        # Print out some metrics as we go
        print(f"Train / Val Loss = {train2_loss:.2f} / {val_loss:.2f}")
        print(f"Train / Val R2 = {train2_r2:.2f} / {val_r2:.2f}")

        # Update the save file every time we get a better model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), save_file)
            print("Next best model!  Updating save file.")

    print(f"\n\nLoading up best weights into model. {save_file}")
    model.load_state_dict(torch.load(save_file))

    return model


def evaluate_model(model, dataloader, device, criterion, eval_mode=True, multioutput: bool = True):
    """Run a model in eval mode against a Dataset to calculate loss, accuracy and number correct.

    Args:
        model:  The pytorch model to evaluate
        dataloader:  The data on which to evaluate the model
        device:  The device where the model will operate
        criterion:  The loss function to be calculated
        eval_mode:  Should the model be placed into evaluation mode.  This disables dropout, etc.
        multioutput: Should scores/losses be calculated for outputs in addition.
    """
    if eval_mode:
        model.eval()

    num_batches = len(dataloader)
    batches_run = 0.0

    total_loss = 0

    # Save the predicted values and their matching labels for later analysis.  Save both since the label order is random
    y_preds = []
    y_true = []
    with torch.no_grad():
        for idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            preds = model(data)
            y_preds.append(preds)
            y_true.append(labels)
            total_loss += criterion(preds, labels).item()
            batches_run += 1.0
            if not eval_mode:
                # We're just trying to get an estimate of the training loss.  This gives us a decently stable estimate.
                if (batches_run / num_batches) > 0.1:
                    break

    # Assuming MSE loss, then we need to re-normalize by the number of batches.
    total_loss = total_loss / batches_run

    # Calculate R2
    y_preds = torch.cat(y_preds).to("cpu")
    y_true = torch.cat(y_true).to("cpu")
    r2 = r2_score(y_true.detach().numpy(), y_preds.detach().numpy())

    multi_mse = None
    multi_r2 = None
    if multioutput:
        multi_mse = pd.DataFrame(mean_squared_error(y_true=y_true, y_pred=y_preds, multioutput='raw_values')).T.to_dict(
            orient='list')
        multi_mse = {f"{col}": multi_mse[col][0] for col in multi_mse.keys()}
        multi_r2 = pd.DataFrame(
            r2_score(y_true.detach().numpy(), y_preds.detach().numpy(), multioutput='raw_values')).T.to_dict(
            orient='list')
        multi_r2 = {f"{col}": multi_r2[col][0] for col in multi_r2.keys()}

    output = {'loss': total_loss, 'r2': r2, 'multi_mse': multi_mse, 'multi_r2': multi_r2}
    return output


# def evaluate_model(model, dataloader, device, criterion, eval_mode=True):
#     """Run a model in eval mode against a Dataset to calculate loss, accuracy and number correct."""
#     if eval_mode:
#         model.eval()
#
#     weights = torch.tensor([[50, 50, 50, 50, 50, 50, 1, 1, 1, 1, 1, 1]]).to(device)
#     criterion = new_weighted_mse_loss(weights=weights,
#                                       reduction='none')
#     total_loss = torch.zeros(12).to(device)
#     num_batches = len(dataloader)
#     batches_run = 0.0
#     batch_size = dataloader.batch_size
#     y_preds = torch.zeros(dataloader.dataset.y.numpy().shape)
#     with torch.no_grad():
#         for idx, (inputs, labels) in enumerate(dataloader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             preds = model(inputs)
#             y_preds[idx*batch_size:len(labels) + idx*batch_size, :] = preds
#             # total_loss += criterion(preds, labels).item()
#             total_loss += criterion(preds, labels)
#             batches_run += 1.0
#             if not eval_mode:
#                 # We're just trying to get an estimate of the training loss.  This gives us a decently stable estimate.
#                 if (batches_run / num_batches) > 0.1:
#                     break
#
#     # Assuming MSE loss, then we need to re-normalize by the number of batches.
#     total_loss = (total_loss / batches_run).tolist()
#
#     # Calculate R2
#     y = dataloader.dataset.y.to(torch.device("cpu"))
#     X = dataloader.dataset.X.to(device)
#     # y_pred = model(X).to(torch.device("cpu"))
#     y.detach().numpy()
#     y_preds.numpy()
#     # r2 = r2_score(y.detach().numpy(), y_preds.detach().numpy())
#     r2 = r2_score(y.detach().numpy(), y_preds.detach().numpy(), multioutput='raw_values')
#     return total_loss, r2


# def plot_results(model, split, section='train', epoch=None):
# def plot_results(model, split, gdata: data.FEData, epoch=None, dataloader=None):
#     """Plots results on subset of data.  The dataloader can be used to override auto-creation of test_dataloader"""
#     device = torch.device("cpu")
#     print(f"Using device {device}")
#
#     model = model.to(device)
#
#     model.eval()
#     m_name = model.__class__.__name__
#     if epoch is not None:
#         m_name = f"{m_name} epoch={epoch}"
#
#     batch_size = 256
#
#     if dataloader is None:
#         # master_file = "gradient-scans-2021-08-rf-ndx-data.csv"
#         # master_file = "processed_gradient-scan-2021-11-05_113837.646876-cleaned.csv"
#         # onset_file = 'fe_onset-all-2021-08.tsv'
#         # split_to_train_test_files(master_file)
#
#         # gradient_data = GradientData(filename=master_file, onset_file=onset_file, section=section)
#         train_loader, test_loader = gdata.get_train_test(split=split, train_size=0.75, batch_size=batch_size)
#
#         print("===========================================")
#         print(f"model: {model.__class__.__name__}")
#         print(f"gradient_scan: {gdata.filename}")
#         print(f"onset data: {gdata.onset_file}")
#         print(f"split: {split}")
#         print(f"section: {gdata.section}")
#         print("===========================================")
#     else:
#         test_loader = dataloader
#
#     # dataset is our custom NDX_RF_Dataset class.  It has an X and y member
#     X_test = test_loader.dataset.X
#     y_test = test_loader.dataset.y
#
#     y_pred = model(X_test)
#     y_test = y_test.detach().numpy()
#     y_pred = y_pred.detach().numpy()
#     y_test = pd.DataFrame(y_test, columns=gdata.y_cols)
#     y_pred = pd.DataFrame(y_pred, columns=gdata.y_cols)
#
#     print(f"Avg. R2: {r2_score(y_test, y_pred)}")
#     print(f"Avg. MSE: {mean_squared_error(y_test, y_pred)}")
#     print(f"Avg. MAE: {mean_absolute_error(y_test, y_pred)}")
#
#     # The first 32 columns are GMES, the second 32 columns are ODVH (if provided)
#     summed_gmes = X_test[:, 0:32].sum(axis=1) * data.c100_max_operational_gmes
#
#     y_test1 = y_test.copy().reset_index(drop=True)
#     y_test1['SUMMED_GMES'] = summed_gmes
#     y_test_melt = y_test1.melt(value_name='observed', id_vars='SUMMED_GMES')
#     y_test1['type'] = ['Observed'] * len(y_test1)
#
#     y_pred = pd.DataFrame(y_pred, columns=gdata.y_cols)
#     y_pred['SUMMED_GMES'] = summed_gmes
#     y_pred_melt = y_pred.melt(value_name='predicted', id_vars='SUMMED_GMES')
#     y_pred['type'] = ["Predicted"] * len(y_pred)
#
#     errors = y_test.reset_index(drop=True) - y_pred
#     errors['type'] = ['Error'] * len(errors)
#     errors['SUMMED_GMES'] = summed_gmes
#
#     out = pd.concat((y_test1, y_pred, errors))
#     out_melt = out.melt(id_vars=['SUMMED_GMES', 'type'], var_name="variable")
#     out_melt['value'] = out_melt['value'].astype('float64')
#
#     gamma_df = out_melt.loc[out_melt.variable.str.contains("_gDsRt"), :]
#     neutron_df = out_melt.loc[out_melt.variable.str.contains("_nDsRt"), :]
#
#     show_sensor_plot(gamma_df, title=f"Gammas vs Aggregate C100 Gradient\n{m_name}")
#     show_sensor_plot(neutron_df, title=f"Neutrons vs Aggregate C100 Gradient\n{m_name}")
#
#     diag_df = y_test_melt.copy()
#     diag_df['predicted'] = y_pred_melt['predicted'].astype('float64')
#     diag_df['observed'] = diag_df['observed'].astype('float64')
#     g_diag_df = diag_df.loc[diag_df.variable.str.contains("_gDsRt"), :]
#     n_diag_df = diag_df.loc[diag_df.variable.str.contains("_nDsRt"), :]
#
#     show_pred_vs_obs_plot(g_diag_df, f"Gamma Predicted Vs. Observed (rem/h)\n{m_name}")
#     show_pred_vs_obs_plot(n_diag_df, f"Neutron Predicted Vs. Observed (rem/h)\n{m_name}")
#
#     show_radiation_percent_error(g_diag_df, f"Gamma %Error\n{m_name}", x='observed', xlab="Observed (rem/h)",
#                                  trimmed=False)
#     show_radiation_percent_error(g_diag_df, f"Gamma %Error Trimmed\n{m_name}", x='observed',
#                                  xlab="Observed (rem/h)", trimmed=True)
#     show_radiation_percent_error(g_diag_df, f"Gamma %Error\n{m_name}", x='SUMMED_GMES',
#                                  xlab="Total C100 Gradient (MV/m)", trimmed=False)
#     show_radiation_percent_error(g_diag_df, f"Gamma %Error Trimmed\n{m_name}", x='SUMMED_GMES',
#                                  xlab="Total C100 Gradient (MV/m)", trimmed=True)
#
#     show_radiation_percent_error(n_diag_df, f"Neutron %Error\n{m_name}", x='observed', xlab="Observed (rem/h)",
#                                  trimmed=False)
#     show_radiation_percent_error(n_diag_df, f"Neutron %Error Trimmed\n{m_name}", x='observed',
#                                  xlab="Observed (rem/h)", trimmed=True)
#     show_radiation_percent_error(n_diag_df, f"Neutron %Error\n{m_name}", x='SUMMED_GMES',
#                                  xlab="Total C100 Gradient (MV/m)", trimmed=False)
#     show_radiation_percent_error(n_diag_df, f"Neutron %Error Trimmed\n{m_name}", x='SUMMED_GMES',
#                                  xlab="Total C100 Gradient (MV/m)", trimmed=True)


# def show_sensor_plot(df: pd.DataFrame, title):
#     plt.figure(figsize=(10, 10))
#
#     g = sns.FacetGrid(data=df, col='variable', row='type', hue='type', aspect=1.5, height=1.5)
#     g.map_dataframe(sns.scatterplot, x='SUMMED_GMES', y='value', alpha=0.1)
#     g.set_axis_labels("Total C100 Gradient (MV/m)", "rem/h")
#     for (row_key, col_key), ax in g.axes_dict.items():
#         ax.set_title(f"\n{col_key[3:7]} {row_key}", loc='center', pad=-20)
#         # ax.invert_yaxis()
#
#     g.fig.subplots_adjust(top=0.9)
#     g.fig.suptitle(title)
#     plt.show()
#     return g


# def const_line(*args, **kwargs):
#     ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     top = max(ylim + xlim)
#     x = [0, top]
#     y = x
#     plt.plot(x, y, color='k')


# def show_pred_vs_obs_plot(df, title):
#     plt.figure(figsize=(10, 10))
#
#     g = sns.FacetGrid(data=df, col='variable', hue='variable', aspect=1, height=3, sharex=False, sharey=False)
#     g.map_dataframe(sns.scatterplot, x='observed', y='predicted', alpha=0.05)
#     g.map(const_line)
#     g.set_axis_labels("Observed (rem/h)", "Predicted (rem/h)")
#     for col_key, ax in g.axes_dict.items():
#         ax.set_title(f"\n{col_key[3:7]}", loc='center', pad=-20)
#         # ax.invert_yaxis()
#
#     g.fig.subplots_adjust(top=0.9)
#     g.fig.suptitle(title)
#     plt.show()
#     return g


# def trim_outliers(df: pd.DataFrame, iqr_mult=1.5, cols: Optional[List[str]] = None) -> pd.DataFrame:
#     """Returns a copy of the DataFrame with outliers removed."""
#
#     df = df.copy()
#     if cols is None:
#         Q1 = df.quantile(0.25)
#         Q3 = df.quantile(0.75)
#         IQR = Q3 - Q1
#         df = df[~((df < (Q1 - iqr_mult * IQR)) | (df > (Q3 + iqr_mult * IQR))).any(axis=1)]
#     else:
#         Q1 = df[cols].quantile(0.25)
#         Q3 = df[cols].quantile(0.75)
#         IQR = Q3 - Q1
#         df = df[~((df[cols] < (Q1 - iqr_mult * IQR)) | (df[cols] > (Q3 + iqr_mult * IQR))).any(axis=1)]
#
#     return df


# def set_ylim_symmetric(*args, **kwargs):
#     ax = plt.gca()
#     y = np.max(np.fabs(ax.get_ylim()))
#     ax.set_ylim([-y, y])


# def set_grid_on(*args, **kwargs):
#     ax = plt.gca()
#     ax.grid(b=True, which='major', color='0')
#     ax.grid(b=True, which='minor', color='0.5')


# def show_radiation_percent_error(df, title, trimmed=False, x: str = 'SUMMED_GMES', xlab=None):
#     """Show the percent error in radiation predictions across the range of observed values.
#
#     Args:
#         df: The data frame of data.  Assumed to have 'observed' and 'predicted' columns
#         title: The title to put on the entire figure (suptitle)
#         trimmed: Should outliers be removed.  Outliers :=  > +/- 1.5 IQR.
#         x: The column to use as the X-axis.
#         xlab: The label for the X-axis.
#     """
#     plt.figure(figsize=(10, 10))
#
#     df = df.copy()
#     df['p_err'] = (df.predicted - df.observed) / df.observed
#
#     if trimmed:
#         df = trim_outliers(df, iqr_mult=3.0, cols=['p_err'])
#
#     g = sns.FacetGrid(data=df, col='variable', hue='variable', aspect=1, height=3, sharex=False, sharey=False)
#     g.map_dataframe(sns.scatterplot, x=x, y='p_err', alpha=0.05)
#     g.map(set_ylim_symmetric)
#     g.map(set_grid_on)
#     g.set_axis_labels(xlab, "Prediction %Error")
#     for col_key, ax in g.axes_dict.items():
#         ax.set_title(f"\n{col_key[3:7]}", loc='center', pad=-20)
#
#     g.fig.subplots_adjust(top=0.9)
#     g.fig.suptitle(title)
#     plt.show()
#     return g
