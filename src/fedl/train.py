from typing import Optional

from datetime import datetime
import math

import numpy as np

from . import data
import mlflow
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, mean_squared_error


def train(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
          num_epochs: int, optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.Module, save_file: str,
          tb_log_dir: str, lr_scheduler: Optional[optim.lr_scheduler.StepLR] = None,
          start_epoch: int = 0, params: Optional = None, show=True) -> torch.nn.modules.Module:
    """Train the given module on the given data and other parameters.

    Note: criterion is a standard pytorch _Loss object.
    """
    best_val_r2 = -math.inf
    if save_file is None:
        save_file = f"{model.__class__.__name__}.{datetime.now().strftime('%Y-%m-%d_%H%M%S')}-state-dict"

    tb = SummaryWriter(log_dir=tb_log_dir)
    num_batches = len(train_loader)

    best_model_epoch = None
    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss = []
        if show:
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

            train_loss.append(loss.item())
            if batch_idx % math.ceil(num_batches / 10.) == 0 and show:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss={loss.item():.4f}")

        train_loss = np.mean(train_loss)
        # Step the learning rate scheduler along
        learn_rate = optimizer.param_groups[0]['lr']
        if lr_scheduler is not None:
            if type(lr_scheduler).__name__ == "ReduceLROnPlateau":
                learn_rate = optimizer.param_groups[0]['lr']
                lr_scheduler.step(val_loss)
            else:
                learn_rate = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

        if params is not None and epoch % params.log_interval == 0:
            # If I don't do this after the epoch, then validation loss is consistently lower than train loss.
            output = evaluate_model(model, train_loader, device, criterion, multioutput=True)
            train2_loss, train2_r2, train_mmse, train_mr2 = output['loss'], output['r2'], output['multi_mse'], output[
                'multi_r2']

            output = evaluate_model(model, val_loader, device, criterion, multioutput=True)
            val_loss, val_r2, val_mmse, val_mr2 = output['loss'], output['r2'], output['multi_mse'], output['multi_r2']

            log_metrics(model, tb, epoch, train2_loss, val_loss, train2_r2, val_r2, train_mr2, val_mr2, train_mmse,
                        val_mmse, learn_rate, params)
        else:
            # Log the simple train loss metrics
            mlflow.log_metric("Train Loss", train_loss)

        # Print out some metrics as we go
        if show:
            print(f"lr = {learn_rate}")
            print(f"Train / Val Loss = {train2_loss:.4f} / {val_loss:.4f}")
            print(f"Train / Val R2 = {train2_r2:.4f} / {val_r2:.4f}")

        # Update the save file every time we get a better model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), save_file)
            best_model_epoch = epoch
            if show:
                print("Next best model!  Updating save file.")

    if show:
        print(f"\n\nLoading up best weights into model.  Best epoch={best_model_epoch}. {save_file}")
    model.load_state_dict(torch.load(save_file))

    return model


def log_metrics(model, tb, epoch, train_loss, val_loss, train_r2, val_r2, train_mr2, val_mr2, train_mmse, val_mmse,
                learn_rate, params) -> None:

    for name, weight in model.named_parameters():
        tb.add_histogram(name, weight, epoch)
        tb.add_histogram(f"{name}.grad", weight.grad, epoch)

    loss_scalars = {'Train Loss': train_loss, 'Val Loss': val_loss}
    r2_scalars = {'Train R2': train_r2, 'Val R2': val_r2}

    # Track metrics in tensorboard.
    tb.add_scalars("Loss", loss_scalars, epoch)
    tb.add_scalars("R2", r2_scalars, epoch)
    tb.add_scalar("Learning Rate", learn_rate, epoch)

    # Track metrics in MLFlow
    mlflow.log_metrics(loss_scalars, epoch)
    mlflow.log_metrics(r2_scalars, epoch)
    mlflow.log_metric("Learning Rate", learn_rate, epoch)

    # Try to put better names to individual output metrics if possible.  Then log them to MLflow and tensorboard.
    names = [f"out-{i}" for i in train_mr2.keys()]
    ndx_names = names
    if params is not None:
        ndx_names = data.get_ndx_columns(params.linac, params.radiation_zones)

    t_mr2 = {}
    v_mr2 = {}
    t_mmse = {}
    v_mmse = {}
    for name in train_mr2.keys():
        t_mr2[f"Train R2 {ndx_names[int(name)]}"] = train_mr2[name]
        v_mr2[f"Val R2 {ndx_names[int(name)]}"] = val_mr2[name]
        t_mmse[f"Train MSE {ndx_names[int(name)]}"] = train_mmse[name]
        v_mmse[f"Val MSE {ndx_names[int(name)]}"] = val_mmse[name]

    tb.add_scalars("Train Individual R2", t_mr2, epoch)
    tb.add_scalars("Val Individual R2", v_mr2, epoch)
    tb.add_scalars("Train Individual MSE", t_mmse, epoch)
    tb.add_scalars("Val Individual MSE", v_mmse, epoch)
    mlflow.log_metrics(t_mr2, epoch)
    mlflow.log_metrics(v_mr2, epoch)
    mlflow.log_metrics(t_mmse, epoch)
    mlflow.log_metrics(v_mmse, epoch)


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
