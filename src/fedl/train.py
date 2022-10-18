from typing import Optional

from datetime import datetime
import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, mean_squared_error


def train(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
          num_epochs: int, optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.Module, save_file: str,
          tb_log_dir: str, lr_scheduler: Optional[optim.lr_scheduler.StepLR] = None,
          start_epoch: int = 0) -> torch.nn.modules.Module:
    """Train the given module on the given data and other parameters.

    Note: criterion is a standard pytorch _Loss object.
    """
    best_val_r2 = -math.inf
    if save_file is None:
        save_file = f"{model.__class__.__name__}.{datetime.now().strftime('%Y-%m-%d_%H%M%S')}-state-dict"

    print(f"in train(): device = {device}")
    tb = SummaryWriter(log_dir=tb_log_dir)
    num_batches = len(train_loader)

    best_model_epoch = None
    for epoch in range(start_epoch, start_epoch + num_epochs):
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

        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f"{name}.grad", weight.grad, epoch)

        # If I don't do this after the epoch, then validation loss is consistently lower than train loss.
        output = evaluate_model(model, train_loader, device, criterion, multioutput=True)
        train2_loss, train2_r2, train_mmse, train_mr2 = output['loss'], output['r2'], output['multi_mse'], output[
            'multi_r2']

        output = evaluate_model(model, val_loader, device, criterion, multioutput=True)
        val_loss, val_r2, val_mmse, val_mr2 = output['loss'], output['r2'], output['multi_mse'], output['multi_r2']

        learn_rate = optimizer.param_groups[0]['lr']

        # Step the learning rate scheduler along
        if lr_scheduler is not None:
            if type(lr_scheduler).__name__ == "ReduceLROnPlateau":
                learn_rate = optimizer.param_groups[0]['lr']
                lr_scheduler.step(val_loss)
            else:
                learn_rate = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

        # Track all of these metrics in tensorboard.
        loss_scalars = {'Train_Loss': train2_loss, 'Val Loss': val_loss}
        r2_scalars = {'Train R2': train2_r2, 'Val R2': val_r2}
        tb.add_scalars("Loss", loss_scalars, epoch)
        tb.add_scalars("R2", r2_scalars, epoch)
        tb.add_scalars("Train_Individual_R2", train_mr2, epoch)
        tb.add_scalars("Val_Individual_R2", val_mr2, epoch)
        tb.add_scalars("Train_Individual_MSE", train_mmse, epoch)
        tb.add_scalars("Val_Individual_MSE", val_mmse, epoch)
        tb.add_scalar("LR", learn_rate, epoch)

        # Print out some metrics as we go
        print(f"lr = {learn_rate}")
        print(f"Train / Val Loss = {train2_loss:.4f} / {val_loss:.4f}")
        print(f"Train / Val R2 = {train2_r2:.4f} / {val_r2:.4f}")

        # Update the save file every time we get a better model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), save_file)
            best_model_epoch = epoch
            print("Next best model!  Updating save file.")

    print(f"\n\nLoading up best weights into model.  Best epoch={best_model_epoch}. {save_file}")
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
