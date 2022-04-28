from datetime import datetime
from typing import List

import torch
from torch import optim
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

import src.fedl.data as data
import src.fedl.utils as utils
import src.fedl.train as train
from . import models

import torch.nn as nn
import pathlib

parent_dir = pathlib.Path(__file__).parent.as_posix()
data_dir = f'{parent_dir}/../../../fe-data'

tb_log_dir = 'runs'

# def train_model(model_class: nn.Module, model_kw_args: dict, gradient_data_file: str, ced_file: str, data_dir: str,
#                 split: str, optimizer: optim.Optimizer.__class__, optimizer_kwargs: dict, criterion: nn.MSELoss,
#                 batch_size: int, lr_scheduler: optim.lr_scheduler.MultiStepLR.__class__, num_epochs: int,
#                 lr_scheduler_kwargs: Optional[dict], tb_log_dir: str, gmes_zones: List[str], rad_zones: List[str],
#                 linac: str, onset_file: Optional[str] = None, save_file: Optional[str] = None,
#                 data_section: str = 'train', model_state_file: str = None,
#                 data_filter: Optional[str] = None) -> None:
#     """Main method for training a model."""
#
#     device = torch.device("cpu") if not torch.has_cuda else torch.device("cuda:0")
#     print(f"Using device {device}")
#
#     gradient_data = data.OperationsData(filename=gradient_data_file, onset_file=onset_file, section=data_section,
#                                         ced_file=ced_file, data_dir=data_dir, gmes_zones=gmes_zones,
#                                         rad_zones=rad_zones, linac=linac)
#     # gradient_data = data.GradientScanData(filename=gradient_data_file, onset_file=onset_file, section=data_section,
#     #                                       ced_file=ced_file, data_dir=data_dir, gmes_zones=gmes_zones,
#     #                                       rad_zones=rad_zones, linac=linac)
#     gradient_data.load_data()
#     if data_filter is not None:
#         gradient_data.filter_data(query=data_filter)
#     train_loader, val_loader = gradient_data.get_train_test(split=split, train_size=0.75, batch_size=batch_size)
#     print(len(train_loader), len(val_loader))
#
#     # Eight cavities per zone
#     n_inputs = len(gmes_zones * 8)
#     if onset_file is not None:
#         # Onsets will be included for each cavity, so double the size
#         n_inputs = n_inputs * 2
#
#     # Two radiation signals per ndx detector, and one ndx detector per zone
#     n_outputs = len(rad_zones * 2)
#     model = model_class(n_inputs=n_inputs, n_outputs=n_outputs, **model_kw_args)
#     if model_state_file is not None:
#         model.load_state_dict(torch.load(model_state_file))
#     model = model.to(device)
#     print(f"model: {model}")
#
#     optimizer = optimizer(model.parameters(), **optimizer_kwargs)
#     if lr_scheduler is not None:
#         lr_scheduler = lr_scheduler(optimizer, **lr_scheduler_kwargs)
#
#     gmes, rad = next(iter(train_loader))
#     gmes = gmes.to(device)
#     tb = SummaryWriter(log_dir=tb_log_dir)
#     tb.add_graph(model, gmes)
#     tb.close()
#
#     print('About to train model')
#     model = train.train(model=model, train_loader=train_loader, val_loader=val_loader, device=device,
#                         num_epochs=num_epochs, optimizer=optimizer, criterion=criterion, save_file=save_file,
#                         tb_log_dir=tb_log_dir, lr_scheduler=lr_scheduler)
#     # model = train_model(split=split, tb_log_dir=tb_log_dir, gradient_data=gradient_data, model=model,
#     #                     optimizer=optimizer, lr_scheduler=lr_scheduler, criterion=criterion, **kwargs)
#     train.plot_results(model=model, split=split, gdata=gradient_data)


def train_baseline_trip_model():
    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Data To be used
    gradient_data_file = 'nl-rf-ndx-trip-data-2021-11-06_2022-02-08.csv'
    onset_file = None
    data_section = 'train'
    split = 'level_0'
    batch_size = 256
    ced_file = 'ced-data/ced-2021-11-06.tsv'
    linac = '1L'
    query = "Datetime < '2021-11-18' & Datetime > '2021-11-06'"     # This is the first two weeks of trip data.

    device = torch.device("cpu") if not torch.has_cuda else torch.device("cuda:0")
    print(f"Using device {device}")

    # 192 = 8*24 cavities, 20 = 2 * 10 NDX signals
    hidden_layers = [1024, 512]
    model = models.MLPRelu(n_inputs=192, n_outputs=20, layers=hidden_layers)
    # Where the model will be saved
    layers_str = "_".join([str(i) for i in hidden_layers])
    save_file = f'baseline_trip_model-{layers_str}-{now_str}-state-dict'

    gd = data.OperationsData(filename=gradient_data_file, onset_file=onset_file, section=data_section,
                             ced_file=ced_file, data_dir=data_dir, linac=linac)
    gd.load_data()
    gd.filter_data(query=query)
    train_loader, val_loader = gd.get_train_test(split=split, train_size=0.75, batch_size=batch_size)
    print(len(train_loader), len(val_loader))

    print(f"Linac: {linac}")
    print(f"Number GMES columns: {len(gd.gmes_cols)}")
    print(f"Number NDX columns: {len(gd.ndx_cols)}")

    # Training details
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = None
    criterion = MSELoss()
    num_epochs = 1000

    # I'm not sure what the range for all of these radiation signals are.  Keep it simple as gamma are typically 10x
    # neutron.
    data.max_gamma_rad_per_hour = 10
    data.max_neutron_rad_per_hour = 1

    # Add the model to Tensorboard
    gmes, rad = next(iter(train_loader))
    gmes = gmes.to(device)
    tb = SummaryWriter(log_dir=tb_log_dir + f'/baseline_trip_model/run-{datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
    tb.add_graph(model, gmes)
    tb.close()

    print('Training model')
    model = train.train(model=model, train_loader=train_loader, val_loader=val_loader, device=device,
                        num_epochs=num_epochs, optimizer=optimizer, criterion=criterion, save_file=save_file,
                        tb_log_dir=tb_log_dir, lr_scheduler=lr_scheduler)

    train_loader, val_loader, df_train, df_val = gd.get_train_test(split=split, train_size=0.75, batch_size=batch_size,
                                                                   shuffle=False, provide_df=True)

    y_pred = utils.make_predictions(model=model, data_loader=val_loader, device=device, y_cols=gd.y_cols)
    data.report_performance(y_pred, val_loader.dataset.y, egain=df_val.EGAIN, dtime=df_val.Datetime,
                            set_name="Trip Val")
