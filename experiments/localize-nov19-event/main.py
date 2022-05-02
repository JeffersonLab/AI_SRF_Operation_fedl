from datetime import datetime, timedelta

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.fedl.data as data
import src.fedl.utils as utils
import src.fedl.train as train
import models

import torch.nn as nn
import pathlib

parent_dir = pathlib.Path(__file__).parent.as_posix()
data_dir = f'{parent_dir}/../../../fe-data'

tensorboard_base_dir = 'runs'


def train_baseline_trip_model():
    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Data To be used
    # gradient_data_file = 'nl-rf-ndx-trip-data-2021-11-06_2022-02-08.csv'
    gradient_data_file = 'nl-rf-ndx-ONLY-nl-trips-with-ids-data-2021-11-06_2022-02-08.csv'
    onset_file = None
    data_section = 'train'
    split = 'level_0'
    batch_size = 256
    ced_file = 'ced-data/ced-2021-11-06.tsv'
    linac = '1L'
    query = "Datetime < '2021-11-18' & Datetime > '2021-11-06'"  # This is the first two weeks of trip data.

    tb_log_dir = f'{tensorboard_base_dir}/baseline_trip_model/run-{datetime.now().strftime("%Y-%m-%d_%H%M%S")}'
    device = torch.device("cpu") if not torch.has_cuda else torch.device("cuda:0")
    print(f"Using device {device}")

    # 192 = 8*24 cavities, 20 = 2 * 10 NDX signals
    hidden_layers = [1024, 512]
    model = models.MLPRelu(n_inputs=192, n_outputs=20, layers=hidden_layers).to(device)
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
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.9, step_size=50)
    criterion = MSELoss()
    # num_epochs = 1000
    num_epochs = 2

    # Add the model to Tensorboard
    gmes, rad = next(iter(train_loader))
    gmes = gmes.to(device)
    tb = SummaryWriter(log_dir=tb_log_dir)
    tb.add_graph(model, gmes)
    tb.close()

    print('Training model')
    model = train.train(model=model, train_loader=train_loader, val_loader=val_loader, device=device,
                        num_epochs=num_epochs, optimizer=optimizer, criterion=criterion, save_file=save_file,
                        tb_log_dir=tb_log_dir, lr_scheduler=lr_scheduler)

    train_loader, val_loader, df_train, df_val = gd.get_train_test(split=split, train_size=0.75, batch_size=batch_size,
                                                                   shuffle=False, provide_df=True)

    evaluate_model(model=model, data_loader=val_loader, device=device, gd=gd, egain=df_val.EGAIN, dtime=df_val.Datetime,
                   set_name='Baseline Trip Data')


def evaluate_model(model: nn.Module, data_loader: DataLoader, gd, device, egain, dtime, set_name):
    y_pred = utils.make_predictions(model=model, data_loader=data_loader, device=device, y_cols=gd.y_cols)
    y_true = pd.DataFrame(data_loader.dataset.y.cpu().numpy(), columns=gd.y_cols)

    y_pred = gd.unnormalize_radiation(y_pred)
    y_true = gd.unnormalize_radiation(y_true)

    data.report_performance(y_pred=y_pred, y_true=y_true, egain=egain, dtime=dtime, set_name=set_name,
                            unnormalize=False)


def one_off_evaluation_on_test_set():
    device = torch.device("cpu") if not torch.has_cuda else torch.device("cuda:0")
    model = models.MLPRelu(192, 20, [1024, 512]).to(device)
    model.load_state_dict(torch.load("models/baseline_trip_model-training_all_trip_data-state-dict"))

    gradient_data_file = 'nl-rf-ndx-ONLY-nl-trips-with-ids-data-2021-11-06_2022-02-08.csv'
    onset_file = None
    data_section = 'test'
    ced_file = 'ced-data/ced-2021-11-06.tsv'
    linac = '1L'
    query = "Datetime < '2021-11-18' & Datetime > '2021-11-06'"  # This is the first two weeks of trip data.

    gd = data.OperationsData(filename=gradient_data_file, onset_file=onset_file, section=data_section,
                             ced_file=ced_file, data_dir=data_dir, linac=linac)
    gd.load_data()
    gd.filter_data(query=query)
    test_loader = gd.get_data_loader(shuffle=False)

    evaluate_model(model=model, data_loader=test_loader, gd=gd, device=device, egain=gd.df.EGAIN, dtime=gd.df.Datetime,
                   set_name="Baseline Trip Data Only NL Trips")


def run_model_through_time(update: str = 'weekly', start: datetime = datetime.strptime('2021-11-18', '%Y-%m-%d'),
                           finish: datetime = datetime.strptime('2021-11-30', '%Y-%m-%d')):
    device = torch.device("cpu") if not torch.has_cuda else torch.device("cuda:0")

    # 192 = 8*24 cavities, 20 = 2 * 10 NDX signals
    hidden_layers = [1024, 512]
    model = models.MLPRelu(n_inputs=192, n_outputs=20, layers=hidden_layers).to(device)
    model.load_state_dict(torch.load("models/baseline_trip_model-training_all_trip_data-state-dict"))

    gdata = data.OperationsData(filename='nl-rf-ndx-ONLY-nl-trips-with-ids-data-2021-11-06_2022-02-08.csv',
                                onset_file=None, section='all', ced_file='ced-data/ced-2021-11-06.tsv',
                                data_dir="../../../fe-data", linac='1L')
    gdata.load_data(match_scaling=True)

    if update == 'weekly':
        time_step = timedelta(weeks=1)
    elif update == 'daily':
        time_step = timedelta(days=1)
    else:
        raise ValueError(f"Unsupported update value: '{update}'")

    scores = []
    cols = ['begin', 'end', 'R2', 'MSE', 'MAE']
    for score in ['R2', 'MSE', 'MAE']:
        for rad in ['N', 'G']:
            for zone in ['1L05', '1L06', '1L07', '1L08', '1L22', '1L23', '1L24', '1L25', '1L26', '1L27']:
                cols.append(f"{score}_{zone}_{rad}")

    while (start + time_step) < finish:
        # Start of probable loop
        begin_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = (start + time_step).strftime("%Y-%m-%d %H:%M:%S")

        # Clear the filter, apply another, then get a data loader
        gdata.filter_data(query=None)
        gdata.filter_data(f"Datetime < '{end_str}' & Datetime > '{begin_str}'")
        data_loader = gdata.get_data_loader(shuffle=False)

        # Evaluate the model and store the scores
        y_pred = utils.make_predictions(model=model, data_loader=data_loader, device=device, y_cols=gdata.y_cols)
        mr2, mmse, mmae = utils.score_model(y_test=gdata.y, y_pred=y_pred, multioutput='raw_values')
        r2, mse, mae = utils.score_model(y_test=gdata.y, y_pred=y_pred, multioutput='uniform_average')
        scores.append([start, start+time_step, r2, mse, mae, *mr2.tolist(), *mmse.tolist(), *mmae.tolist()])

        start = start + time_step

    score_df = pd.DataFrame(scores, columns=cols)

    print(score_df)
    score_df.R2.plot()
    plt.show()
    score_df.MSE.plot()
    plt.show()
    score_df.MAE.plot()
    plt.show()
    score_df[score_df.columns[score_df.columns.str.startswith("R2_")]].plot()
    plt.show()
    score_df[score_df.columns[score_df.columns.str.startswith("MSE_")]].plot()
    plt.show()
    score_df[score_df.columns[score_df.columns.str.startswith("MAE_")]].plot()
    plt.show()
    return score_df


if __name__ == '__main__':
    # train_baseline_trip_model()
    run_model_through_time()
