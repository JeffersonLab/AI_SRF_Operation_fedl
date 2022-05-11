import math
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
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


def train_baseline_trip_model(include_scan_data: bool = False):
    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Data To be used
    # gradient_data_file = 'nl-rf-ndx-trip-data-2021-11-06_2022-02-08.csv'
    # gradient_data_file = 'nl-rf-ndx-ONLY-nl-trips-with-ids-data-2021-11-06_2022-02-08.csv'
    gradient_data_file = 'nl-rf-ndx-power-event-and-ONLY-nl-trips-with-ids-data-2021-11-06_2022-02-08.csv'

    onset_file = None
    data_section = 'train'
    split = 'level_0'
    batch_size = 256
    ced_file = 'ced-data/ced-2021-11-06.tsv'
    linac = '1L'
    query = "Datetime < '2021-11-17' & Datetime > '2021-11-06'"  # Big fault happens on Nov 18/19

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

    print(f"Linac: {linac}")
    print(f"Number GMES columns: {len(gd.gmes_cols)}")
    print(f"Number NDX columns: {len(gd.ndx_cols)}")

    # Training details
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.9, step_size=50)
    criterion = MSELoss()
    num_epochs = 1000

    # Add the model to Tensorboard
    gmes, rad = next(iter(train_loader))
    gmes = gmes.to(device)
    tb = SummaryWriter(log_dir=tb_log_dir)
    tb.add_graph(model, gmes)
    tb.close()

    start_epoch = 0
    if include_scan_data:
        scan_data = data.GradientScanData(filename='processed_gradient-scan-2021-11-05_113837.646876.txt.csv',
                                          onset_file=onset_file, section='all', ced_file=ced_file, data_dir=data_dir,
                                          linac=linac)
        scan_data.load_data()
        t_loader, v_loader = scan_data.get_train_test(split='settle')
        start_epoch = 2001
        opt = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)
        model = train.train(model=model, train_loader=t_loader, val_loader=v_loader, device=device,
                            num_epochs=2000, optimizer=opt,
                            criterion=criterion, save_file=save_file,
                            tb_log_dir=tb_log_dir,
                            lr_scheduler=optim.lr_scheduler.StepLR(optimizer=opt, gamma=0.9, step_size=50))
        t_loader, v_loader, df_train, df_val = scan_data.get_train_test(split='settle', shuffle=False,
                                                                        provide_df=True)
        val_pred = utils.make_predictions(model=model, data_loader=v_loader, device=device, y_cols=scan_data.y_cols)
        val_true = df_val[scan_data.y_cols]
        utils.print_model_scores(*utils.score_model(y_pred=val_pred, y_test=val_true, multioutput='raw_values'),
                                 set_name='Scan Test Set')

    print('Training model')
    model = train.train(model=model, train_loader=train_loader, val_loader=val_loader, device=device,
                        num_epochs=num_epochs, optimizer=optimizer, criterion=criterion, save_file=save_file,
                        tb_log_dir=tb_log_dir, lr_scheduler=lr_scheduler, start_epoch=start_epoch)

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


def evaluate_model_by_cm(model: nn.Module, gd, device, base_data_filter: Optional[str] = None):
    cm_names = [f"R1{z}" for z in '23456789ABCDEFGHIJKLMNOP']

    names = []
    r2_list = []
    mse_list = []
    mae_list = []
    rel_error_list = []
    n_examples = []
    for cm in cm_names:
        # Re-filter the data to include only the zone and any other base filtering
        gd.filter_data(None)
        if base_data_filter is not None:
            gd.filter_data(query=base_data_filter)
        gd.filter_data(f"faulted_zones.str.contains('{cm}')")
        data_loader = gd.get_data_loader(shuffle=False)
        n = len(data_loader.dataset.X)
        if n <= 1:
            r2, mse, mae, rel_error = math.nan, math.nan, math.nan, math.nan
        else:
            y_pred = utils.make_predictions(model=model, data_loader=data_loader, device=device, y_cols=gd.y_cols)
            y_pred = gd.unnormalize_radiation(y_pred)
            y_true = gd.unnormalize_radiation(gd.y)
            rel_error = np.mean((y_pred.values - y_true.values) / y_true.values)

            r2, mse, mae = utils.score_model(y_pred=y_pred, y_test=y_true, multioutput='uniform_average')

        names.append(cm)
        r2_list.append(r2)
        mse_list.append(mse)
        mae_list.append(mae)
        rel_error_list.append(rel_error)
        n_examples.append(n)

    r2_df = pd.DataFrame([r2_list], columns=names)
    mse_df = pd.DataFrame([mse_list], columns=names)
    mae_df = pd.DataFrame([mae_list], columns=names)
    rel_error_df = pd.DataFrame([rel_error_list], columns=names)
    n_df = pd.DataFrame([n_examples], columns=names)
    return r2_df, mse_df, mae_df, rel_error_df, n_df


def one_off_evaluation_on_test_set():
    device = torch.device("cpu") if not torch.has_cuda else torch.device("cuda:0")
    model = models.MLPRelu(192, 20, [1024, 512]).to(device)
    model.load_state_dict(torch.load("models/baseline_trip_model-training_all_trip_data-state-dict"))

    gradient_data_file = 'nl-rf-ndx-ONLY-nl-trips-with-ids-data-2021-11-06_2022-02-08.csv'
    onset_file = None
    data_section = 'test'
    ced_file = 'ced-data/ced-2021-11-06.tsv'
    linac = '1L'
    query = "Datetime < '2021-11-17' & Datetime > '2021-11-06'"  # This is the first two weeks of trip data.

    gd = data.OperationsData(filename=gradient_data_file, onset_file=onset_file, section=data_section,
                             ced_file=ced_file, data_dir=data_dir, linac=linac)
    gd.meta_cols = gd.meta_cols + ['faulted_zones']
    print(gd.meta_cols)
    gd.load_data()
    gd.filter_data(query=query)
    test_loader = gd.get_data_loader(shuffle=False)

    evaluate_model(model=model, data_loader=test_loader, gd=gd, device=device, egain=gd.df.EGAIN, dtime=gd.df.Datetime,
                   set_name="Test Set Baseline NL Trips")
    r2_df, mse_df, mae_df, rel_error_df, n_df = evaluate_model_by_cm(model=model, gd=gd, device=device,
                                                                     base_data_filter=query)
    print("R2")
    print(r2_df)
    print("MSE")
    print(mse_df)
    print("MAE")
    print(mae_df)
    print("Rel Error")
    print(rel_error_df)
    print("# Examples")
    print(n_df)


def run_model_through_time(unit: str = 'week', update: bool = False, update_epochs=1000, lr=5e-3,
                           start: datetime = datetime.strptime('2021-11-17', '%Y-%m-%d'),
                           finish: datetime = datetime.strptime('2022-01-15', '%Y-%m-%d')):
    device = torch.device("cpu") if not torch.has_cuda else torch.device("cuda:0")

    # 192 = 8*24 cavities, 20 = 2 * 10 NDX signals
    hidden_layers = [1024, 512]
    model = models.MLPRelu(n_inputs=192, n_outputs=20, layers=hidden_layers).to(device)
    # model.load_state_dict(torch.load("models/baseline_trip_model-training_all_trip_data-state-dict"))
    # model.load_state_dict(torch.load("models/baseline_trip_model-training_nl_trips_only-state-dict"))
    model.load_state_dict(torch.load("models/baseline_trip_model-scan_and_nl_trip_data-state-dict"))

    #gradient_file = 'nl-rf-ndx-ONLY-nl-trips-with-ids-data-2021-11-06_2022-02-08.csv'
    gradient_file = 'nl-rf-ndx-power-event-and-ONLY-nl-trips-with-ids-data-2021-11-06_2022-02-08.csv'
    gdata = data.OperationsData(filename=gradient_file, onset_file=None, section='all',
                                ced_file='ced-data/ced-2021-11-06.tsv',
                                data_dir="../../../fe-data", linac='1L')
    gdata.meta_cols = gdata.meta_cols + ['faulted_zones']
    gdata.load_data()

    # Get the baseline performance of the model
    gdata.filter_data("Datetime > '2021-11-06' & Datetime < '2021-11-17'")
    data_loader = gdata.get_data_loader(shuffle=False)
    y_pred = utils.make_predictions(model=model, data_loader=data_loader, device=device, y_cols=gdata.y_cols)
    y_pred = gdata.unnormalize_radiation(y_pred)
    y_true = gdata.unnormalize_radiation(gdata.y)
    mr2, mmse, mmae = utils.score_model(y_test=y_true, y_pred=y_pred, multioutput='raw_values')
    r2, mse, mae = utils.score_model(y_test=y_true, y_pred=y_pred, multioutput='uniform_average')
    utils.print_model_scores(r2=r2, mse=mse, mae=mae, set_name="Training Range")
    utils.print_model_scores(r2=mr2, mse=mmse, mae=mmae, set_name="Training Range")

    if unit == 'week':
        time_step = timedelta(weeks=1)
    elif unit == 'day':
        time_step = timedelta(days=1)
    else:
        raise ValueError(f"Unsupported update value: '{unit}'")

    scores = []
    cm_r2 = []
    cm_mse = []
    cm_mae = []
    cm_rel_error = []
    cm_n_examples = []
    y_pred_list = []
    y_true_list = []
    cols = ['begin', 'end', 'R2', 'MSE', 'MAE']
    for score in ['R2', 'MSE', 'MAE']:
        for rad in ['N', 'G']:
            for zone in ['1L05', '1L06', '1L07', '1L08', '1L22', '1L23', '1L24', '1L25', '1L26', '1L27']:
                cols.append(f"{score}_{zone}_{rad}")
    now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    tb_log_dir = f"{tensorboard_base_dir}/trip_history_training/run-{now_string}"
    out_dir = f"out-run-{now_string}"
    os.mkdir(tb_log_dir)
    os.mkdir(out_dir)

    steps_run = 0
    while (start + time_step) < finish:
        # Start of probable loop
        begin_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = (start + time_step).strftime("%Y-%m-%d %H:%M:%S")
        print(begin_str)

        # Clear the filter, apply another, then get a data loader
        date_filter = f"Datetime < '{end_str}' & Datetime > '{begin_str}'"
        gdata.filter_data(query=None)
        gdata.filter_data(date_filter)
        data_loader = gdata.get_data_loader(shuffle=False)

        # Only evaluate / train if we have more than a few data points
        if len(gdata.df) < 5:
            print("Skipping this time period due to low available data.")
            start = start + time_step
            continue

        # Evaluate the model and store the scores
        y_pred = utils.make_predictions(model=model, data_loader=data_loader, device=device, y_cols=gdata.y_cols)

        y_pred = gdata.unnormalize_radiation(y_pred)
        y_true = gdata.unnormalize_radiation(gdata.y)

        y_pred_list.append(y_pred.set_index(gdata.df.Datetime))
        y_true_list.append(y_true.set_index(gdata.df.Datetime))

        mr2, mmse, mmae = utils.score_model(y_test=y_true, y_pred=y_pred, multioutput='raw_values')
        r2, mse, mae = utils.score_model(y_test=y_true, y_pred=y_pred, multioutput='uniform_average')
        scores.append([start, start + time_step, r2, mse, mae, *mr2.tolist(), *mmse.tolist(), *mmae.tolist()])

        # Now run the model against the data split along CMs
        r2_df, mse_df, mae_df, rel_error_df, n_df = evaluate_model_by_cm(model=model, gd=gdata, device=device,
                                                                         base_data_filter=date_filter)

        r2_df['begin'] = [start]
        mse_df['begin'] = [start]
        mae_df['begin'] = [start]
        rel_error_df['begin'] = [start]
        n_df['begin'] = [start]

        cm_r2.append(r2_df)
        cm_mse.append(mse_df)
        cm_mae.append(mae_df)
        cm_rel_error.append(rel_error_df)
        cm_n_examples.append(n_df)

        # Update the model
        if update:
            gdata.filter_data(query=None)
            gdata.filter_data(date_filter)
            train_loader, val_loader = gdata.get_train_test(split=None)
            model = train.train(model, train_loader=train_loader, val_loader=val_loader, device=device,
                                num_epochs=update_epochs,
                                optimizer=optim.SGD(params=model.parameters(), lr=lr, momentum=0.9),
                                criterion=nn.MSELoss(),
                                save_file='temp-state-file', tb_log_dir=tb_log_dir, start_epoch=steps_run*update_epochs)

        steps_run += 1
        start = start + time_step

    score_df = pd.DataFrame(scores, columns=cols)
    score_df = score_df.set_index(score_df.begin)
    score_df.to_csv(out_dir + "/scores.csv")
    print(score_df)

    cm_r2_df = pd.concat(cm_r2)
    cm_r2_df = cm_r2_df.set_index('begin')
    cm_r2_df.to_csv(out_dir + "/cm_r2.csv")
    cm_mse_df = pd.concat(cm_mse)
    cm_mse_df = cm_mse_df.set_index('begin')
    cm_mse_df.to_csv(out_dir + "/cm_mse.csv")
    cm_inv_mse_df = 1/cm_mse_df
    cm_inv_mse_df.to_csv(out_dir + "/cm_inv_mse.csv")
    cm_mae_df = pd.concat(cm_mae)
    cm_mae_df = cm_mae_df.set_index('begin')
    cm_mae_df.to_csv(out_dir + "/cm_mae.csv")
    cm_rel_error_df = pd.concat(cm_rel_error)
    cm_rel_error_df = cm_rel_error_df.set_index('begin')
    cm_rel_error_df.to_csv(out_dir + "/cm_rel_error.csv")
    cm_n_examples_df = pd.concat(cm_n_examples)
    cm_n_examples_df = cm_n_examples_df.set_index('begin')
    cm_n_examples_df.to_csv(out_dir + "/cm_n_examples.csv")

    line_types = [f"{p[0]}{p[1]}" for p in zip(['-', '--', '-.', ':'] * 6, ['o', 'v', '^', '*', 's', '+'] * 4)]
    cm_r2_df.plot(figsize=(10, 10), cmap='viridis', style=line_types)
    plt.title("R2 By CM Trip")
    plt.xlabel(f"{unit} Starting On")
    plt.show()

    cm_mse_df.plot(figsize=(10, 10), cmap='viridis', style=line_types)
    plt.title("MSE By CM Trip")
    plt.xlabel(f"{unit} Starting On")
    plt.show()

    cm_inv_mse_df.plot(figsize=(10, 10), cmap='viridis', style=line_types)
    plt.title("1/MSE By CM Trip")
    plt.xlabel(f"{unit} Starting On")
    plt.show()

    cm_mae_df.plot(figsize=(10, 10), cmap='viridis', style=line_types)
    plt.title("MAE By CM Trip")
    plt.xlabel(f"{unit} Starting On")
    plt.show()

    cm_rel_error_df.plot(figsize=(10, 10), cmap='viridis', style=line_types)
    plt.title("Avg. Rel Error By CM Trip")
    plt.xlabel(f"{unit} Starting On")
    plt.show()

    line_types = ['-'] * 3 + ['--'] * 3 + ['-.'] * 2 + [':'] * 2
    score_df.R2.plot(figsize=(6, 6))
    plt.title("Avg. Model R2")
    plt.xlabel(f"{unit} Starting On")
    plt.show()
    score_df.MSE.plot(figsize=(6, 6))
    plt.title("Avg. Model MSE")
    plt.xlabel(f"{unit} Starting On")
    plt.show()
    score_df.MAE.plot(figsize=(6, 6))
    plt.title("Avg. Model MAE")
    plt.xlabel(f"{unit} Starting On")
    plt.show()
    score_df[score_df.columns[score_df.columns.str.startswith("R2_") &
                              score_df.columns.str.endswith("N")]].plot(figsize=(10, 10), cmap='tab10',
                                                                        style=line_types)
    plt.title("Individual Neutron Output R2")
    plt.xlabel(f"{unit} Starting On")
    plt.show()
    score_df[score_df.columns[score_df.columns.str.startswith("MSE_") &
                              score_df.columns.str.endswith("N")]].plot(figsize=(10, 10), cmap='tab10',
                                                                        style=line_types)
    plt.title("Individual Neutron Output MSE")
    plt.xlabel(f"{unit} Starting On")
    plt.show()
    score_df[score_df.columns[score_df.columns.str.startswith("MAE_") &
                              score_df.columns.str.endswith("N")]].plot(figsize=(10, 10), cmap='tab10',
                                                                        style=line_types)
    plt.title("Individual Neutron Output MAE")
    plt.xlabel(f"{unit} Starting On")
    plt.show()
    score_df[score_df.columns[score_df.columns.str.startswith("R2_") &
                              score_df.columns.str.endswith("G")]].plot(figsize=(10, 10), cmap='tab10',
                                                                        style=line_types)
    plt.title("Individual Gamma Output R2")
    plt.xlabel(f"{unit} Starting On")
    plt.show()
    score_df[score_df.columns[score_df.columns.str.startswith("MSE_") &
                              score_df.columns.str.endswith("G")]].plot(figsize=(10, 10), cmap='tab10',
                                                                        style=line_types)
    plt.title("Individual Gamma Output MSE")
    plt.xlabel(f"{unit} Starting On")
    plt.show()
    score_df[score_df.columns[score_df.columns.str.startswith("MAE_") &
                              score_df.columns.str.endswith("G")]].plot(figsize=(10, 10), cmap='tab10',
                                                                        style=line_types)
    plt.title("Individual Gamma Output MAE")
    plt.xlabel(f"{unit} Starting On")
    plt.show()

    y_true = pd.concat(y_true_list).reset_index()
    y_pred = pd.concat(y_pred_list).reset_index()
    dtime = y_true.Datetime
    egain = pd.DataFrame({'EGAIN': [math.nan] * len(y_true)})
    y_true = y_true.drop(columns=['Datetime'])
    y_pred = y_pred.drop(columns=['Datetime'])

    g_df, n_df, g_diag_df, n_diag_df = utils.get_sensor_plot_data(y_true=y_true, y_pred=y_pred, egain=egain,
                                                                  dtime=dtime)
    utils.plot_predictions_over_time(g_df, title="Gamma Pred & Obs Over Time", log_y=True)
    utils.plot_predictions_over_time(n_df, title="Neutron Pred & Obs Over Time", log_y=True)

    return score_df


if __name__ == '__main__':
    train_baseline_trip_model(include_scan_data=True)
    # one_off_evaluation_on_test_set()
    run_model_through_time(unit='day', update=True, update_epochs=1000, lr=1e-3)
