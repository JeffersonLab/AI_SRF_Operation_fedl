import os
from typing import Optional, Tuple, List, Union

import sklearn.model_selection
import torch
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# The highest a C100 can do is around 21.5 MV/m in practice and 25 MV/m (I think) per administrative limit.
from . import utils

c100_max_operational_gmes = 25


class FEData:
    """This class read data from disk and has methods for presenting data in pytorch friendly objects."""

    def __init__(self, filename: str, ced_file: str, onset_file: Optional[str] = None, section: str = 'train',
                 data_dir: str = "data", rad_zones: Optional[List[str]] = None,
                 gmes_zones: Optional[List[str]] = None, rad_suffix: str = '',
                 meta_cols: Optional[List[str]] = None, linac: str = '1L', scaler_x: Optional[MinMaxScaler] = None,
                 scaler_y: Optional[MinMaxScaler] = None) -> None:
        r"""Loads the gradient scan data for the given file name and section (train vs test).

        Args:
            filename: F containing gradient data and radiation response.  Relative to data_dir
            ced_file: A file containing CED data about the cavities in use.  Relative to data_dir
            onset_file: The name of the file containing radiation onset gradients for each cavity.  None for no onsets.
            section: can be 'train' or 'test'.  Determines the directory path that is taken to file name.
            data_dir: The path to the directory holding the data
            rad_zones: A list of zone names for which to include NDX signals
            gmes_zones: A list of zones for which to include cavity GMES data
            rad_suffix: A string to append to the end of the lookup string for radiation columns.  Helpful for lagged
                        variables
            meta_cols: A list containing column names that are metadata and should be excluded from X and y dataframes
            linac: The linac to model.  E.g., 1L or 2L.  Defaults to 1L.
        """
        self.filename = filename
        self.section = section
        self.data_dir = data_dir
        self.onset_file = onset_file
        self.ced_file = ced_file
        self.linac = linac

        # ID the columns to keep
        self.gmes_cols = get_gmes_columns(linac=linac, zones=gmes_zones)
        self.ndx_cols = get_ndx_columns(linac=linac, zones=rad_zones, suffix=rad_suffix)
        self.neutron_cols = [col for col in self.ndx_cols if "_nDsRt" in col]
        self.gamma_cols = [col for col in self.ndx_cols if "_gDsRt" in col]
        if meta_cols is None:
            self.meta_cols = ['Datetime']
        else:
            self.meta_cols = meta_cols

        self.X_cols = None
        self.y_cols = None

        # Placeholders for data to be loaded later
        self.df = None
        self.X = None
        self.y = None
        self.ced_df = None

        # If it's None, then we will make one.  If not None, we assume that it has been fit.
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    @staticmethod
    def normalize_data(df, scaler, fit=True):
        """Run Min-Max Scaling on the input and output data"""
        if fit:
            df = scaler.fit_transform(df)
        else:
            df = scaler.transform(df)

        return df

    @staticmethod
    def unnormalize_data(df, scaler):
        """Run Min-Max Scaling on the input and output data"""
        return scaler.inverse_transform(df)

    def load_data(self) -> None:
        """This loads the data from files and generates all downstream attributes of the FEData object.

        This can and should be overridden for subclasses.  This method

        1. Reads in the data_file, keeping meta_col, ndx_cols, gmes_cols
        2. If an onset file is supplied, it is read in add onsets are added to each row.
        3. Reads in the ced_file, and calculates a column, EGAIN, that is the total energy gain for all cavities.
        4. This normalizes the GMES by the c100_max_operational_gmes variable.  Typically, this equals 25.
        5
        """
        # Reduce the data to only the parts we need to keep for the model and or processing
        self.df = pd.read_csv(f"{self.data_dir}/{self.section}/{self.filename}")
        self.df = self.df[self.meta_cols + self.ndx_cols + self.gmes_cols].copy()
        self.X_cols = self.gmes_cols
        self.y_cols = self.ndx_cols

        # Add on the onset data if a file was specified
        if self.onset_file is not None:
            onset_df = self._get_onset_data()
            self.df = pd.concat((self.df, onset_df), axis=1)
            self.X_cols += onset_df.columns.to_list()

        # Some rows may have NaNs, especially if we include zones that weren't under study.  Drop those rows.
        for col in self.X_cols + self.y_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df[self.df[self.X_cols + self.y_cols].notna().all(axis=1)]

        # Read in the CED data and keep it in memory.  This should be a small ~400x4 DataFrame
        self.ced_df = pd.read_csv(f"{self.data_dir}/{self.ced_file}", sep='\t')

        # Add the EGAIN column.  Do this prior to normalization of gradients as it is not used in training
        self._add_energy_gain()

        # Normalize the GMES data to an operational range, and radiation so that gamma and neutron are of a similar unit
        # scale.

        # Set datetime dtypes
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'].str.replace('_', ' '), yearfirst=True)

        # Set aside the X and y for inputs and labels
        self.X = self.df[self.X_cols]
        self.y = self.df[self.y_cols]

    def _get_onset_data(self) -> pd.DataFrame:
        """Generate a DataFrame a cavity onset per column, repeated for 'length' rows.
        """
        # This file has two columns, Cavity ID, and onset gradient.  Convert it to be one row with column headers.
        df = pd.read_csv(f"{self.data_dir}/{self.onset_file}", sep='\t', comment='#', header=None).T
        df.columns = df.iloc[0]
        df.drop(df.index[0], inplace=True)
        df = df.astype('float32')

        # Only keep the NL C100s
        nl_cols = []
        for cav in range(1, 9):
            for zone in "MNOP":
                nl_cols.append(f"R1{zone}{cav}GSET")
        df = df[nl_cols]
        # 100 is the placeholder for ODVH limited.  Let's assume that they will not generate FE for the time being.
        df.replace(100, c100_max_operational_gmes, inplace=True)
        df = df.add_suffix("_ONSET").reset_index(drop=True)
        df = df.iloc[np.ndarray([1]).repeat(len(self.df)), :].reset_index(drop=True)

        return df

    def _add_energy_gain(self):
        """This adds an EGAIN column which is the sum of GMES*length across all cavities in an example."""
        energy_gain = np.zeros([len(self.df), ])
        for gmes_col in self.gmes_cols:
            epics_name = gmes_col[0:4]
            energy_gain += self.df[gmes_col].values * self.ced_df[self.ced_df['EPICSName'] == epics_name].length.values
        self.df['EGAIN'] = energy_gain

    def get_train_test(self, split: str = 'egain', train_size: float = 0.75, batch_size: int = 256, seed: int = 732,
                       shuffle: bool = True, provide_df: bool = True, scale: bool = True) \
            -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]]:
        """Get train and test splits as pytorch DataLoaders.  Subclasses will should override this as makes sense.

        This uses a basic randomized splitting approach.
        Args:
            split: How should the data be split.  FEData only supports 'egain' which stratifies data on the EGAIN column
            train_size: train_size for GroupShuffleSplit.  0.75 by default is used assuming that we want a 60/20/20
                        train/val/test split and that we have already pulled off the 20 for testing.
            batch_size: The batch size used in the DataLoaders
            seed: The value used to seed the data splitter's random_state
            shuffle: Should the DataLoaders reshuffle every epoch
            provide_df: Should the internal dataframe be split and returned
            scale: Should the data sets be scaled/normalized

        Returns:  train_dataloader, test_dataloader
        """

        df_train, df_test = None, None
        if split == 'egain':
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(self.X, self.y,
                                                                                        train_size=train_size,
                                                                                        stratify=self.df.EGAIN,
                                                                                        random_state=seed)
            if provide_df:
                df_train, df_test = sklearn.model_selection.train_test_split(self.df,
                                                                             train_size=train_size,
                                                                             stratify=self.df.EGAIN,
                                                                             random_state=seed)
        else:
            raise RuntimeError(f"Unsupported split argument '{split}'")

        if scale:
            # Make new scalers if needed
            fit = self.initialize_scalers()

            # Make sure to fit the data to the training set and not the test set
            X_train = self.normalize_data(X_train, self.scaler_x, fit=fit)
            y_train = self.normalize_data(y_train, self.scaler_y, fit=fit)
            X_test = self.normalize_data(X_test, self.scaler_x, fit=False)
            y_test = self.normalize_data(y_test, self.scaler_y, fit=False)

        train_loader = DataLoader(NDX_RF_Dataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(NDX_RF_Dataset(X_test, y_test), batch_size=batch_size, shuffle=shuffle)

        if provide_df:
            return train_loader, test_loader, df_train, df_test
        else:
            return train_loader, test_loader

    def initialize_scalers(self) -> bool:
        """Initialize scalers.  Return True if we had to make new ones.  False otherwise.  No fitting done."""
        if self.scaler_x is None or self.scaler_y is None:
            self.scaler_x = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
            return True
        return False

    def get_data_loader(self, batch_size: int = 256, shuffle: bool = True, scale: bool = True) -> DataLoader:
        """A method for return a DataLoader object comprised of all the data in the GradientData object.

        Args:
            batch_size: The batch size used in the DataLoader.
            shuffle:  Should the DataLoader shuffle the data.
            scale:  Should the data be scaled prior to use
        """
        X = self.X
        y = self.y
        if scale:
            fit = self.initialize_scalers()
            X = self.normalize_data(self.X, self.scaler_x, fit=fit)
            y = self.normalize_data(self.y, self.scaler_y, fit=fit)

        return DataLoader(NDX_RF_Dataset(X, y), batch_size=batch_size, shuffle=shuffle)

    def filter_data(self, query: Optional[str]):
        """Down select the data to only entries matching the query string.  This only affects the X and y attributes.

        Args:
            query: A string to pass to a DataFrame.query() call.  None implies to reset X and y back to all of df.
        """
        if query is None:
            tmp = self.df
        else:
            tmp = self.df.query(query)
        self.X = tmp[self.X_cols]
        self.y = tmp[self.y_cols]


class GradientScanData(FEData):
    """This class handles the peculiarities of the gradient scan data."""

    def __init__(self, filename: str, ced_file: str, onset_file: Optional[str] = None, section: str = 'train',
                 data_dir: str = "data", rad_zones: Optional[List[str]] = None,
                 gmes_zones: Optional[List[str]] = None, rad_suffix: str = '_lag-1', linac='1L') -> None:

        # Load up the data in the subclass
        super().__init__(filename=filename, ced_file=ced_file, onset_file=onset_file, section=section,
                         data_dir=data_dir, rad_zones=rad_zones, gmes_zones=gmes_zones, rad_suffix=rad_suffix,
                         meta_cols=['Datetime', 'sample_type', 'settle_start'], linac=linac)

    def load_data(self):
        # Set settle_start to be a datetime object
        super().load_data()
        self.df['settle_start'] = pd.to_datetime(self.df['settle_start'])

    def get_train_test(self, split: str = 'settle', train_size: float = 0.75, batch_size: int = 256, seed: int = 732,
                       shuffle: bool = True, provide_df: bool = True, scale: bool = True) \
            -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]]:
        """Get train and test splits as DataLoaders.
        Args:
            split: can be 'settle' or 'gradient_sorted'.  settle splits by grouping settle samples, gradient
                   sections out high from low gradients, to check extrapolating from low to high.
            train_size: train_size for GroupShuffleSplit.  0.75 by default is used assuming that we want a 60/20/20
                        train/val/test split and that we have already pulled off the 20 for testing.
            batch_size: The batch size used in the DataLoaders
            seed: The value used to seed the random_state used to split the train and test sets.
            shuffle: Should the DataLoaders reshuffle every epoch
            provide_df: Should the internal dataframe be split and returned

        Returns:  train_dataloader, test_dataloader
        """

        df_train, df_test = None, None
        if split == 'settle':
            gss = GroupShuffleSplit(train_size=train_size, random_state=seed, n_splits=2)
            train_idx, test_idx = next(gss.split(self.df, groups=self.df.settle_start))

            X_train = self.X.iloc[train_idx, :]
            y_train = self.y.iloc[train_idx, :]
            X_test = self.X.iloc[test_idx, :]
            y_test = self.y.iloc[test_idx, :]

            df_train = self.df.iloc[train_idx, :]
            df_test = self.df.iloc[test_idx, :]

        elif split == 'gradient_sorted':
            # Split out train and test on sorted C100 Egain (this is proportional to the summed gradient)
            egain_categories = pd.qcut(self.df['EGAIN'], q=[0., train_size, 1.], labels=['train', 'test'])

            X_train = self.X[egain_categories == 'train']
            y_train = self.y[egain_categories == 'train']
            X_test = self.X[egain_categories == 'test']
            y_test = self.y[egain_categories == 'test']

            df_train = self.df[egain_categories == 'train']
            df_test = self.df[egain_categories == 'test']


        else:
            raise RuntimeError(f"Unsupported split argument '{split}'")

        if scale:
            # Make new scalers if needed
            fit = self.initialize_scalers()

            # Make sure to fit the data to the training set and not the test set
            X_train = self.normalize_data(X_train, self.scaler_x, fit=fit)
            y_train = self.normalize_data(y_train, self.scaler_y, fit=fit)
            X_test = self.normalize_data(X_test, self.scaler_x, fit=False)
            y_test = self.normalize_data(y_test, self.scaler_y, fit=False)

        train_loader = DataLoader(NDX_RF_Dataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(NDX_RF_Dataset(X_test, y_test), batch_size=batch_size, shuffle=shuffle)

        if provide_df:
            return train_loader, test_loader, df_train, df_test
        else:
            return train_loader, test_loader


class OperationsData(FEData):
    """Class for managing data collected during CEBAF operations.

    This class can be used to load up data from trips or standard beam operations.
    """

    def __init__(self, filename: str, ced_file: str, onset_file: Optional[str] = None, section: str = 'train',
                 data_dir: str = "data", rad_zones: Optional[List[str]] = None,
                 gmes_zones: Optional[List[str]] = None, rad_suffix: str = '', linac='1L',
                 max_current: float = 0.01) -> None:
        # Load up the data in the subclass
        super().__init__(filename=filename, ced_file=ced_file, onset_file=onset_file, section=section,
                         data_dir=data_dir, rad_zones=rad_zones, gmes_zones=gmes_zones, rad_suffix=rad_suffix,
                         meta_cols=['Datetime', 'level_0', 'IBC0R08CRCUR1'], linac=linac)

        # Track the maximum allowable beam current and filter it out when loading the data
        self.max_current = max_current

    def load_data(self):
        """Load up the data and filter out examples where the beam current is too high."""
        super().load_data()
        allowable_current = self.df['IBC0R08CRCUR1'] < self.max_current
        self.df = self.df[allowable_current]
        self.X = self.X[allowable_current]
        self.y = self.y[allowable_current]


def split_to_train_test_files(gradient_scan_file: str, train_size: float = 0.8, group_col='settle_start',
                              data_dir: str = 'data') -> None:
    """Method for splitting gradient scan file into train and test sets that are stored separately on the file system.

    Only makes the new files if they don't exist.  Nothing done if they both exist.
    """

    master_file = f'{data_dir}/all/{gradient_scan_file}'
    test_file = f'{data_dir}/test/{gradient_scan_file}'
    train_file = f'{data_dir}/train/{gradient_scan_file}'

    if os.path.exists(test_file) and os.path.exists(train_file):
        return

    df = pd.read_csv(master_file, comment='#')

    gss = GroupShuffleSplit(train_size=train_size, random_state=732, n_splits=2)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))

    df_train = df.iloc[train_idx, :]
    df_test = df.iloc[test_idx, :]

    if not os.path.exists(test_file):
        df_test.to_csv(test_file, index=False)
    if not os.path.exists(train_file):
        df_train.to_csv(train_file, index=False)


def get_gmes_columns(linac: str, zones: Optional[List[str]]) -> List[str]:
    """This gets the measured gradient column names for a linac and set of zones.

    Args:
        linac: The name of the linac to construct GMES column names for.  Only '1L' or '2L' supported.
        zones: A list of zone names to be explicitly included.  E.g., ['2', 'A', 'M'].
    """
    supported = ('1L', '2L')
    if linac not in supported:
        raise ValueError(f"Unsupported linac name '{linac}'.  Supported = {supported}.")

    if linac == '1L':
        # The NL still does not have a 1L26 CM
        lin = '1'
    else:
        lin = '2'

    c100_gmes_cols = []
    if zones is None:
        if lin == '1':
            zones = [x for x in '23456789ABCDEFGHIJKLMNOP']
        if lin == '2':
            zones = [x for x in '23456789ABCDEFGHIJKLMNOPQ']

    for zone in zones:
        for cav in range(1, 9):
            c100_gmes_cols.append(f"R{lin}{zone}{cav}GMES")

    return c100_gmes_cols


def get_ndx_columns(linac: str, zones: Optional[List[str]] = None, suffix: str = "") -> List[str]:
    """Generate NDX column names in a standard way.  Optionally include a lag suffix if needed.

    Args:
        linac:  Which linac to generate column names for.  1L or 2L.
        zones:  A list of zones names (e.g., 22 or 05) to include in the output
        suffix:  The string value to append to the end of the radiation column name.  Useful for lagged variables.
    """
    supported = ('1L', '2L')
    if linac not in supported:
        raise ValueError(f"Unsupported linac name '{linac}'.  Supported = {supported}.")

    c100_ndx_cols = []

    if zones is None:
        if linac == '1L':
            zones = ('05', '06', '07', '08', '22', '23', '24', '25', '26', '27')
        elif linac == '2L':
            zones = ('22', '23', '24', '25', '26', '27')

    # Group the outputs by radiation type, not location, so we can do a two branch model
    for zone in zones:
        c100_ndx_cols.append(f"INX{linac}{zone}_nDsRt{suffix}")
    for zone in zones:
        c100_ndx_cols.append(f"INX{linac}{zone}_gDsRt{suffix}")

    return c100_ndx_cols


class NDX_RF_Dataset(Dataset):
    """Custom data set for presenting GMES and NDX radiation."""

    def __init__(self, gmes: np.ndarray, rad: np.ndarray) -> None:
        if len(gmes) != len(rad):
            raise RuntimeError("Length of X and y do not match.")

        self.X = torch.from_numpy(gmes.astype("float32"))
        self.y = torch.from_numpy(rad.astype("float32"))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gmes = self.X[index, :]
        rads = self.y[index, :]

        return gmes, rads

    def __len__(self) -> int:
        return len(self.X)


def report_performance(y_pred: pd.DataFrame, y_true: pd.DataFrame, egain: pd.Series, dtime: pd.Series, set_name: str):
    """Reports the performance of a set of predictions versus the ground truth.  gd used for"""
    r2, mse, mae = utils.score_model(y_pred=y_pred, y_test=y_true)
    utils.print_model_scores(r2=r2, mse=mse, mae=mae, set_name=set_name)

    print()
    r2, mse, mae = utils.score_model(y_pred=y_pred, y_test=y_true, multioutput='uniform_average')
    print(set_name)
    utils.print_model_scores(r2=r2, mse=mse, mae=mae, set_name="")

    g_df, n_df, g_diag_df, n_diag_df = utils.get_sensor_plot_data(y_true=y_true, y_pred=y_pred, egain=egain,
                                                                  dtime=dtime)
    utils.plot_data(g_df=g_df, n_df=n_df, g_diag_df=g_diag_df, n_diag_df=n_diag_df, title=set_name)


def eval_model_on_nov5_data(model, gmes_zones, rad_zones):
    data_file = "processed_gradient-scan-2021-11-05_113837.646876.txt.csv"
    ced_file = 'ced-data/ced-2021-11-06.tsv'
    gd = GradientScanData(filename=data_file, onset_file=None, section='all', ced_file=ced_file,
                          gmes_zones=gmes_zones, rad_zones=rad_zones, linac='1L')
    gd.load_data()
    data_loader = gd.get_data_loader(shuffle=False)

    print("All Nov 5 Set")
    device = torch.device("cpu") if not torch.has_cuda else torch.device("cuda:0")
    y_pred = utils.make_predictions(model=model, data_loader=data_loader, device=device, y_cols=gd.y_cols)
    report_performance(y_pred=y_pred, y_true=gd.y, egain=gd.df.EGAIN, dtime=gd.df.Datetime,
                       set_name="Test on Nov 5 Data")


def evaluate_model_on_trip_data(model: nn.Module, gmes_zones: List[str], rad_zones: List[str], data_file: str,
                                data_filter: Optional[str] = None, set_name: str = "Trip Data") -> None:
    ced_file = 'ced-data/ced-2021-11-06.tsv'
    gd = OperationsData(filename=data_file, onset_file=None, section='all', ced_file=ced_file,
                        gmes_zones=gmes_zones, rad_zones=rad_zones, linac='1L')
    gd.load_data()
    gd.filter_data(data_filter)
    data_loader = gd.get_data_loader(shuffle=False)

    print(f"data_file: {data_file}")
    print(f"query: {data_filter}")

    device = torch.device("cpu") if not torch.has_cuda else torch.device("cuda:0")
    y_pred = utils.make_predictions(model=model, data_loader=data_loader, device=device, y_cols=gd.y_cols)
    report_performance(y_pred=y_pred, y_true=gd.y, egain=gd.df.EGAIN, dtime=gd.df.Datetime,
                       set_name=set_name)


def filter_trip_data_to_nl(input_file: str, output_file: str, min_gradient_range: float = 0.2,
                           add_fault_ids: bool = True):
    """This filters out data that does not include a trip of any north linac cavity and saves a new data file."""

    if input_file == output_file:
        raise RuntimeError("input_file == output_file")
    df = pd.read_csv(input_file)
    gmes_cols = df.columns[df.columns.str.contains("GMES")].to_list()

    gmes_range = df[['level_0'] + gmes_cols].groupby(['level_0']).apply(lambda x: x.max() - x.min())
    has_fault = (gmes_range > min_gradient_range).any(axis=1)

    df = df.set_index(df.level_0)

    if add_fault_ids:
        faulted_cavities = gmes_range.apply(
            lambda x: gmes_range.columns[x > min_gradient_range].str[0:4].unique().to_list(),
            axis=1)
        faulted_zones = gmes_range.apply(
            lambda x: gmes_range.columns[x > min_gradient_range].str[0:3].unique().to_list(),
            axis=1)
        df['faulted_cavities'] = faulted_cavities.apply(lambda x: ":".join(x))
        df['faulted_zones'] = faulted_zones.apply(lambda x: ":".join(x))

    df = df.loc[has_fault, :]
    df.to_csv(output_file, index=False)
