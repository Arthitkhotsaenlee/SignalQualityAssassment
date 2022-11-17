import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MA():
    '''
    Sample code:
    from utils.signal_loading import MA
    file_path_ma = 's7_math_1_16092022_115848.ma'
    file_path_anno = 's7_arithmetics_exp_2022_Sep_16_1159_test.csv'
    ma = MA.load_file(file_path_ma)
    raw = ma.to_raw(resample_freq=500)
    raw = ma.set_anno(file_path_anno=file_path_anno, anno_col='difficulty_level')
    ma.to_fif(folder_path, 'fif_file_name', file_type='eeg', overwrite=True)
    '''
    unit_scale = {
        'no': 1,
        'm': 1e-3,
        'u': 1e-6,
        'n': 1e-9,
    }

    def __init__(self, signal_start_timestamp=0, df_ma=None, voltage_scale='u', sfreq=500.5,
                 ch_types=['eeg', 'eeg'],
                 ch_names=['ch1', 'ch2']):
        """
        Params
        --------------------
        df_ma: pandas.DataFrame
            The dataframe contains the package number in the first column
            while the remaining ones contain the data payload
            from the two channels alternatively
        voltage_scale: string
        sfreq: float
            sampling frequency in Hz
        ch_types: a list of string
            the types of signal of the channels
        ch_names: a list of string
            the channel names
        Returns
        --------------------
        """
        self.signal_start_timestamp = signal_start_timestamp
        self.df_ma = df_ma
        self.voltage_scale = voltage_scale
        self.sfreq = sfreq
        self.ch_types = ch_types
        self.ch_names = ch_names

        if self.df_ma is not None:
            # get the signal data
            self.signal_raw = self.df_ma.copy(deep=True)

            # separate the package number

            # +----------------+-----------+-----------+-----------+-----------+-----+
            # | package number | channel 1 | channel 2 | channel 1 | channel 2 | ... |
            # +----------------+-----------+-----------+-----------+-----------+-----+
            self.signal_raw = self.signal_raw.drop(axis=1, columns=0).values.reshape(-1, 2) * self.unit_scale[self.voltage_scale]
            self.packages = self.df_ma.copy(deep=True)[0].values

            self.check_package_sequence()

    def check_package_sequence(self):
        # Check package sequence of package

        package_diff = np.diff(self.packages)
        if max(package_diff) > 1:
            print('Missing package detected')

    def _check_raw_attri(self):

        if not hasattr(self, 'raw'):
            raise Exception('The MA class does not have the "raw" attribute yet.')

    @classmethod
    def load_file(cls, file_path_ma, voltage_scale='u', sfreq=500.5):

        with open(file_path_ma, "r") as f:
            dataline = f.readline()

        signal_start_timestamp = int(dataline)
        df_ma = pd.read_csv(file_path_ma, sep=',', header=None, skiprows=1)

        return cls(signal_start_timestamp, df_ma, voltage_scale, sfreq)

    def load_fif(self, file_path_fif, resample_freq=None):
        self.raw = mne.io.read_raw_fif(file_path_fif)

        if resample_freq is not None:
            self.raw = self.raw.resample(resample_freq)

    def to_raw(self, resample_freq=None,):

        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types=self.ch_types)
        self.raw = mne.io.RawArray(self.signal_raw.T, info, verbose=False)

        if resample_freq is not None:
            self.raw = self.raw.resample(sfreq=resample_freq).copy()

        return self.raw

    def set_anno(self, file_path_anno=None, anno_col=None):

        self._check_raw_attri()

        if file_path_anno is not None and anno_col is None:
            raise Exception('The "anno_col" is not defined.')

        elif file_path_anno is None and anno_col is not None:
            raise Exception('The "file_path_anno" is not defined.')

        elif file_path_anno is not None and anno_col is not None:
            self.df_anno = pd.read_csv(file_path_anno)
            self.df_anno = self.df_anno.dropna(subset=["stim_start"])
            self.exp_start_timestamp = self.df_anno["start_time_stamp"].copy().dropna().values[0]

            # find the starting time difference between
            # the eeg device operation and the behavioral experiment
            # millisecond to second
            time_diff = (self.exp_start_timestamp - self.signal_start_timestamp) / 1000

            start = self.df_anno["stim_start"].values
            start = start + time_diff
            duration = self.df_anno["stim_end"].values - self.df_anno["stim_start"].values
            desc = self.df_anno[anno_col].copy().values

            annotations = mne.Annotations(onset=start, duration=duration, description=desc)
            self.raw.set_annotations(annotations)

        return self.raw

    def to_fif(self, save_path, file_name, file_type='eeg', compress=False, overwrite=False):

        self._check_raw_attri()

        suffix = {
            'eeg': '_eeg.fif',
        }

        if file_type in ['eeg']:
            file_name = file_name + suffix[file_type]
        else:
            raise Exception('File type not supported')

        if compress:
            file_name = file_name + '.gz'

        save_path = os.path.join(save_path, file_name)

        self.raw.save(fname=save_path, overwrite=overwrite)



datafolder_path = "Z:\sample_data\quality_assessment"
folder_list = os.listdir(datafolder_path)
for i in folder_list:
    file_list = os.listdir(os.path.join(datafolder_path,i))
    for j in file_list:
        file_path = os.path.join(datafolder_path,i,j)
        print(file_path)
        if file_path.endswith(".ma"):
            ma = MA.load_file(file_path)
            raw = ma.to_raw()
            raw.notch_filter(np.arange(50,round(raw.info["sfreq"]/2),50)).filter(1,49)
            raw.plot(block=True)
            del raw
