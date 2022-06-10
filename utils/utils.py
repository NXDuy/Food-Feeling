# from pickletools import uint1
import torch
import pandas as pd
import re
from torch.utils.data import Dataset

def read_file(file_dir, columns=None):
    try:
        raw_data = pd.read_excel(file_dir)
    except:
        return None
    # print(raw_data.columns)
    using_data = raw_data[columns]
    using_data = using_data.dropna(axis=0)
    
    index_data = list()
    for str_time in using_data['start time']:
        if str_time[0].isdigit() and str_time[-1].isdigit():
            index_data.append(True)
        else:
            index_data.append(False)
    
    using_data = using_data[index_data]

    index_data = list()
    for str_time in using_data['end time']:
        if str_time[0].isdigit() and str_time[-1].isdigit():
            index_data.append(True)
        else:
            index_data.append(False)

    using_data = using_data[index_data]
    df_time = list()
    for start_time, end_time in zip(using_data['start time'], using_data['end time']):
        df_time.append((process_time_str(start_time), process_time_str(end_time)))
    
    using_data[['start time', 'end time']] = df_time
    using_data["viewer feeling of youtuber's style "] = using_data["viewer feeling of youtuber's style "].astype('int') 
    
    # Normalization
    # using_data['start time'] = (using_data['start time'] - using_data['start time'].mean())/using_data['start time'].var()
    # using_data['end time'] = (using_data['end time'] - using_data['end time'].mean())/using_data['end time'].var()
    # using_data['Unnamed: 11'] = (using_data['Unnamed: 11'] - using_data['Unnamed: 11'].mean())/using_data['Unnamed: 11'].var()
    
    # using_data = using_data.dropna(axis=0)

    input_data = using_data[['start time', 'end time', 'Unnamed: 11']]
    output_data = using_data["viewer feeling of youtuber's style "]
    # print(using_data.head())
    return torch.tensor(input_data.values, dtype=torch.float32), torch.tensor(output_data.values, dtype=torch.float32)
        
def __check_value_time(hours=0, minutes=0, seconds=0):
    # print('Time out: 'hours, minutes, seconds)
    if hours > 24:
        print('Time error: ',hours, minutes, seconds)
        raise ValueError('Hour is out of range')

    if seconds > 60:
        print('Time error: ',hours, minutes, seconds)
        raise ValueError('Second is out of range')

    if minutes > 60:
        print('Time error: ',hours, minutes, seconds)
        raise ValueError('Minutes is out of range')

def process_time_str(time_str):
    time_data = re.findall('[0-9]?[0-9]', time_str)
    TIME_WITH_HOUR = 3
    TIME_WITH_MINUTES = 2
    TIME_WITH_SECONDS = 1
    MAX_SECONDS = 90060
    # print(time_data)
    if len(time_data) == TIME_WITH_HOUR:
        hours = int(time_data[0])
        minutes = int(time_data[1])
        seconds = int(time_data[2])

        __check_value_time(hours, seconds, minutes)

        return (hours*3600 + minutes*60 + seconds)
    if len(time_data) == TIME_WITH_MINUTES:
        minutes = int(time_data[0])
        seconds = int(time_data[1])

        __check_value_time(minutes=minutes, seconds=seconds)

        return (minutes*60 + seconds)
    if len(time_data) == TIME_WITH_SECONDS:
        seconds = int(time_data[0])

        __check_value_time(seconds=seconds)

        return seconds
    

FILE_DIR = './Data/Data_AIL.xlsx'
COLUMNS = ['start time', 'end time', 'Unnamed: 11',
            "viewer feeling of youtuber's style "]

class FoodFeeling(Dataset):
    def __init__(self, file_dir=FILE_DIR, columns=COLUMNS):
        super(FoodFeeling, self).__init__()
        self.input_data, self.output_data = read_file(file_dir, columns)
        self.n_samples = self.output_data.shape[0]

    def __getitem__(self, index):
        return self.input_data[index], self.output_data[index]

    def __len__(self):
        return self.n_samples

# data = FoodFeeling()