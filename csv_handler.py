from sklearn.utils import shuffle
import pandas as pd
import time, os
from shutil import copyfile

def read_csv(num_csv_rows):
    """Reads CSV, shuffles, and returns training and test data.

    Args:
        num_csv_rows: Number of training data rows to be returned.
    Returns:
        Training data (only num_csv_rows rows) and all test data.

    """    
    cols = [0, 5]

    col_names = ['polarity', 'tweet']

    print('Reading data:')

    train = shuffle(pd.read_csv('data/stanfordtraindata.csv', header=None, usecols=cols, names=col_names, encoding='utf-8'))

    test = shuffle(pd.read_csv('data/stanfordtestdata.csv', header=None, usecols=cols, names=col_names))

    return train[:num_csv_rows], test

def write_csv(df):
    """Adds timestamp to dataframe, then writes to CSV file.
    Args:
        df: Dataframe to write to CSV file.
        
    """
    timestamp = int(time.time())

    df['timestamp'] = timestamp

    print(df)

    log_path = 'logs/log.csv'

    if os.path.exists(log_path):
        copyfile(log_path, ''.join(['logs/log', '_', str(timestamp), '.csv']))
        # append to csv
        with open(log_path, 'a', newline='') as f:
            df.to_csv(f, header=True, index=False)
    else:
        # create new csv
        df.to_csv(path_or_buf=log_path, index=False)
