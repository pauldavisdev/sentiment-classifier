from sklearn.utils import shuffle
import pandas as pd

def read_csv(num_csv_rows):
    """ 
    Reads CSV, shuffles, and returns training and test data.

    :param num_csv_rows: Number of rows to be returned
    :return train: Specified number of rows of training data
    :return test: Test data
    """    
    cols = [0, 5]

    col_names = ['polarity', 'tweet']

    print('Reading data:')

    train = shuffle(pd.read_csv('data/stanfordtraindata.csv', header=None, usecols=cols, names=col_names))

    test = shuffle(pd.read_csv('data/stanfordtestdata.csv', header=None, usecols=cols, names=col_names))

    return train[:num_csv_rows], test