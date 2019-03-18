from csv_handler import read_csv
import numpy as np

def main():
    num_csv_rows = 10000

    train, test = read_csv(num_csv_rows)

    testX, testY = test['tweet'].values, test['polarity'].values
    trainX, trainY = train['tweet'].values, train['polarity'].values

    trainY = np.where(trainY == 4, 1, trainY)
    testY = np.where(testY == 4, 1, testY)

if __name__ == '__main__':
    main()