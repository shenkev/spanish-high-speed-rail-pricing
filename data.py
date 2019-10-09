import numpy as np 
import pandas as pd 
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

tqdm.pandas()

SAMPLE_RATE = 0.01
SAMPLES_TRAIN = 80000
SAMPELS_VAL = 20000

datetimeFormat = '%Y-%m-%d %H:%M:%S'

def delta_min(a, b):
    diff = datetime.datetime.strptime(b, datetimeFormat) - datetime.datetime.strptime(a, datetimeFormat)
    return (diff.seconds/60.0)

def start_min(a, m):
    time = datetime.datetime.strptime(a, datetimeFormat) - datetime.datetime.strptime(m, datetimeFormat)
    return (time.seconds/60.0)

def main():
    data = pd.read_csv('./renfe.csv', index_col=0)
    data.shape

    # data = data.sample(frac=SAMPLE_RATE, replace=False, random_state=1)
    data = data[pd.notnull(data['price'])]  # seems to get rid of missing values in other fields as well
    train_data, rest_data = data.iloc[:SAMPLES_TRAIN], data.iloc[SAMPLES_TRAIN:]
    # need to make sure train_class and train_type are same with val data
    for condition in ['train_class', 'train_type', 'origin', 'destination', 'fare']:
        rest_data = rest_data[rest_data[condition].isin(train_data[condition].unique())]

    data = pd.concat([train_data, rest_data.iloc[-SAMPELS_VAL:]])
    data.shape
    # get one-hot features
    [origin, destination, train_type, train_class, fare] = [pd.get_dummies(data[x]) for x in ['origin', 'destination', 'train_type', 'train_class', 'fare']]

    min_date = data['start_date'].min()

    data['travel_time_in_min'] = data.progress_apply(lambda x:delta_min(x['start_date'], x['end_date']), axis=1) 
    # data['start_time_in_min'] = data.progress_apply(lambda x:delta_min(x['start_date'], min_date), axis=1) 

    data.drop(['start_date','end_date'], axis=1,inplace=True)
    # new_data = data[['price', 'travel_time_in_min','start_time_in_min']]
    new_data = data[['price', 'travel_time_in_min']]

    for df in [origin, destination, train_type, train_class, fare]:
        new_data = pd.concat([new_data, df], axis=1)
    
    # normalize the label a bit
    new_data['price'] = 10*(new_data['price'])/ (new_data['price'].max() - new_data['price'].min())
    
    print(new_data.head())

    # save as numpy
    np_arr = new_data.values
    # np.random.shuffle(np_arr)

    # VAL_PERCENT = 0.20
    # VAL_INDEX = int(np_arr.shape[0]*(1.0-VAL_PERCENT))
    VAL_INDEX = SAMPLES_TRAIN

    val_X, val_y, train_X, train_y = np_arr[VAL_INDEX:, 1:], np_arr[VAL_INDEX:, 0], np_arr[:VAL_INDEX, 1:], np_arr[:VAL_INDEX, 0]

    np.savez_compressed('./data/ready_data_{}.npz'.format(SAMPLE_RATE), train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y)


main()