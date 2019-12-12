import pandas as pd
import os
PATH = '/data/hawfinch/clarkson/data/us_eq'

try:
    os.mkdir('{}/cache'.format(PATH))
except FileExistsError:
    pass
returns = pd.DataFrame()

for folder in os.listdir(PATH):
    for file in os.listdir('{}/{}'.format(PATH, folder)):
        if file[-3:] == 'csv':
            df = pd.read_csv('{}/{}/{}'.format(PATH, folder, file))
            df['date'] = pd.to_datetime(file[:-4], format='%Y%m%d')
            returns = returns.append(df)

returns.set_index(['date', 'ticker'], inplace=True)
returns.sort_index(inplace=True)
returns.to_pickle('{}/cached_multi.pkl'.format(PATH))
