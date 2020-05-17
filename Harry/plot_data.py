import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Dictionary of the different data categories in the dataset
data_divisions = {
    'state_id': ['CA', 'TX', 'WI'],
    'store_id': [['CA_1', 'CA_2', 'CA_3', 'CA_4'], ['TX_1', 'TX_2', 'TX_3'], ['WI_1', 'WI_2', 'WI_3']],
    'cat_id': ['HOBBIES', 'HOUSEHOLD', 'FOODS'],
    'dept_id': [['HOBBIES_1', 'HOBBIES_2'], ['HOUSEHOLD_1', 'HOUSEHOLD_2'], ['FOODS_1', 'FOODS_2', 'FOODS_3']]
}

d1_col = 6              # First day column
n_days = 100            # Number of days to analyse

filename = '../data/'   # Data directory

state = 'CA'            # Data subdivision to analyse, leave blank study entire dataset

if state:
    filename += '%s/%s_data.csv' % (state, state)
else:
    filename += 'sales_train_validation.csv'

df = pd.read_csv(filename, index_col=0)                 # Read data
calendar_df = pd.read_csv('../data/calendar.csv')       # Read the calendar csv for notable dates

snap_CA = np.array(calendar_df.loc[:n_days, 'snap_CA'])

for dept in data_divisions['dept_id']:
    for item in dept:
        df_split = df.loc[df['dept_id'] == str(item)].iloc[:, d1_col:d1_col+n_days]
        time_series = np.transpose(np.array(df_split))
        cuml_time_series = np.zeros((time_series.shape[0], time_series.shape[1]))
        for i in range(time_series.shape[0]):
            cuml_time_series[i] = cuml_time_series[i-1] + time_series[i]
        print(time_series)

        fig, ax = plt.subplots(1,1)
        plt.title('Day by day sales for %s items in state: %s' % (item, state))
        plt.plot(time_series)
        for i in range(snap_CA.shape[0]):
            if snap_CA[i]:
                ax.axvspan(i, i+1, alpha=0.2, color='gray')

        plt.figure(2)
        plt.plot(cuml_time_series)

        plt.show()