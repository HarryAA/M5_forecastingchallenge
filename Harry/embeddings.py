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
n_days = 365            # Number of days to analyse

filename = '../data/'   # Data directory

state = 'CA'            # Data subdivision to analyse, leave blank study entire dataset

if state:
    filename += '%s/%s_data.csv' % (state, state)
else:
    filename += 'sales_train_validation.csv'

df = pd.read_csv(filename, index_col=0)                 # Read data
calendar_df = pd.read_csv('../data/calendar.csv')       # Read the calendar csv for notable dates

days_df = calendar_df.loc[:n_days, 'weekday'].astype(str).values.tolist()
snap_CA = np.array(calendar_df.loc[:n_days, 'snap_CA'])
events_CA = calendar_df.loc[:n_days, 'event_name_1']
events_type_CA = calendar_df.loc[:n_days, 'event_type_1']
events_CA = events_CA.fillna('')
events_type_CA = events_type_CA.fillna('')