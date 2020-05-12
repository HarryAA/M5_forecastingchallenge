import os
import pandas as pd

data_divisions = {
    'state_id': ['CA', 'TX', 'WI'],
    'dept_id': ['HOBBIES', 'FOODS', 'HOUSEHOLD']
}

df = pd.read_csv('../data/sales_train_validation.csv')

for key in data_divisions:
    print(key)
    for item in data_divisions[key]:
        print(item)
        path = '../data/%s/' % str(item)
        df_split = df.loc[df[str(key)] == str(item)]
        if not os.path.isdir(path):
            os.mkdir(path)
        df_split.to_csv(path + ('%s_data.csv' % str(item)))