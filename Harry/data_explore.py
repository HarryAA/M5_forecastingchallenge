import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('sales_train_validation.csv')

print(df.columns)
print(df.head())
print(df.shape)

entry = df.iloc[:, 6:106]
entry_id = df.iloc[:, 1]
cat_id = df.iloc[:, 3]
print(cat_id)
plt.figure(1)
plt.plot(entry)
entry = np.array(entry)
entry_id = np.array(entry_id)
cat_id = np.array(cat_id)


print(entry)
entry = np.reshape(entry, (entry.shape[1], entry.shape[0]))
#entry_id = np.reshape(entry_id, (entry_id[1], entry_id.shape[0]))
print(entry.shape)

item_type = 'init_string'
dept_id = 'init_string'
store_id = 'init_string'

for i in range(df.shape[0]):
    if store_id not in df['store_id'][i]:
        print(df['store_id'][i])
        store_id = df['store_id'][i]
    if item_type not in cat_id[i]:
        print(cat_id[i])
        item_type = cat_id[i]
    if dept_id not in df['dept_id'][i]:
        print(df['dept_id'][i])
        dept_id = df['dept_id'][i]




cuml_entry = np.zeros(entry.shape)

plt.figure(2)
for i in range(cuml_entry.shape[0]):
    cuml_entry[i] = cuml_entry[i-1] + entry[i]

hobbies_CA = np.zeros((entry.shape[0], 2))
hobbies_TX = np.zeros((entry.shape[0], 2))
hobbies_WI = np.zeros((entry.shape[0], 2))
print(df.shape)
print(entry.shape)
for i in range(1, 100):
    print(i)
    for j in range(df.shape[0]):
        if store_id not in df['store_id'][j]:
            store_id = df['store_id'][j]
        if item_type not in cat_id[j]:
            item_type = cat_id[j]
        if dept_id not in df['dept_id'][j]:
            dept_id = df['dept_id'][j]

        if item_type == 'HOBBIES':
            if 'CA' in store_id:
                hobbies_CA[i][0] += entry[i][j]
            if 'TX' in store_id:
                hobbies_TX[i][0] += entry[i][j]
            if 'WI' in store_id:
                hobbies_WI[i][0] += entry[i][j]

    hobbies_CA[i][1] = hobbies_CA[i - 1][1] + hobbies_CA[i][0]
    hobbies_TX[i][1] = hobbies_TX[i - 1][1] + hobbies_TX[i][0]
    hobbies_WI[i][1] = hobbies_WI[i - 1][1] + hobbies_WI[i][0]

plt.plot(cuml_entry)

plt.figure(3)
plt.plot(hobbies_CA[:,0])
plt.plot(hobbies_TX[:,0])
plt.plot(hobbies_WI[:,0])

plt.figure(4)
plt.plot(hobbies_CA[:,1])
plt.plot(hobbies_TX[:,1])
plt.plot(hobbies_WI[:,1])

plt.show()