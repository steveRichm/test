import pandas as pd;
from tensorflow.python.keras.layers import Bidirectional

print (pd.__version__)
import numpy as np; print (np.__version__)
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense,Dropout
import os
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('clean.csv')
df.columns = ['A','B','C','D','E','F']
df = df.drop(['B','C','D','E','F'], axis = 1)
df = df.loc[::-1].reset_index(drop = True)
df.head()

# Normalize
scalar = StandardScaler().fit(df.values)
transformed_dataset = scalar.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)


number_of_rows= df.values.shape[0] #all our games
window_length = 17 #amount of past games we need to take in consideration for prediction
number_of_features = df.values.shape[1] #balls count

train = np.empty([number_of_rows-window_length, window_length, number_of_features], dtype=float)
label = np.empty([number_of_rows-window_length, number_of_features], dtype=float)

for i in range(0, number_of_rows-window_length):
    train[i]=transformed_df.iloc[i:i+window_length, 0: number_of_features]
    label[i]=transformed_df.iloc[i+window_length: i+window_length+1, 0: number_of_features]

# for i in range(0,number_of_rows-window_length):
#     print(f'T: {train[i]}')
#
# for i in range(0,number_of_rows-window_length):
#     print(f'L: {label[i]}')
run = 4000
#Ltsm = [124,64,37]
batch_size = 100

file_name = f'res_row_{number_of_rows}_lines_{window_length}_balls_{number_of_features}_e_{run}_batch_{batch_size}1.sav'


# Model and layers
if os.path.exists(os.path.join(os.getcwd(),file_name)):
    model = load_model(file_name)
else:
    model = Sequential()
    model.add(Bidirectional(LSTM(37,
               input_shape=(window_length, number_of_features),
               return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(37,
               input_shape=(window_length, number_of_features),
               return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(37,
               input_shape=(window_length, number_of_features),
               return_sequences=True)))
    model.add(Dense(number_of_features))
    model.add(Bidirectional(LSTM(37,
               return_sequences=False)))
    model.add(Dense(number_of_features))
    #model.compile(loss='mse', optimizer='rmsprop',metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
    model.fit(train, label,
          batch_size=batch_size, epochs=run)
    model.save(file_name)

from1 = window_length * -1
to_predict=df.iloc[from1:]
print(f'\nto Predict\n{to_predict}\n')
scaled_to_predict = scalar.transform(to_predict)
print(f'\nscaled\n{scaled_to_predict}\n')
scaled_predicted_output_1 = model.predict(np.array([scaled_to_predict]))
for i in scaled_predicted_output_1:
    print(f'\n{i}\n')
data = scalar.inverse_transform(scaled_predicted_output_1).astype(int)
print(data)
dfoutput = pd.DataFrame(data, columns=df.columns)
dfoutput.to_csv(f'predict.csv', index=False, mode='a')
dfoutput
print(file_name)



'''
model = Sequential()
model.add(Bidirectional(LSTM(64,      
           input_shape=(window_length, number_of_features),
           return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32,           
           return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(number_of_features))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.fit(train, label,
          batch_size=64, epochs=300)
'''