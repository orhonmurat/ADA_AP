# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:29:18 2024

@author: murat
"""
#import related data
import pandas as pd
df_air_data = pd.read_csv('C:/Users/murat/OneDrive/Masaüstü/unil master/semester 2/advanced data analysis/project/data/order122373/air_data.csv', delimiter=';')
df_air_data.columns = ['Canton', 'Time', 'Cloud_cover','Temperature', 'Humidity', 'Sunshine'] 

df_air_data = df_air_data[df_air_data['Time'].str.match(r'\d{8}')]
df_air_data['Time'] = pd.to_datetime(df_air_data['Time'], format='%Y%m%d')

# filter by cantons geneve and vaud
geneva_air_data = df_air_data[df_air_data["Canton"]=="GVE"]
geneva_air_data["Time"]= pd.to_datetime(geneva_air_data["Time"])

vaud_air_data = df_air_data[df_air_data["Canton"]=="VEV"]
vaud_air_data["Time"]= pd.to_datetime(vaud_air_data["Time"])

combined_air_data =  pd.merge(vaud_air_data, geneva_air_data,on='Time', how='left')
combined_air_data[["Humidity_x","Cloud_cover_x","Temperature_x","Humidity_y","Cloud_cover_y","Temperature_y","Sunshine_y"]] = combined_air_data[["Humidity_x","Cloud_cover_x","Temperature_x","Humidity_y","Cloud_cover_y","Temperature_y","Sunshine_y"]].astype(float)

# find the average values
combined_input_data = pd.DataFrame(combined_air_data["Time"])
combined_input_data["Humidity"] = (combined_air_data["Humidity_x"]+combined_air_data["Humidity_y"])/2
combined_input_data["Cloud_cover"] = (combined_air_data["Cloud_cover_x"]+combined_air_data["Cloud_cover_y"])/2
combined_input_data["Temperature"] = (combined_air_data["Temperature_x"]+combined_air_data["Temperature_y"])/2
combined_input_data["Sunshine"] = combined_air_data["Sunshine_y"]

combined_input_data.set_index('Time', inplace=True)

# import electricity consumption dataset
df_daily_consumption = pd.read_csv('C:/Users/murat/OneDrive/Masaüstü/unil master/semester 2/advanced data analysis/project/data/daily_consumption.csv')
df_daily_consumption.columns = ['Time', 'Demand'] 
df_daily_consumption['Time'] = pd.to_datetime(df_daily_consumption['Time'])
df_daily_consumption.set_index('Time', inplace=True)



combined_input_data = pd.merge(combined_input_data, df_daily_consumption, left_index=True, right_index=True)

#import population data set
df_population = pd.read_excel('C:/Users/murat/OneDrive/Masaüstü/unil master/semester 2/advanced data analysis/project/data/population_of_cantons_2.xlsx')



################################### find population of geneva
population_geneve = df_population[df_population["Canton"]=="Genève"]

population_geneve['Year'] = pd.to_datetime(population_geneve['Year'], format='%Y')


################################## find population of vaud
population_vaud = df_population[df_population["Canton"]=="Vaud"  ]
population_vaud['Year'] = pd.to_datetime(population_vaud['Year'], format='%Y')



# calculate the total population
total_population=pd.DataFrame(population_vaud["Year"])
total_population["Total_Population"] = 0
for m in range(len(population_vaud)):
    total_population["Total_Population"].iloc[m] = (population_vaud['Population on 1 January'].iloc[m]+population_geneve['Population on 1 January'].iloc[m])








# create input dataset for lstm and gru
combined_input_data["Population"] = 0
for m in range(len(combined_input_data)) :
    for t in range(len(total_population)):
        if combined_input_data.index.year[m] == total_population["Year"].iloc[t].year:
            combined_input_data["Population"].iloc[m]  = total_population["Total_Population"].iloc[t]


# population data is available until 2023
combined_input_data = combined_input_data[combined_input_data.index.year < 2023]



########################## LSTM
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
# Normalize the data
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))



# Split data into inputs and outputs
X = scaler_x.fit_transform(combined_input_data.iloc[:,[0,1,2,3,5]])  # 
y = scaler_y.fit_transform(combined_input_data.iloc[:,4].values.reshape(-1, 1))  # 

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

predicted_values = model.predict(X_test)
predicted_values = scaler_y.inverse_transform(predicted_values)  # Inverse scaling to original scale

# Inverse scaling the test labels for actual comparison
actual_y_test = scaler_y.inverse_transform(y_test)

# RMSE Calculation
mse = mean_squared_error(actual_y_test, predicted_values)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)


#import matplotlib.pyplot as plt
#plt.figure(figsize=(10, 6))
#plt.plot(actual_y_test, label='Actual Values', marker='.')
#plt.plot(predicted_values, label='Predicted Values', linestyle='--', marker='x')
#plt.title('LSTM Prediction vs Actual Data')
#plt.xlabel('Sample Index')
#plt.ylabel('Target Variable')
#plt.legend()
#plt.show()


############################GRU

from tensorflow.keras.layers import GRU, Dense

# Assuming X_train.shape[1] is the sequence length and X_train.shape[2] is the number of features
model_gru = Sequential([
    GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    GRU(50),
    Dense(1)
])

# Compile the model
model_gru.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model_gru.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)


test_predictions_gru = model_gru.predict(X_test)


test_predictions_gru = scaler_y.inverse_transform(test_predictions_gru) 



# Calculate RMSE
rmse_gru = np.sqrt(mean_squared_error(actual_y_test, test_predictions_gru))
print(f"Root Mean Squared Error (RMSE) on Test Data: {rmse_gru}")










