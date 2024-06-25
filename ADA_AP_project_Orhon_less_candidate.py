# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:11:13 2024

@author: murat
"""

import pandas as pd 
import numpy as np
import streamlit as st
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)




options = ["Do optimization", "Use determined values"]
choice = st.selectbox("Choose an option", options)
if choice== "Do optimization" :
    value_opt = 1
else:
    value_opt = 0

sliding_window = st.number_input("Enter sliding window value", min_value=1, max_value=730, value=30)

min_date = pd.Timestamp('2015-09-09')
max_date = pd.Timestamp('2022-12-31')
default_date=pd.Timestamp('2015-09-09')
specific_date_optimization_starts = st.date_input("Select a date for optimization starts", min_value=min_date, max_value=max_date, value=default_date)
specific_date_optimization_ends = st.date_input("Select a date for optimization ends", min_value=min_date, max_value=max_date, value=default_date)
specific_date_estimation_starts= st.date_input("Select a date for estimation starts ", min_value=min_date, max_value=max_date, value=default_date)
specific_date_estimation_ends = st.date_input("Select a date for estimation ends", min_value=min_date, max_value=max_date, value=default_date)


specific_date_optimization_starts= pd.Timestamp(specific_date_optimization_starts)
specific_date_optimization_ends= pd.Timestamp(specific_date_optimization_ends)
specific_date_estimation_starts= pd.Timestamp(specific_date_estimation_starts)
specific_date_estimation_ends= pd.Timestamp(specific_date_estimation_ends)


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
df_daily_consumption = pd.read_csv('C:/Users/murat/OneDrive/Masaüstü/unil master/semester 2/advanced data analysis/project/data/daily_consumption.csv', encoding='utf-8')
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




# model is trained and tested between spesific dates
combined_input_data_bfr_2022 = combined_input_data[combined_input_data.index < specific_date_optimization_ends]
combined_input_data_bfr_2022 = combined_input_data[combined_input_data.index > specific_date_optimization_starts]

# apply one shift in order to estimatedemand in next day 
combined_input_data_x = combined_input_data_bfr_2022.iloc[:-1,:]
combined_input_data_y = combined_input_data_bfr_2022.iloc[1:,:]
# Split data into inputs and outputs
X = scaler_x.fit_transform(combined_input_data_x.iloc[:,[0,1,2,3,4,5]])  # 
y = scaler_y.fit_transform(combined_input_data_y.iloc[:,4].values.reshape(-1, 1))  # 

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def lstm_calculation(X_train,y_train,X_test,y_test ):
    print("lstm beggining2")
    batch_size = [64,128]
    lstm_unit = [128]
    number_of_epoch = [100]
    results_lstm = pd.DataFrame()
    results_lstm['Batch_size'] = None
    results_lstm['LSTM_units'] = None
    results_lstm['Epoch'] = None
    results_lstm['RMSE_LSTM'] = None
    for m in number_of_epoch:
        for t in lstm_unit:
            for i in batch_size:
                # Build the LSTM model
                model_lstm = Sequential([
                    LSTM(t, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                    LSTM(t),
                    Dense(1)
                ])
                
                # Compile the model
                model_lstm.compile(optimizer='adam', loss='mean_squared_error')
                
                # Train the model
                model_lstm.fit(X_train, y_train, epochs=m, batch_size=i, validation_data=(X_test, y_test), verbose=1)
                
                predicted_values_lstm = model_lstm.predict(X_test)
                predicted_values_lstm = scaler_y.inverse_transform(predicted_values_lstm)  # Inverse scaling to original scale
                
                # Inverse scaling the test labels for actual comparison
                actual_y_test_lstm = scaler_y.inverse_transform(y_test)
                
                # RMSE Calculation
                mse_lstm = mean_squared_error(actual_y_test_lstm, predicted_values_lstm)
                rmse_lstm = np.sqrt(mse_lstm)
                print("Root Mean Squared Error (RMSE):", rmse_lstm)
                new_df_lstm = []
                new_df_lstm = pd.DataFrame({ 'Batch_size': [i],'LSTM_units': [t],'Epoch': [m], 'RMSE_LSTM': [rmse_lstm] })
                results_lstm = pd.concat([results_lstm, new_df_lstm], ignore_index=True)
                print("lstm")

    return results_lstm


def gru_calculation(X_train,y_train,X_test,y_test ):

    batch_size = [64,128]
    gru_unit = [128]
    number_of_epoch = [100]
    results_gru = pd.DataFrame()
    results_gru['Batch_size'] = None
    results_gru['Gru_units'] = None
    results_gru['Epoch'] = None
    results_gru['RMSE_GRU'] = None
 
    for z in number_of_epoch:
        for d in gru_unit:
            for p in batch_size:
                # Assuming X_train.shape[1] is the sequence length and X_train.shape[2] is the number of features
                model_gru = Sequential([
                    GRU(d, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                    GRU(d),
                    Dense(1)
                ])
                
                # Compile the model
                model_gru.compile(optimizer='adam', loss='mean_squared_error')
                
                # Train the model
                model_gru.fit(X_train, y_train, epochs=z, batch_size=p, validation_data=(X_test, y_test), verbose=1)
                
                
                test_predictions_gru = model_gru.predict(X_test)
                
                
                test_predictions_gru = scaler_y.inverse_transform(test_predictions_gru) 
                
                
                actual_y_test_gru = scaler_y.inverse_transform(y_test)
                # Calculate RMSE
                rmse_gru = np.sqrt(mean_squared_error(actual_y_test_gru, test_predictions_gru))
                new_df_gru = []
                new_df_gru = pd.DataFrame({ 'Batch_size': [p],'Gru_units': [d],'Epoch': [z], 'RMSE_GRU': [rmse_gru] })
                results_gru = pd.concat([results_gru, new_df_gru], ignore_index=True)
                
              
    return results_gru


## compute parellelly
import concurrent.futures
from tensorflow.keras.layers import GRU, Dense

if value_opt == 1:
    if __name__ == '__main__':
        # Define input data frames for each loop
    
    
        # Parallelize the execution of the function for each loop
       with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(lstm_calculation,X_train,y_train,X_test,y_test )
            future2 = executor.submit(gru_calculation, X_train,y_train,X_test,y_test )
    
            result_lstm = future1.result()
            result_gru = future2.result()
    lstm_best_params = result_lstm[result_lstm['RMSE_LSTM'] == result_lstm['RMSE_LSTM'].min()]
    gru_best_params = result_gru[result_gru['RMSE_GRU'] == result_gru['RMSE_GRU'].min()]
    st.write("LSTM Best Parameters;")
    st.write(lstm_best_params)
    st.write("GRU Best Parameters;")
    st.write(gru_best_params)
else: 
    lstm_best_params_val = [[16,16,60]]
    lstm_best_params = pd.DataFrame(lstm_best_params_val, columns=['Batch_size', 'LSTM_units', 'Epoch'])
    gru_best_params_val = [[16,128,100]]
    gru_best_params = pd.DataFrame(gru_best_params_val, columns=['Batch_size', 'Gru_units', 'Epoch'])
    st.write("LSTM Best Parameters;")
    st.write(lstm_best_params)
    st.write("GRU Best Parameters;")
    st.write(gru_best_params)



# 3d plot if optimization is chosen 
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import matplotlib as plt
if value_opt == 1:
    st.write("LSTM Parameters;")
       # MinMaxScaler
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    sizes = (scaler_x.fit_transform(result_lstm["RMSE_LSTM"].values.reshape(-1, 1))+1) * 25
    
    # Define categories and colors
    categories = result_lstm['LSTM_units'].unique()
    colors = ["blue"]
    color_map = dict(zip(categories, colors))
    
    # Scatter plot
    fig = go.Figure()
    for cat in categories:
        mask = result_lstm['LSTM_units'] == cat
        fig.add_trace(go.Scatter3d(
            x=result_lstm[mask]['Batch_size'],
            y=result_lstm[mask]['Epoch'],
            z=result_lstm[mask]['RMSE_LSTM'],
            mode='markers',
            marker=dict(
                size=sizes[mask],
                color=color_map[cat],
                opacity=1,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            name=cat
        ))
    
    # Set layout properties
    fig.update_layout(
        scene=dict(
            xaxis_title='Batch size',
            yaxis_title='Epoch',
            zaxis_title='RMSE',
            aspectmode='cube',  # Adjust aspect ratio
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Adjust viewing angle
        ),
        margin=dict(r=0, l=0, b=0, t=0),  # Adjust margins
        title='LSTM Parameters Plot',  # Add a title
    )
    
    # Add legend
    fig.update_layout(legend_title="LSTM Units")
    
    # Title
    fig.update_layout(title='LSTM Parameters')
    
    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

    #####gru plot
       # MinMaxScaler
    st.write("GRU Parameters;")
    sizes = (scaler_x.fit_transform(result_gru["RMSE_GRU"].values.reshape(-1, 1))+1) * 10
    
    # Define categories and colors
    categories = result_gru['Gru_units'].unique()
    colors = ["blue"]
    color_map = dict(zip(categories, colors))
    
    # Scatter plot
    fig = go.Figure()
    for cat in categories:
        mask = result_gru['Gru_units'] == cat
        fig.add_trace(go.Scatter3d(
            x=result_gru[mask]['Batch_size'],
            y=result_gru[mask]['Epoch'],
            z=result_gru[mask]['RMSE_GRU'],
            mode='markers',
            marker=dict(
                size=sizes[mask],
                color=color_map[cat],
                opacity=1,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            name=cat
        ))
    
    # Set layout properties
    fig.update_layout(
        scene=dict(
            xaxis_title='Batch size',
            yaxis_title='Epoch',
            zaxis_title='RMSE',
            aspectmode='cube',  # Adjust aspect ratio
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Adjust viewing angle
            #xaxis=dict(nticks=4),  # Adjust number of ticks
           # yaxis=dict(nticks=4),
            #zaxis=dict(nticks=10),
        ),
        margin=dict(r=0, l=0, b=0, t=0),  # Adjust margins
        title=('GRU Parameters Plot'),  # Add a title
    )
    
    # Add legend
    fig.update_layout(legend_title="GRU Units")
    
    # Title
    fig.update_layout(title='GRU Parameters')
    
    # Display the plot in the Streamlit app
    st.plotly_chart(fig)


######estimation part
def lstm_estimation(specific_date_estimation_starts,specific_date_estimation_ends ,combined_input_data,sliding_window,lstm_best_params):
    estimation_period = specific_date_estimation_starts + relativedelta(days=-(sliding_window+2))
    
    combined_input_data_aftr_2022 = combined_input_data[combined_input_data.index > estimation_period]
    combined_input_data_aftr_2022 = combined_input_data[combined_input_data.index < specific_date_estimation_ends]
    # find the estimation interval 
    estimation_range = (combined_input_data_aftr_2022.index.max() - specific_date_estimation_starts)
    
    
    # sliding window size will be 24, model trained for 24 month and estimates 1 month 
    
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    
    RMSE_lstm = 0
    entire_predicted_values_lstm = []
    
    for i in range(estimation_range.days):
        # apply one shift in order to estimatedemand in next day  we use the previous month data to predict next month
        # same month data can be used but in this case, we need to wait until end of month to get the air data correctly
        combined_input_data_x = combined_input_data_aftr_2022.iloc[(-estimation_range.days-(sliding_window+2)+i):(-estimation_range.days-1+i),:] # the last element will be considered as x test
        combined_input_data_y = combined_input_data_aftr_2022.iloc[(-estimation_range.days-(sliding_window+1)+i):(-estimation_range.days-1+i),:]
        
        # Normalize the data
        # Split data into inputs and outputs
        X = scaler_x.fit_transform(combined_input_data_x.iloc[:,[0,1,2,3,4,5]]) 
        X = X[:-1,:]# 
        X_test = np.transpose(X[-1,:])
        y = scaler_y.fit_transform(combined_input_data_y.iloc[:,4].values.reshape(-1, 1))  # 
        
        # Reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        X_test = X_test.reshape((-1, 1, 6))
        
        
        # Split the data into training and testing sets
        # model trained with data one before specific_date_raw, so it will use  specific_date_raw date as first estimation
        
        
        X_train = X
        y_train = y
        model_lstm = Sequential([
            LSTM(int(lstm_best_params["LSTM_units"].iloc[0]), return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(int(lstm_best_params["LSTM_units"].iloc[0])),
            Dense(1)
        ])
    
        # Compile the model
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        # Train the model
        model_lstm.fit(X_train, y_train, epochs=int(lstm_best_params["Epoch"].iloc[0]), batch_size=int(lstm_best_params["Batch_size"].iloc[0]), verbose=1)
        
        predicted_values_lstm = model_lstm.predict(X_test)
        predicted_values_lstm = scaler_y.inverse_transform(predicted_values_lstm)  # Inverse scaling to original scale
        # Inverse scaling the test labels for actual comparison
        actual_y_test_lstm = combined_input_data_aftr_2022.iloc[(-estimation_range.days+i),4]
        predicted_values_lstm = float(predicted_values_lstm[0, 0])
        # RMSE Calculation
        mse_lstm = np.square((actual_y_test_lstm - predicted_values_lstm))
        #rmse_lstm = np.sqrt(mse_lstm)
        rmse_lstm = mse_lstm
        RMSE_lstm = RMSE_lstm + rmse_lstm
        entire_predicted_values_lstm.append(predicted_values_lstm)
    
    
    result_df = combined_input_data_aftr_2022.iloc[(-estimation_range.days-1):(-1),:]
    result_df["lstm_estimation"] = entire_predicted_values_lstm
    RMSE_lstm = RMSE_lstm/(len(result_df))   # divide by the number of data points
    RMSE_lstm_sqrt = np.sqrt(RMSE_lstm)
    result_df["RMSE_lstm"] = RMSE_lstm_sqrt
    
    return result_df


##########################



def gru_estimation(specific_date_estimation_starts,specific_date_estimation_ends ,combined_input_data,sliding_window,gru_best_params):
    
    estimation_period = specific_date_estimation_starts + relativedelta(days=-(sliding_window+2))
    combined_input_data_aftr_2022 = combined_input_data[combined_input_data.index > estimation_period]
    combined_input_data_aftr_2022 = combined_input_data[combined_input_data.index < specific_date_estimation_ends]
    # find the estimation interval 
    estimation_range = (combined_input_data_aftr_2022.index.max() - specific_date_estimation_starts)
    
    # sliding window size will be 24, model trained for 24 month and estimates 1 month 
    
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    
    RMSE_gru = 0
    entire_predicted_values_gru = []
    
    for i in range(estimation_range.days):
        # apply one shift in order to estimatedemand in next day  we use the previous month data to predict next month
        # same month data can be used but in this case, we need to wait until end of month to get the air data correctly
        combined_input_data_x = combined_input_data_aftr_2022.iloc[(-estimation_range.days-(sliding_window+2)+i):(-estimation_range.days-1+i),:] # the last element will be considered as x test
        combined_input_data_y = combined_input_data_aftr_2022.iloc[(-estimation_range.days-(sliding_window+1)+i):(-estimation_range.days-1+i),:]
        
        
        
        # Normalize the data
        
        
        # Split data into inputs and outputs
        X = scaler_x.fit_transform(combined_input_data_x.iloc[:,[0,1,2,3,4,5]]) 
        X = X[:-1,:]# 
        X_test = np.transpose(X[-1,:])
        y = scaler_y.fit_transform(combined_input_data_y.iloc[:,4].values.reshape(-1, 1))  # 
        
        # Reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        X_test = X_test.reshape((-1, 1, 6))
        
        
        # Split the data into training and testing sets
        # model trained with data one before specific_date_raw, so it will use  specific_date_raw date as first estimation
        
    
        
        X_train = X
        y_train = y
        model_gru = Sequential([
            GRU(int(gru_best_params["Gru_units"].iloc[0]), return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            GRU(int(gru_best_params["Gru_units"].iloc[0])),
            Dense(1)
        ])
    
        # Compile the model
        model_gru.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        model_gru.fit(X_train, y_train, epochs=int(gru_best_params["Epoch"].iloc[0]), batch_size=int(gru_best_params["Batch_size"].iloc[0]), verbose=1)
        
        
        test_predictions_gru = model_gru.predict(X_test)
        
        
        test_predictions_gru = scaler_y.inverse_transform(test_predictions_gru) 
        actual_y_test_gru = combined_input_data_aftr_2022.iloc[(-estimation_range.days+i),4]
        test_predictions_gru = float(test_predictions_gru[0, 0])
        # RMSE Calculation
        mse_gru = np.square((actual_y_test_gru - test_predictions_gru))
        rmse_gru = mse_gru
        RMSE_gru = RMSE_gru + rmse_gru
        entire_predicted_values_gru.append(test_predictions_gru)
    
    
    result_df_gru = combined_input_data_aftr_2022.iloc[(-estimation_range.days-1):(-1),:]
    result_df_gru["gru_estimation"] = entire_predicted_values_gru
    RMSE_gru = RMSE_gru/(len(result_df_gru))   # divide by the number of data points
    RMSE_gru_sqrt = np.sqrt(RMSE_gru)
    result_df_gru["RMSE_Gru"] = RMSE_gru_sqrt

    return result_df_gru

from dateutil.relativedelta import relativedelta


if __name__ == '__main__':
    # Define input data frames for each loop


    # Parallelize the execution of the function for each loop
   with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(lstm_estimation,specific_date_estimation_starts,specific_date_estimation_ends ,combined_input_data,sliding_window,lstm_best_params )
        future2 = executor.submit(gru_estimation, specific_date_estimation_starts,specific_date_estimation_ends ,combined_input_data,sliding_window,gru_best_params )

        result_df_lstm = future1.result()
        result_df_gru = future2.result()



st.write('Plotting Demand, LSTM Prediction, and GRU Prediction')


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# Plot the first line
ax.plot(result_df_lstm.index.date, result_df_lstm["Demand"], label='Demand', color='black', alpha = 0.5)

ax.plot(result_df_lstm.index.date, result_df_lstm["lstm_estimation"], label='LSTM Prediction', color='red')
ax.plot(result_df_lstm.index.date, result_df_gru["gru_estimation"], label='GRU Prediction', color='blue')
# Set labels and title
ax.set_xlabel('Date')
ax.set_ylabel('kWh')
ax.tick_params(axis='x', rotation=45)
plt.title('LSTM and GRU Prediction vs Demand')
ax.legend(loc='upper left')
st.pyplot(fig)



