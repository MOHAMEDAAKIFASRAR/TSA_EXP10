# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 21-11-2026
### DataSet Name : AirPassengers
### AIM:
To implement the SARIMA model using Python to analyze monthly air passenger data, make future predictions and evaluate the model performance.

### ALGORITHM:
1. Import the necessary Python libraries.
2. Load the AirPassengers dataset and explore the data.
3. Convert the Month column into datetime format and set it as the index.
4. Plot the monthly air passenger time series data.
5. Check the stationarity of the time series using the ADF test.
6. Plot the ACF and PACF graphs to analyze autocorrelation.
7. Split the dataset into training data (80%) and testing data (20%).
8. Build and fit the SARIMA model using the parameters (1,1,1) and seasonal order (1,1,1,12).
9. Make predictions using the fitted SARIMA model.
10. Calculate the Root Mean Squared Error (RMSE) to evaluate the model.
11. Plot the actual and predicted passenger data.

### PROGRAM:
```python

# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


# Load AirPassengers dataset
data = pd.read_csv('/content/AirPassengers.csv')

# Convert Month column to datetime
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')

# Set Month column as index
data.set_index('Month', inplace=True)
data.sort_index(inplace=True)

# Extract passenger data
data = data['#Passengers'].asfreq('MS')


# AIR PASSENGER TIME SERIES

plt.figure(figsize=(10, 5))

plt.plot(data.index, data)

plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('Air Passengers Time Series')

plt.grid()
plt.show()

# CHECK STATIONARITY

def check_stationarity(timeseries):

    result = adfuller(timeseries)

    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    print('Critical Values:')

    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


check_stationarity(data)

# ACF PLOT

plot_acf(data, lags=30)

plt.show()

# PACF PLOT

plot_pacf(data, lags=30, method='ywm')

plt.show()
 
# SPLIT TRAIN AND TEST DATA


train_size = int(len(data) * 0.8)

train = data.iloc[:train_size]
test = data.iloc[train_size:]

# BUILD SARIMA MODEL

sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

sarima_result = sarima_model.fit(disp=False)

# MAKE PREDICTIONS

predictions = sarima_result.forecast(
    steps=len(test)
)

# CALCULATE RMSE

mse = mean_squared_error(test, predictions)

rmse = np.sqrt(mse)

print('RMSE:', rmse)
# ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(10, 5))

plt.plot(
    test.index,
    test,
    label='Actual'
)

plt.plot(
    test.index,
    predictions,
    color='red',
    label='Predicted'
)

plt.xlabel('Date')
plt.ylabel('Number of Passengers')

plt.title('SARIMA Model Predictions')

plt.legend()
plt.grid()

plt.show()


```
### OUTPUT:
<img width="917" height="600" alt="image" src="https://github.com/user-attachments/assets/5ab050ff-21ba-4d69-bc17-c7f3fb79f9ad" />

<img width="667" height="450" alt="image" src="https://github.com/user-attachments/assets/80503315-95e4-4398-acdc-97410afb4d68" />

<img width="682" height="470" alt="image" src="https://github.com/user-attachments/assets/7fc749bf-6074-4a63-b409-5cf68a1e6409" />

<img width="927" height="477" alt="image" src="https://github.com/user-attachments/assets/b723558a-8c02-4eff-83b8-6913bdd00dee" />

### RESULT:
Thus, the SARIMA model was successfully implemented using the AirPassengers dataset in Python. The monthly air passenger data was analyzed, and predictions were made using the SARIMA model. The model performance was evaluated using the Root Mean Squared Error (RMSE).
