import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import math
import numpy as np
import scipy as sc
import requests
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
import json
import mysql.connector
from datetime import datetime, timedelta


mydb = mysql.connector.connect(
host="localhost",
user="root",
password="1234",
database= "rivercast"
)

class initiate_model():

    def __init__(self):
        self.initialize_model()

    def initialize_model(self):

        mydb._open_connection()
        query = "SELECT * FROM rivercast.modeldata;"
        result_dataFrame = pd.read_sql(query, mydb)


        # Specify the column to exclude (change 'column_to_exclude' to the actual column name)
        column_to_exclude = 'Date_Time'

        # Exclude the specified column
        df = result_dataFrame.drop(column_to_exclude, axis=1, errors='ignore')

        # Print the DataFrame without the excluded column

        # Now 'df' can be used as 'mainDataToDB' or for further processing

        # convert month name to integer

        # create datetime column
        df[['Year', 'Month', 'Day', 'Hour']] = df[['Year', 'Month', 'Day', 'Hour']].astype(int)
        df['Hour'] = df['Hour'].apply(lambda x: x if x < 24 else 0)

        # convert year, month, day, and hour columns into timestamp
        df['Datetime'] = df[['Year', 'Month', 'Day', 'Hour']].apply(lambda row: datetime(row['Year'], row['Month'], row['Day'], row['Hour']).isoformat(), axis=1)
        df["Datetime"] = pd.to_datetime(df["Datetime"], format='ISO8601')

        # assign timestamps as the data frame index
        df.index = df["Datetime"]
        df = df.drop(['Datetime'], axis=1)

        # select the parameters
        df = df[['Waterlevel', 'Waterlevel.1', 'Waterlevel.2', 'Waterlevel.3', 'RF-Intensity', 'RF-Intensity.1', 'RF-Intensity.2', 'RF-Intensity.3', 'Precipitation', 'Precipitation.1', 'Precipitation.2', 'Humidity', 'Humidity.1', 'Humidity.2', 'Temperature', 'Temperature.1', 'Temperature.2']] 
        df = df.astype(np.float64)  # convert parameters into a double precision floating number

        # fill in missing values using linear interpolation
        df = df.interpolate(method='linear', limit_direction='forward')
        df = df.resample('6H').max()  # resample dataset using the max value for each 24-hours
        df = df.rolling(120).mean().dropna()  # perform moving average smoothing

        self.sampling = df

        self.rawData = df

        self.dataset_min = df.min()
        self.dataset_max = df.max()

        df = (df - self.dataset_min) / (self.dataset_max - self.dataset_min)

        #PCA AND EUCLIDEAN KERNEL

        # center data
        rainfall_df = df[['RF-Intensity', 'RF-Intensity.1', 'RF-Intensity.2', 'RF-Intensity.3']]

        plt.plot(rainfall_df, color='k', alpha=0.2)

        # calculate pairwise squared Euclidean distances
        sq_dists = sc.spatial.distance.pdist(rainfall_df.values.T, 'sqeuclidean')

        # convert pairwise distances into a square matrix
        mat_sq_dists = sc.spatial.distance.squareform(sq_dists)

        # compute the symmetric kernel matrix.
        gamma = 1 / len(rainfall_df.columns)
        K = np.exp(-gamma * mat_sq_dists)

        # center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        # calculate components
        rainfall_df = np.matmul(rainfall_df, eigenvectors) 
        rainfall_df = rainfall_df.iloc[:, 1]

        # center data
        precipitation_df = df[['Precipitation', 'Precipitation.1', 'Precipitation.2']]

        plt.plot(precipitation_df, color='k', alpha=0.2)

        # calculate pairwise squared Euclidean distances
        sq_dists = sc.spatial.distance.pdist(precipitation_df.values.T, 'sqeuclidean')

        # convert pairwise distances into a square matrix
        mat_sq_dists = sc.spatial.distance.squareform(sq_dists)

        # compute the symmetric kernel matrix.
        gamma = 1/len(precipitation_df.columns)
        K = np.exp(-gamma * mat_sq_dists)

        # center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        # calculate components
        precipitation_df = np.matmul(precipitation_df, eigenvectors) 
        precipitation_df = precipitation_df.iloc[:, 1]

        # center data
        humidity_df = df[['Humidity', 'Humidity.1', 'Humidity.2']]

        plt.plot(humidity_df, color='k', alpha=0.2)

        # calculate pairwise squared Euclidean distances
        sq_dists = sc.spatial.distance.pdist(humidity_df.values.T, 'sqeuclidean')

        # convert pairwise distances into a square matrix
        mat_sq_dists = sc.spatial.distance.squareform(sq_dists)

        # compute the symmetric kernel matrix.
        gamma = 1/len(humidity_df.columns)
        K = np.exp(-gamma * mat_sq_dists)

        # center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        # calculate components
        humidity_df = np.matmul(humidity_df, eigenvectors) 
        humidity_df = humidity_df.iloc[:, 1]

        # center data
        temp_df = df[['Temperature', 'Temperature.1', 'Temperature.2']]

        plt.plot(temp_df, color='k', alpha=0.2)

        # calculate pairwise squared Euclidean distances
        sq_dists = sc.spatial.distance.pdist(temp_df.values.T, 'sqeuclidean')

        # convert pairwise distances into a square matrix
        mat_sq_dists = sc.spatial.distance.squareform(sq_dists)

        # compute the symmetric kernel matrix.
        gamma = 1/len(temp_df.columns)
        K = np.exp(-gamma * mat_sq_dists)

        # center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        # calculate components
        temp_df = np.matmul(temp_df, eigenvectors) 
        temp_df = temp_df.iloc[:, 1]

        weather_df = pd.concat([rainfall_df, precipitation_df, humidity_df, temp_df], axis=1)
        weather_df.columns = ['Rainfall', 'Precipitation', 'Humidity', 'Temperature']

        river_df = df[['Waterlevel', 'Waterlevel.1', 'Waterlevel.2', 'Waterlevel.3']]
        self.reduced_df = pd.concat([river_df, weather_df], axis=1)

        self.cleanData = self.reduced_df

        mydb.close()

initiate_model_instance = initiate_model()

BATCH_SIZE = 128
SEQ_LEN = 180
SEQ_STEP = 60
PRED_SIZE = 8
D_MODEL = 8
NUM_HEADS = 4
D_FF = 2048 

# neural network functions
def linear_activation(input, weights, biases):
    batch_size, seq_length, d_model = input.shape  # extract input shape
    
    x_flat = np.reshape(input, (batch_size * seq_length, d_model))  # flatten input into (batch_size, d_model)
    z_flat = np.dot(x_flat, weights.T) + biases
    
    return np.reshape(z_flat, (batch_size, seq_length, -1))  # reshape back to (batch_size, seq_length, d_model)


def relu(input):
    batch_size, seq_length, d_model = input.shape  # extract input shape
    
    x_flat = np.reshape(input, (batch_size * seq_length, d_model))  # flatten input into (batch_size, d_model)
    a_flat = np.maximum(x_flat, 0) 
    
    return np.reshape(a_flat, (batch_size, seq_length, -1))  # reshape back to (batch_size, seq_length, d_model)


def sigmoid(input):
    batch_size, seq_length, d_model = input.shape  # extract input shape
    
    x_flat = np.reshape(input, (batch_size * seq_length, d_model))  # flatten input into (batch_size, d_model)
    a_flat = 1 / (1 + np.exp(-x_flat))
    
    return np.reshape(a_flat, (batch_size, seq_length, -1))  # reshape back to (batch_size, seq_length, d_model)


def softmax(input):
    batch_size, seq_length, d_model = input.shape
    
    x_flat = np.reshape(input, (batch_size * seq_length, d_model)).T  # flatten input into (batch_size, d_model)
    a_flat = np.exp(x_flat) / (np.sum(np.exp(x_flat), axis=0) + 1e-8)
    
    return np.reshape(a_flat.T, (batch_size, seq_length, -1))  # reshape back to (batch_size, seq_length, d_model)


def layer_normalization(input, gamma, beta):
    mean = np.mean(input, axis=-1, keepdims=True)  # get mean in each axis
    std = np.std(input, axis=-1, keepdims=True)  # get standard deviation in each axis
    
    normalized = (input - mean) / (std + 1e-8)  # normalized activations 
    
    # reshape parameters to fit the input shape
    gamma = np.reshape(gamma, (1, 1, -1))
    beta = np.reshape(beta, (1, 1, -1))
    
    return gamma * normalized + beta  # normalized activations with size of (batch_size, seq_length, d_model)


# positional encoding
def positional_encoding(input, n=10000):
    batch_size, seq_length, d_model = input.shape
    
    pe = np.zeros(shape=(seq_length, d_model))
    for k in range(seq_length):
        for i in np.arange(int(d_model / 2)):
            denominator = np.power(n, 2 * i / d_model)
            pe[k, 2*i] = np.sin(k / denominator)
            pe[k, 2*i+1] = np.cos(k / denominator)
            
    return input + pe  # add positional encoding to input

# multi-head attention
def split_heads(input, num_heads):
    batch_size, seq_length, d_model = input.shape
    
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads
    
    heads = np.reshape(input, (batch_size, seq_length, num_heads, head_dim))
    heads = np.transpose(heads, (0, 2, 1, 3))
    
    return heads  # attention heads with size of (batch_size, num_heads, seq_length, head_dim)


def combine_heads(input):
    combined = np.transpose(input, (0, 2, 1, 3))
    combined = np.reshape(combined, (combined.shape[0], combined.shape[1], -1))
    
    return combined  # combined attention heads with size of (batch_size, seq_length, d_model)


def scaled_dot_product_attention(query, key, value):
    batch_size, num_heads, seq_length, head_dim = query.shape
    
    # convert input into (batch_size, seq_length, d_model)
    query = np.reshape(query, (batch_size * num_heads, seq_length, head_dim))  
    key = np.reshape(key, (batch_size * num_heads, seq_length, head_dim))
    value = np.reshape(value, (batch_size * num_heads, seq_length, head_dim))
    
    key = np.transpose(key, (0, 2, 1))  # transpose key
    attn_scores = np.matmul(query, key) / math.sqrt(head_dim)  # get dot product attention
    
    # generate auto-regressive mask
    attn_scores = np.ma.array(attn_scores, mask=np.triu(np.ones(shape=(batch_size * num_heads, seq_length, seq_length)), k=1))
    attn_scores = attn_scores.filled(fill_value=-1e9)
    
    attn_scores = softmax(attn_scores)  # convert attention scores into probabilities
    
    value = np.matmul(attn_scores, value)  # embed attention scores into value
    
    return np.reshape(attn_scores, (batch_size, num_heads, seq_length, seq_length)), np.reshape(value, (batch_size, num_heads, seq_length, head_dim))  # reshape to original size


def multi_head_self_attention(query, key, value, num_heads, params):
    query = split_heads(linear_activation(query, params[0], params[1]), num_heads)
    key = split_heads(linear_activation(key, params[2], params[3]), num_heads)
    value = split_heads(linear_activation(value, params[4], params[5]), num_heads)
    
    attn_scores, attn_output = scaled_dot_product_attention(query, key, value)
    attn_output = linear_activation(combine_heads(attn_output), params[6], params[7])
    
    return attn_scores, attn_output

# feed forward network
def feed_forward_network(input, params):
    out = linear_activation(input, params[0], params[1])
    out = relu(out)
    out = linear_activation(out, params[2], params[3])
    
    return out

# decoder layer
def transformer_decoder(input, num_heads, params):
    attn_scores, attn_out = multi_head_self_attention(
        query=input,
        key=input, 
        value=input, 
        num_heads=num_heads, 
        params=params[:8])
    norm1 = layer_normalization(input + attn_out, params[12], params[13])
    ff_out = feed_forward_network(norm1, params[8:12])
    norm2 = layer_normalization(norm1 + ff_out, params[14], params[15])
    
    return attn_scores, norm2

# model
def transformer(input, num_heads, params):
    out = positional_encoding(input)
    
    # decoder layers
    _, out = transformer_decoder(out, num_heads, params[:16])
    scores, out = transformer_decoder(out, num_heads, params[16:32])
    
    # final layer
    out = linear_activation(out, params[32], params[33])
    out = sigmoid(out)
    
    return scores, out

# load parameters from file
with open("rivercast_parameters.json", "r") as parameters:
    saved_params = json.load(parameters)
    
# iterate through layer parameters
params = []
for key in saved_params.keys():
    param = np.asarray(saved_params[key], dtype=np.float32)  # convert saved parameters back to numpy
    params.append(param)
    
len(params)  # print number of layer parameters




def inverse_transform(data):
    data_min = initiate_model_instance.dataset_min[['Waterlevel', 'Waterlevel.1', 'Waterlevel.2', 'Waterlevel.3']].to_numpy()
    data_max = initiate_model_instance.dataset_max[['Waterlevel', 'Waterlevel.1', 'Waterlevel.2', 'Waterlevel.3']].to_numpy()
    
    return (data_max - data_min) * data + data_min


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def forecast():
    test_data = initiate_model_instance.reduced_df[-180:].values
    test_dates = initiate_model_instance.reduced_df[-180:].index
    test_dates = test_dates[60:240]    
    x_test = test_data[:180]
    y_label = test_data[60:]
    y_label = inverse_transform(y_label[:, :4])

    x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

    attn_scores, y_test = transformer(input=x_test, num_heads=NUM_HEADS, params=params)  # make forecast
    y_test = np.reshape(y_test, (y_test.shape[1], y_test.shape[2]))
    y_test = inverse_transform(y_test[:, :4])


    time_steps_per_day = 4  # Assuming 4 time steps per day (6 hours per time step)
    forecast_days = 15
    mydb._open_connection()
    cursor = mydb.cursor()
    cursor.execute("SELECT DateTime FROM rivercast.rivercast_waterlevel_prediction order by DateTime DESC LIMIT 1")
    lastPredDT = cursor.fetchone()[0]
    formatted_lastPredDT = lastPredDT.strftime('%Y-%m-%d %H:%M:%S')
    # Extract the forecast for the next 15 days
    forecast_values = y_test[:]

    # Create a DataFrame with the forecasted values and dates
    forecast_dates = pd.date_range(test_dates[-120], periods=180, freq='6H')[:] # the -120 selects the oldest date as the start date, then match the size of y_test in the periods.
    forecast_df = pd.DataFrame(data=forecast_values, columns=['P.Waterlevel', 'P.Waterlevel-1', 'P.Waterlevel-2', 'P.Waterlevel-3'])
    forecast_df.insert(0, "DateTime", forecast_dates)

    matches_and_following_rows_pred = forecast_df[forecast_df['DateTime'] >= formatted_lastPredDT] # then match the last datetime predicted to dynamically adjust the predicted values in the database



    cursor.execute("SELECT DateTime FROM rivercast.rivercast_waterlevel_obs order by DateTime DESC LIMIT 1")
    lastTrueDT = cursor.fetchone()[0] + timedelta(hours=6)

    # Extract the forecast for the next 15 days
    true_values = y_label[-120:]

    true_dates = pd.date_range(test_dates[-120], periods=120, freq='6H')[:]
    true_df = pd.DataFrame(data=true_values ,columns=['T.Waterlevel', 'T.Waterlevel-1', 'T.Waterlevel-2', 'T.Waterlevel-3']) #converting numpy to dataframe
    true_df.insert(0, "DateTime", true_dates) #adding DateTime column


    formatted_lastTrueDT = lastTrueDT.strftime('%Y-%m-%d %H:%M:%S')

    matches_and_following_rows = true_df[true_df['DateTime'] >= formatted_lastTrueDT]
    mydb.close()
    return matches_and_following_rows_pred[1:2], matches_and_following_rows
 

test_data = initiate_model_instance.reduced_df['2021-03-15':].values
dataset_len = len(test_data) - (SEQ_LEN + SEQ_STEP) + 1

# prepare batches
batches = []
for index in range(dataset_len):
    in_start = index
    in_end = in_start + SEQ_LEN
    out_start = index + SEQ_STEP
    out_end = out_start + SEQ_LEN
    
    input = test_data[in_start:in_end]
    label = test_data[out_start:out_end]
    
    batches.append((np.array(input), np.array(label)))

def getRiverCastMAE():
    # measure accuracy of each window
    accuracy = []
    predictions = []
    for input, label in batches:
        
        input = np.reshape(input, (1, SEQ_LEN, D_MODEL))
        scores, pred = transformer(input=input, num_heads=NUM_HEADS, params=params)  # make forecast
        pred = np.reshape(pred, (SEQ_LEN, D_MODEL))  
        pred = inverse_transform(pred[:, :4])  # scale output to original value
        pred = pred[-SEQ_STEP:]   # get only the forecast window
        
        ground = inverse_transform(label[:, :4])  # scale output to original value
        ground = ground[-SEQ_STEP:]  # get only the forecast window
        
        accuracy.append(mean_absolute_error(ground, pred))  # collect mean absolute error of each window
        predictions.append(np.concatenate((pred[0], ground[0])))  # collect first element of output

        
    accuracy_df = pd.DataFrame(np.array(accuracy), columns=['MAE'])
    predictions_df = pd.DataFrame(np.array(predictions), columns=['P_Waterlevel', 'P_Waterlevel.1', 'P_Waterlevel.2', 'P_Waterlevel.3', 'T_Waterlevel', 'T_Waterlevel.1', 'T_Waterlevel.2', 'T_Waterlevel.3'])
    metric_df = pd.concat([accuracy_df, predictions_df], axis=1)
    metric_df.index = initiate_model_instance.sampling.index[-len(metric_df):]

    metric_df = metric_df.resample('24H').max()
    metric_df.to_csv('rivercast_results.csv')  # save test results

    pass_metric = pd.read_csv('rivercast_results.csv')


    a_MAEs = []
    std_p = []

    aMAE = metric_df['MAE'].mean()
    a_MAEs.append(aMAE)

    st_d = metric_df['MAE'].std()
    std_p.append(st_d)


    aveMAE = pd.DataFrame(np.array(a_MAEs), columns = ['aMAE'])
    pass_std = pd.DataFrame(np.array(std_p), columns = ['std'])

    pass_MAEs = pd.concat([aveMAE, pass_std], axis=1)

    pass_MAEs['cnt'] = pass_MAEs.index


    return pass_metric, pass_MAEs

def getForecastforDateRangeFunction():
    pass_metric_df = pd.read_csv('numpy_rivercast_date_range.csv')

    return pass_metric_df



def getLatest_Datetime():
    mydb._open_connection()
    cursor = mydb.cursor()

    cursor.execute("SELECT Date_Time FROM rivercast.modeldata order by Date_Time DESC LIMIT 1")
    lastDTindex = cursor.fetchone()

    
    mydb.close()
    return lastDTindex



def updateMainData():
    initiate_model_instance = initiate_model()
    initiate_model_instance.initialize_model()

    getLatest_Datetime()
    lastDTindexDef = getLatest_Datetime()

    ldi = str(lastDTindexDef).replace("(datetime.datetime(", "").replace("),)", "").replace(", ", "-")

    lastDT = datetime.strptime(ldi, '%Y-%m-%d-%H-%M')


    # Common date calculations
    d = lastDT

    h = d.hour
    m = d.minute
    s = d.second
    ms = d.microsecond

    dday = d + timedelta(hours=1)

    datetoday = dday.strftime("%Y-%m-%d:%H")

    startDate = datetoday 

    ed = datetime.today()


    edate = ed - timedelta(minutes=m, seconds=s, microseconds=ms) + timedelta(hours=1)


    endDate = edate.strftime("%Y-%m-%d:%H")

    # Weather data
    weatherbit = f'https://api.weatherbit.io/v2.0/history/hourly?lat=14.679696901082357&lon=121.10970052493437&start_date={startDate}&end_date={endDate}&tz=local&key=0e0b9299ef62428f9deafe70a3f4d39a'
    wbRes = requests.get(weatherbit)
    wbReq = wbRes.json()
    try:
        wbReq = wbRes.json()
        wbitArr = []

        for current_weather in wbReq['data']:
            time_date = current_weather['timestamp_local']
            dpt = current_weather['dewpt']
            prsr = current_weather['pres']
            preci = current_weather['precip']
            temperature = current_weather['temp']
            humi = specific_humidity_from_dewpoint(prsr * units.hPa, dpt * units.degC).to('g/kg')
            
            strHumi = str(humi)
            humidity = strHumi.replace(" gram / kilogram", "")

            wbitArr.append({
                "humidity": humidity,
                "precipitation": preci,
                "temperature": temperature
            })

        weather_df = pd.DataFrame(wbitArr)

        weather_df['temperature1'] = weather_df['temperature'].copy()
        weather_df['humidity1'] = weather_df['humidity'].copy()
        weather_df['precipitation1'] = weather_df['precipitation'].copy()

        weather_df['temperature2'] = weather_df['temperature'].copy()
        weather_df['humidity2'] = weather_df['humidity'].copy()
        weather_df['precipitation2'] = weather_df['precipitation'].copy()

        weather_df = weather_df.reindex(columns=['humidity', 'precipitation', 'temperature', 'temperature1', 'humidity1', 'precipitation1', 'temperature2', 'humidity2', 'precipitation2'])

    except KeyError:
        print("Data are up-to-date")
        weather_df = pd.DataFrame()

    # Water level data
    wlArr = []

    base_url = 'http://121.58.193.173:8080/water/map_list.do?ymdhm='
    obsnm_list = ["Nangka", "Sto Nino", "Montalban"]


    start_date = datetime(dday.year, dday.month, dday.day, dday.hour)
    end_date = datetime(edate.year, edate.month, edate.day, edate.hour)
    current_date = start_date


    try:
        for current_date in pd.date_range(start=start_date, end=end_date, freq='H'):
            formatted_date = current_date.strftime('%Y%m%d%H%M')
            url = f"{base_url}{formatted_date}"

            try:
                response = requests.get(url)
                data = response.json()

                entry_data = {}
                
                for entry in data:
                    for i, obsnm in enumerate(obsnm_list, start=1):
                        
                        if obsnm in entry['obsnm']:
                            year = int(formatted_date[:4])
                            month = int(formatted_date[4:6])
                            day = int(formatted_date[6:8])
                            hour = int(formatted_date[8:10])
                            waterlevel = entry.get('wl', 'null')
                            
                            # Remove "(*)" from waterlevel
                            waterlevel = waterlevel.replace("(*)", "")

                            entry_data.update({
                                "date_time": pd.to_datetime(f"{year}-{month:02d}-{day:02d} {hour:02d}:00:00"),  # Create Date_Time column
                                f"station{i}": entry['obsnm'],
                                f"year{i}": year,
                                f"month{i}": month,
                                f"day{i}": day,
                                f"hour{i}": hour,
                                f"waterlevel{i}": waterlevel
                            })
                wlArr.append(entry_data)

            except Exception as e:
                print(f"No waterlevel fetched for {formatted_date}: {e}")

    except KeyError:
        print("Water level data are up-to-date")
        waterlevel_df = pd.DataFrame()

    waterlevel_df = pd.DataFrame(wlArr)

    # Check if 'waterlevel2' column exists before trying to access it
    if 'waterlevel2' in waterlevel_df.columns:
        waterlevel_df['waterlevel2dup'] = waterlevel_df['waterlevel2'].copy()
    else:
        print("Data are up-to-date")

    waterlevel_df = waterlevel_df.reindex(columns=['station1', 'year1', 'month1', 'day1', 'hour1','waterlevel1', 'station2', 'year2', 'month2', 'day2', 'hour2','waterlevel2', 'waterlevel2dup', 'station3', 'year3', 'month3', 'day3', 'hour3','waterlevel3','date_time'])

    waterlevel_df = waterlevel_df.rename(columns={"station1": "station_1","station2": "station_2"})


    # Rainfall data
    wlArr = []

    base_url = 'http://121.58.193.173:8080/rainfall/map_list.do?ymdhm='
    obsnm_list = ["Nangka", "Mt. Oro"]

    current_date = start_date

    try:
        for current_date in pd.date_range(start=start_date, end=end_date, freq='H'):
            formatted_date = current_date.strftime('%Y%m%d%H%M')
            url = f"{base_url}{formatted_date}"

            try:
                response = requests.get(url)
                data = response.json()

                entry_data = {}

                for entry in data:
                    for i, obsnm in enumerate(obsnm_list, start=1):
                        if obsnm in entry['obsnm']:
                            rainfall = entry.get('rf', 'null')
                            
                            # Remove "(*)" from rainfall
                            rainfall = rainfall.replace("(*)", "")

                            entry_data.update({
                                f"station{i}": entry['obsnm'],
                                f"rainfall{i}": rainfall,
                                "date_time": current_date  # Use current_date instead of start_date
                            })

                wlArr.append(entry_data)

            except Exception as e:
                print(f"No rainfall data fetched for {formatted_date}: {e}")

    except KeyError:
        print("Rainfall data are up-to-date")
        rainfall_df = pd.DataFrame()

    rainfall_df = pd.DataFrame(wlArr)

    # Check if 'rainfall1' column exists before trying to access it
    if 'rainfall1' in rainfall_df.columns:
        rainfall_df['rainfall1dup'] = rainfall_df['rainfall1']
    else:
        print("Data are up-to-date")
    updatedData = ""
    # Similarly, check for 'rainfall2' column
    if 'rainfall2' in rainfall_df.columns:
        rainfall_df['rainfall2dup'] = rainfall_df['rainfall2']
    else:
        updatedData = "Data are up-to-date"

    rainfall_df = rainfall_df.reindex(columns=['station1', 'rainfall1', 'rainfall1dup', 'station2', 'rainfall2', 'rainfall2dup'])


    # Consolidate all data into one DataFrame
    merged_df = pd.concat([waterlevel_df, rainfall_df, weather_df], axis=1).dropna()


    # Assuming your DataFrame is called merged_df
    new_columns = {
        'station_1': 'Station',
        'year1': 'Year',
        'month1': 'Month',
        'day1': 'Day',
        'hour1': 'Hour',
        'waterlevel1': 'Waterlevel',
        'station_2': 'Station.1',
        'year2': 'Year.1',
        'month2': 'Month.1',
        'day2': 'Day.1',
        'hour2': 'Hour.1',
        'waterlevel2': 'Waterlevel.1',
        'waterlevel2dup': 'Waterlevel.2',
        'station3': 'Station.2',
        'year3': 'Year.2',
        'month3': 'Month.2',
        'day3': 'Day.2',
        'hour3': 'Hour.2',
        'waterlevel3': 'Waterlevel.3',
        'station1': 'RF-Station',
        'rainfall1': 'RF-Intensity',
        'rainfall1dup': 'RF-Intensity.1',
        'station2': 'RF-Station.1',
        'rainfall2': 'RF-Intensity.2',
        'rainfall2dup': 'RF-Intensity.3',
        'humidity': 'Humidity',
        'precipitation': 'Precipitation',
        'temperature': 'Temperature',
        'temperature1': 'Temperature.1',
        'humidity1': 'Humidity.1',
        'precipitation1': 'Precipitation.1',
        'temperature2': 'Temperature.2',
        'humidity2': 'Humidity.2',
        'precipitation2': 'Precipitation.2',
        'date_time': 'Date_Time'
    }

    merged_df.rename(columns=new_columns, inplace=True)


    # Save to CSV
    merged_df.to_csv('consolidated_data.csv', index=False)

    return merged_df, updatedData



def get_parameters():

    r_df = initiate_model_instance.rawData.tail(1)
    pass_df =  pd.DataFrame(r_df)

    return pass_df


def get_added_params():
    # Get the current datetime
    current_datetime = datetime.now()

    api_key = '0e0b9299ef62428f9deafe70a3f4d39a'
    # Weather data for location 1
    weatherbit_url1 = f'https://api.weatherbit.io/v2.0/current?lat=14.636063697609517&lon=121.09320016726963&key={api_key}'
    response1 = requests.get(weatherbit_url1)
    data1 = response1.json()

    # Weather data for location 2
    weatherbit_url2 = f'https://api.weatherbit.io/v2.0/current?lat=14.673803785437409&lon=121.10951228864191&key={api_key}'
    response2 = requests.get(weatherbit_url2)
    data2 = response2.json()

    # Weather data for location 3
    weatherbit_url3 = f'https://api.weatherbit.io/v2.0/current?lat=14.733416817860283&lon=121.13032299916969&key={api_key}'
    response3 = requests.get(weatherbit_url3)
    data3 = response3.json()

    # Extract relevant information for location 1
    current_weather1 = data1['data'][0]
    loc1_temp = current_weather1['temp']
    loc1_dewpoint = current_weather1['dewpt']
    loc1_pressure = current_weather1['pres']
    loc1_humidity = specific_humidity_from_dewpoint(loc1_pressure * units.hPa, loc1_dewpoint * units.degC).to('g/kg').magnitude
    loc1_precipitation = current_weather1.get('precip', 0)

    # Extract relevant information for location 2
    current_weather2 = data2['data'][0]
    loc2_temp = current_weather2['temp']
    loc2_dewpoint = current_weather2['dewpt']
    loc2_pressure = current_weather2['pres']
    loc2_humidity = specific_humidity_from_dewpoint(loc2_pressure * units.hPa, loc2_dewpoint * units.degC).to('g/kg').magnitude
    loc2_precipitation = current_weather2.get('precip', 0)

    # Extract relevant information for location 3
    current_weather3 = data3['data'][0]
    loc3_temp = current_weather3['temp']
    loc3_dewpoint = current_weather3['dewpt']
    loc3_pressure = current_weather3['pres']
    loc3_humidity = specific_humidity_from_dewpoint(loc3_pressure * units.hPa, loc3_dewpoint * units.degC).to('g/kg').magnitude
    loc3_precipitation = current_weather3.get('precip', 0)

    # Create a DataFrame
    data = {
        'loc1_temp': [loc1_temp],
        'loc1_humidity': [loc1_humidity],
        'loc1_precipitation': [loc1_precipitation],
        'loc2_temp': [loc2_temp],
        'loc2_humidity': [loc2_humidity],
        'loc2_precipitation': [loc2_precipitation],
        'loc3_temp': [loc3_temp],
        'loc3_humidity': [loc3_humidity],
        'loc3_precipitation': [loc3_precipitation],
        'Datetime': [current_datetime]
    }

    current_weather_df = pd.DataFrame(data)

    return current_weather_df