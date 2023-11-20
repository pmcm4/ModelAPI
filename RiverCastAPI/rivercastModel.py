import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import calendar
import math
import torch
import torch.nn as nn
import numpy as np
import scipy as sc
from sklearn.preprocessing import MinMaxScaler
from skimage.measure import block_reduce
from sklearn.metrics import mean_absolute_error
import io
import requests
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units

import mysql.connector
from datetime import datetime, timedelta
from sqlalchemy import create_engine

mydb = mysql.connector.connect(
host="database-1.cccp1zhjxtzi.ap-southeast-1.rds.amazonaws.com",
user="admin",
password="Nath1234",
database= "rivercast"
)

class initiate_model():
    #IMPORTING
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # configure GPU    utilization
    device

    mydb._open_connection()
    query = "SELECT * FROM rivercast.modelData;"
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


    rawData = df

    # scale data
    scaler = MinMaxScaler()
    scaler.fit(df)
    # train label scaler
    label_scaler = MinMaxScaler()
    label_scaler.fit(df[['Waterlevel', 'Waterlevel.1', 'Waterlevel.2', 'Waterlevel.3']])

    scaled_ds = scaler.transform(df)
    df = pd.DataFrame(scaled_ds, columns=df.columns, index=df.index)

    #PCA AND EUCLIDEAN KERNEL

    # center data
    rainfall_df = df[['RF-Intensity', 'RF-Intensity.1', 'RF-Intensity.2', 'RF-Intensity.3']]


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
    reduced_df = pd.concat([river_df, weather_df], axis=1)




    cleanData = reduced_df


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, step):
        self.data = data
        self.seq_len = seq_len
        self.step = step
        
    def __getitem__(self, index):
        in_start = index
        in_end = in_start + self.seq_len
        out_start = index + self.step
        out_end = out_start + self.seq_len
        
        inputs = self.data[in_start:in_end]
        labels = self.data[out_start:out_end]
        
        return inputs, labels
    
    def __len__(self):
        return len(self.data) - (self.seq_len + self.step) + 1

BATCH_SIZE = 128
SEQ_LEN = 180
SEQ_STEP = 60
PRED_SIZE = 8
D_MODEL = 8
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 2048 
DROPOUT = 0.10

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        return attn_probs, output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_scores, attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return attn_scores, output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=2048):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_scores, attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return attn_scores, x
    
class Transformer(nn.Module):
    def __init__(self, pred_size, d_model, num_heads, num_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, pred_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, tgt):
        seq_length = tgt.size(1)
        tgt_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        return tgt_mask

    def forward(self, tgt):
        mask = self.generate_mask(tgt).to(initiate_model.device)
        tgt_embedded = self.dropout(self.positional_encoding(tgt))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            attn_scores, dec_output = dec_layer(dec_output, mask)

        output = self.sigmoid(self.fc(dec_output))
        return attn_scores, output

# define the model
decomposer = Transformer(
    pred_size=PRED_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT
).float()

decomposer.to(initiate_model.device)

decomposer.load_state_dict(torch.load('rivercast_transformer.pth'))

decomposer.eval()  # set model on test mode

mydb.close()

def forecast():
    test_data = initiate_model.reduced_df[-180:].values
    test_dates = initiate_model.reduced_df[-180:].index
    test_dates = test_dates[60:240]

    x_test = test_data[:180]
    y_label = test_data[60:]
    y_label = initiate_model.label_scaler.inverse_transform(y_label[:, :4])

    x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

    decomposer.eval()  # set model on test mode

    x_test = torch.from_numpy(x_test).float().to(initiate_model.device)
    attn_scores, y_test = decomposer(x_test)  # make forecast
    y_test = y_test.detach().cpu().numpy()
    y_test = np.reshape(y_test, (y_test.shape[1], y_test.shape[2]))
    y_test = initiate_model.label_scaler.inverse_transform(y_test[:, :4])


    time_steps_per_day = 4  # Assuming 4 time steps per day (6 hours per time step)
    forecast_days = 15
    
    mydb._open_connection()
    cursor = mydb.cursor()
    cursor.execute("SELECT DateTime FROM rivercast.rivercast_waterlevel_prediction order by DateTime DESC LIMIT 1")
    lastPredDT = cursor.fetchone()[0]
    formatted_lastPredDT = lastPredDT.strftime('%Y-%m-%d %H:%M:%S')

    # Extract the forecast for the next 15 days
    forecast_values = y_test[:forecast_days * time_steps_per_day]

    # Create a DataFrame with the forecasted values and dates
    forecast_dates = pd.date_range(test_dates[-1], periods=forecast_days * time_steps_per_day + 1, freq='6H')[1:]
    forecast_df = pd.DataFrame(data=forecast_values, columns=['P.Waterlevel', 'P.Waterlevel-1', 'P.Waterlevel-2', 'P.Waterlevel-3'])
    forecast_df.insert(0, "DateTime", forecast_dates)

    matches_and_following_rows_pred = forecast_df[forecast_df['DateTime'] >= formatted_lastPredDT]



    cursor.execute("SELECT DateTime FROM rivercast.rivercast_waterlevel_obs order by DateTime DESC LIMIT 1")
    lastTrueDT = cursor.fetchone()[0] + timedelta(hours=6)

    # Extract the forecast for the next 15 days
    true_values = y_label[-120:]

    true_dates = pd.date_range(test_dates[-120], periods=120, freq='6H')[:]
    true_df = pd.DataFrame(data=true_values ,columns=['T.Waterlevel', 'T.Waterlevel-1', 'T.Waterlevel-2', 'T.Waterlevel-3']) #converting numpy to dataframe
    true_df.insert(0, "DateTime", true_dates) #adding DateTime column

    puirpose = pd.DataFrame(data=y_label ,columns=['T.Waterlevel', 'T.Waterlevel-1', 'T.Waterlevel-2', 'T.Waterlevel-3'])

    formatted_lastTrueDT = lastTrueDT.strftime('%Y-%m-%d %H:%M:%S')

    mydb.close()

    matches_and_following_rows = true_df[true_df['DateTime'] >= formatted_lastTrueDT]

    return matches_and_following_rows_pred[1:2], matches_and_following_rows




def getLatest_Datetime():
    mydb._open_connection()
    cursor = mydb.cursor()

    cursor.execute("SELECT Date_Time FROM rivercast.modelData order by Date_Time DESC LIMIT 1")
    lastDTindex = cursor.fetchone()


    return lastDTindex



def updateMainData():

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
    weatherbit = f'https://api.weatherbit.io/v2.0/history/hourly?lat=14.679696901082357&lon=121.10970052493437&start_date={startDate}&end_date={endDate}&tz=local&key=2b382660ad4843188647514206bf330e'
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
    mydb.close()

    return merged_df, updatedData




def getAttnScores():
    test_data = initiate_model.reduced_df['2023-09-27':].values
    test_dates = initiate_model.reduced_df['2023-09-27':].index
    test_dates = test_dates[60:240]

    x_test = test_data[:180]
    y_label = test_data[60:180]
    y_label = initiate_model.label_scaler.inverse_transform(y_label[:, :4])

    x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

    decomposer.eval()  # set model on test mode

    x_test = torch.from_numpy(x_test).float().to(initiate_model.device)
    attn_scores, y_test = decomposer(x_test)  # make forecast
    y_test = y_test.detach().cpu().numpy()
    y_test = np.reshape(y_test, (y_test.shape[1], y_test.shape[2]))
    y_test = initiate_model.label_scaler.inverse_transform(y_test[:, :4])

        # plot predictions
    for i in [0, 1, 2, 3]:
        plt.plot(np.convolve(y_test[:, i], np.ones(30), 'valid') / 30)
        plt.plot(y_label[30:, i], color='k', alpha=0.3)
        plt.show()

    # plot attention scores
    attn_scores = torch.squeeze(attn_scores, dim=0)
    attn_scores = attn_scores.detach().cpu().numpy()  # transfer output from GPU to CPU
    
    
    attention_score_images = []

    for idx, attention in enumerate(attn_scores):
        selected_attention = attention[10:]
        selected_attention = block_reduce(selected_attention, (15, 15), np.max)

        fig, ax = plt.subplots()
        ax.matshow(selected_attention, cmap='viridis')

        # Save the plot to a BytesIO object
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        # Append the image stream to the list
        attention_score_images.append(image_stream)

    # Return the list of attention score images
    return attention_score_images