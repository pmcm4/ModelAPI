import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import math
import numpy as np
import io
import json
import mysql.connector
from datetime import datetime, timedelta


mydb = mysql.connector.connect(
host="localhost",
user="root",
password="1234",
database= "rivercast"
)

class bi_initiate_model():
    def __init__(self):
        self.initialize_model()

    def initialize_model(self):
        #IMPORTING

        mydb._open_connection()
        query = "SELECT * FROM rivercast.modelData;"
        result_dataFrame = pd.read_sql(query, mydb)
        

        # Specify the column to exclude (change 'column_to_exclude' to the actual column name)
        column_to_exclude = ['Date_Time', 'RF-Intensity.1']

        # Exclude the specified column
        df = result_dataFrame.drop(column_to_exclude, axis=1, errors='ignore')
        self.rawData = df
        

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
        df = df[['Waterlevel', 'Waterlevel.1', 'Waterlevel.2', 'Waterlevel.3', 'RF-Intensity', 'RF-Intensity.2', 'RF-Intensity.3', 'Precipitation', 'Precipitation.1', 'Precipitation.2', 'Humidity', 'Humidity.1', 'Humidity.2', 'Temperature', 'Temperature.1', 'Temperature.2']] 
        df = df.astype(np.float64)  # convert parameters into a double precision floating number
        
        # fill in missing values using linear interpolation
        df = df.interpolate(method='linear', limit_direction='forward')
        df = df.resample('6H').max()  # resample dataset using the max value for each 24-hours
        df = df.rolling(120).mean().dropna()  # perform moving average smoothing
        
        self.sampling = df

        self.rawData = df
        

        self.dataset_min = df.min()
        self.dataset_max = df.max()

        self.normalized_df = (df - self.dataset_min) / (self.dataset_max - self.dataset_min)


        self.cleanData = self.normalized_df

        mydb.close()

initiate_model_instance_bi = bi_initiate_model()

BATCH_SIZE = 128
SEQ_LEN = 180
SEQ_STEP = 60
PRED_SIZE = 4
D_MODEL = 16
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
def transformer_encoder(input, num_heads, params):
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
    _, out = transformer_encoder(out, num_heads, params[:16])
    scores, out = transformer_encoder(out, num_heads, params[16:32])
    
    # final layer
    out = linear_activation(out, params[32], params[33])
    out = sigmoid(out)
    
    return scores, out

# load parameters from file
with open("bidirectional_parameters.json", "r") as parameters:
    saved_params = json.load(parameters)

# iterate through layer parameters
params = []
for key in saved_params.keys():
    param = np.asarray(saved_params[key], dtype=np.float32)  # convert saved parameters back to numpy
    params.append(param)
    
len(params)  # print number of layer parameters

def inverse_transform(data):
    data_min = initiate_model_instance_bi.dataset_min[['Waterlevel', 'Waterlevel.1', 'Waterlevel.2', 'Waterlevel.3']].to_numpy()
    data_max = initiate_model_instance_bi.dataset_max[['Waterlevel', 'Waterlevel.1', 'Waterlevel.2', 'Waterlevel.3']].to_numpy()
    
    return (data_max - data_min) * data + data_min


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))






def bi_forecast():
    test_data = initiate_model_instance_bi.normalized_df[-180:].values
    test_dates = initiate_model_instance_bi.normalized_df[-180:].index
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
    cursor.execute("SELECT DateTime FROM rivercast.bidirectional_waterlevel_prediction order by DateTime DESC LIMIT 1")
    lastPredDT = cursor.fetchone()[0]
    formatted_lastPredDT = lastPredDT.strftime('%Y-%m-%d %H:%M:%S')

    # Extract the forecast for the next 15 days
    forecast_values = y_test[:]

    # Create a DataFrame with the forecasted values and dates
    forecast_dates = pd.date_range(test_dates[-120], periods=180, freq='6H')[:]
    forecast_df = pd.DataFrame(data=forecast_values, columns=['P.Waterlevel', 'P.Waterlevel-1', 'P.Waterlevel-2', 'P.Waterlevel-3'])
    forecast_df.insert(0, "DateTime", forecast_dates)

    matches_and_following_rows_pred = forecast_df[forecast_df['DateTime'] >= formatted_lastPredDT]



    cursor.execute("SELECT DateTime FROM rivercast.bidirectional_waterlevel_obs order by DateTime DESC LIMIT 1")
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



test_data = initiate_model_instance_bi.normalized_df['2021-03-15':].values
dataset_len = len(test_data) - (SEQ_LEN + SEQ_STEP) + 1

# prepare batches
batches = []
for index in range(dataset_len):
    in_start = index
    in_end = in_start + SEQ_LEN
    out_start = index + SEQ_STEP
    out_end = out_start + SEQ_LEN
    
    input = test_data[in_start:in_end]
    label = test_data[out_start:out_end, :PRED_SIZE]
    
    batches.append((np.array(input), np.array(label)))
    


def getBidirectionalMAE():
    # measure accuracy of each window
    accuracy = []
    predictions = []
    for input, label in batches:
        
        input = np.reshape(input, (1, SEQ_LEN, D_MODEL))
        scores, pred = transformer(input=input, num_heads=NUM_HEADS, params=params)  # make forecast
        pred = np.reshape(pred, (SEQ_LEN, PRED_SIZE))  
        pred = inverse_transform(pred[:, :4])  # scale output to original value
        pred = pred[-SEQ_STEP:]   # get only the forecast window
        
        ground = inverse_transform(label[:, :4])  # scale output to original value
        ground = ground[-SEQ_STEP:]  # get only the forecast window
        
        accuracy.append(mean_absolute_error(ground, pred))  # collect mean absolute error of each window
        predictions.append(np.concatenate((pred[0], ground[0])))  # collect first element of output
    
    
    accuracy_df = pd.DataFrame(np.array(accuracy), columns=['MAE'])
    predictions_df = pd.DataFrame(np.array(predictions), columns=['P_Waterlevel', 'P_Waterlevel.1', 'P_Waterlevel.2', 'P_Waterlevel.3', 'T_Waterlevel', 'T_Waterlevel.1', 'T_Waterlevel.2', 'T_Waterlevel.3'])
    metric_df = pd.concat([accuracy_df, predictions_df], axis=1)
    metric_df.index = initiate_model_instance_bi.sampling.index[-len(metric_df):]

    metric_df = metric_df.resample('24H').max()
    metric_df.to_csv('bidirectional_results.csv')  # save test results

    pass_metric = pd.read_csv('bidirectional_results.csv')


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

def getForecastforDateRangeFunction_bi():

    pass_metric_df = pd.read_csv('numpy_bidirectional_date_range.csv')

    return pass_metric_df