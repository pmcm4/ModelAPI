from flask import Flask, jsonify, send_file
import io
from RiverCastAPI.rivercastModel import forecast, initiate_model_instance, getAttnScores, updateMainData, getRiverCastMAE
from bidirectionalAPI.bidirectionalModel import initiate_model_instance_bi, bi_getAttnScores, bi_forecast, getBidirectionalMAE
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import requests
import csv
from datetime import datetime, timedelta
from datetime import date
import pymysql
import mysql.connector
from sqlalchemy import create_engine, inspect, DateTime
from PIL import Image
from flask_cors import CORS  # Import CORS from flask_cors




app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


mydb = mysql.connector.connect(
  host="database-1.cccp1zhjxtzi.ap-southeast-1.rds.amazonaws.com",
  user="admin",
  password="Nath1234",
  database= "rivercast"
)

engine = create_engine("mysql+pymysql://" + "admin" + ":" + "Nath1234" + "@" + "database-1.cccp1zhjxtzi.ap-southeast-1.rds.amazonaws.com" + "/" + "rivercast")

cursor = mydb.cursor()

print(mydb)

# RIVERCAST APIs

# Function to generate and save the plot
def save_plot(data, filename):
    plt.plot(data)
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    plt.clf()  # Clear the plot for the next use
    return image_stream


# Endpoint for raw data plot
@app.route('/RC_raw_data_plot', methods=['GET'])
def rc_raw_data_plot():
    
    rd = initiate_model_instance.rawData
    image_stream = save_plot(rd, 'raw_data_plot.png')
    return send_file(image_stream, mimetype='image/png')

# Endpoint for clean data plot
@app.route('/RC_clean_data_plot', methods=['GET'])
def rc_clean_data_plot():
    
    cd = initiate_model_instance.cleanData
    image_stream = save_plot(cd, 'clean_data_plot.png')
    return send_file(image_stream, mimetype='image/png')


@app.route('/RC_attention_scores', methods=['GET'])
def attention_scores():
    attn_score_images = getAttnScores()

    # Create a composite image containing all attention score images
    composite_image = create_composite_image(attn_score_images)

    # Save the composite image to a BytesIO object
    composite_image_stream = io.BytesIO()
    composite_image.savefig(composite_image_stream, format='png')
    composite_image_stream.seek(0)

    # Close the figure to clear the plot
    plt.close(composite_image)

    # Send the composite image as a response
    return send_file(composite_image_stream, mimetype='image/png')

def create_composite_image(images):
    # Create a composite image horizontally stacking all attention score images
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 5, 5))

    for ax, image in zip(axes, images):
        # Display the image on the axis
        ax.imshow(Image.open(image))
        ax.axis('off')

    # Close the figure to clear the plot
    plt.close(fig)
    
    return fig


@app.route('/addRCPredictionToDB', methods=['GET'])
def rc_addPrediction():
    # Call forecast function
    
    df1 = forecast()[0]

    # Check if the DataFrame is empty
    if df1.empty:
        return jsonify("No prediction data to add to DB")

    # Assuming 'date_time' is the column name in your DataFrame
    df1.set_index('DateTime', inplace=True)

    # Use 'DateTime' as the index label in the database
    df1.to_sql(name='rivercast_waterlevel_prediction', con=engine, index=True, index_label='DateTime', if_exists='append', method='multi')

    return jsonify("Add Prediction Values to DB initiated")
    
@app.route('/addRCTrueValuesToDB', methods=['GET'])
def rc_addTrueValues():
    
    # Call forecast function
    df2 = forecast()[1]

    # Check if the DataFrame is empty
    if df2.empty:
        return jsonify("No true values data to add to DB")

    # Assuming 'date_time' is the column name in your DataFrame
    df2.set_index('DateTime', inplace=True)

    # Use 'DateTime' as the index label in the database
    df2.to_sql(name='rivercast_waterlevel_obs', con=engine, index=True, index_label='DateTime', if_exists='append', method='multi')

    return jsonify("Add True Values to DB initiated")

@app.route('/getrivercastMAE', methods=['GET'])
def rc_addMAE():
    
    # Call forecast function
    df2 = getRiverCastMAE()

    # Assuming 'date_time' is the column name in your DataFrame
    df2.set_index('Datetime', inplace=True)

    # Use 'DateTime' as the index label in the database
    df2.to_sql(name='rivercast_df_with_MAE', con=engine, index=True, index_label='Datetime', if_exists='append', method='multi')

    return jsonify("Add MAE to DB initiated")

@app.route('/updateModelData', methods=['GET'])
def updateModelData():
    update_result = updateMainData()

    if update_result[1] != "Data are up-to-date":
        dftosql = update_result[0]
        engine = create_engine("mysql+pymysql://" + "admin" + ":" + "Nath1234" + "@" + "database-1.cccp1zhjxtzi.ap-southeast-1.rds.amazonaws.com" + "/" + "rivercast")

        # Create an inspector and check if the table 'modelData' already exists in the database
        inspector = inspect(engine)
        table_exists = inspector.has_table("modelData")

        # If the table exists, append data without duplicates
        if table_exists:
            dftosql.to_sql(name="modelData", con=engine, index=False, if_exists="append", index_label=False, method="multi", chunksize=1000)
        else:
            # If the table doesn't exist, create it and insert the data
            dftosql.to_sql(name="modelData", con=engine, index=False, if_exists="replace", index_label=False)

        return jsonify("Model Data updated!")
    else:
        return jsonify("Data are up-to-date")

# BIDIRECTIONAL APIs

# Endpoint for raw data plot
@app.route('/bidirectional_raw_data_plot', methods=['GET'])
def bi_raw_data_plot():
    
    rd = initiate_model_instance.rawData
    image_stream = save_plot(rd, 'raw_data_plot.png')
    return send_file(image_stream, mimetype='image/png')

# Endpoint for clean data plot
@app.route('/bidirectional_clean_data_plot', methods=['GET'])
def bi_clean_data_plot():
    
    cd = initiate_model_instance_bi.cleanData
    image_stream = save_plot(cd, 'clean_data_plot.png')
    return send_file(image_stream, mimetype='image/png')


@app.route('/bidirectional_attention_scores', methods=['GET'])
def bi_attention_scores():
    
    attn_score_images = bi_getAttnScores()

    # Create a composite image containing all attention scores
    composite_image = create_composite_image(attn_score_images)

    # Save the composite image to a BytesIO object
    composite_image_stream = io.BytesIO()
    composite_image.savefig(composite_image_stream, format='png')
    composite_image_stream.seek(0)

    # Send the composite image as a response
    return send_file(composite_image_stream, mimetype='image/png')

@app.route('/addBiPredictionToDB', methods=['GET'])
def bi_addPrediction():
    # Call forecast function
    df1 = bi_forecast()[0]

    # Check if the DataFrame is empty
    if df1.empty:
        return jsonify("No prediction data to add to DB")

    # Assuming 'date_time' is the column name in your DataFrame
    df1.set_index('DateTime', inplace=True)

    # Use 'DateTime' as the index label in the database
    df1.to_sql(name='bidirectional_waterlevel_prediction', con=engine, index=True, index_label='DateTime', if_exists='append', method='multi')

    return jsonify("Add Bidirectional Prediction Values to DB initiated")

@app.route('/addBiTrueValuesToDB', methods=['GET'])
def bi_addTrueValues():
    
    # Call forecast function
    df2 = bi_forecast()[1]

    # Check if the DataFrame is empty
    if df2.empty:
        return jsonify("No true values data to add to DB")

    # Assuming 'date_time' is the column name in your DataFrame
    df2.set_index('DateTime', inplace=True)

    # Use 'DateTime' as the index label in the database
    df2.to_sql(name='bidirectional_waterlevel_obs', con=engine, index=True, index_label='DateTime', if_exists='append', method='multi')

    return jsonify("Add True Values to DB initiated")

@app.route('/getBidirectionalMAE', methods=['GET'])
def bi_addMAE():
    
    # Call forecast function
    df2 = getBidirectionalMAE()

    # Assuming 'date_time' is the column name in your DataFrame
    df2.set_index('Datetime', inplace=True)

    # Use 'DateTime' as the index label in the database with a specified key length
    df2.to_sql(
        name='bidirectional_df_with_MAE',
        con=engine,
        index=True,
        index_label='Datetime',
        if_exists='append',
        method='multi',
        dtype={'Datetime': DateTime(50)}  # Specify the key length as needed
    )

    return jsonify("Add MAE to DB initiated")


if __name__ == '__main__':
    app.run(debug=True)


