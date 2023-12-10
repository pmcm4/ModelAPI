from flask import Flask, jsonify, send_file, g
from RiverCastAPI.rivercastModel import forecast, initiate_model_instance, updateMainData, getRiverCastMAE, getForecastforDateRangeFunction, get_parameters, get_added_params
from bidirectionalAPI.bidirectionalModel import initiate_model_instance_bi, bi_forecast, getBidirectionalMAE, getForecastforDateRangeFunction_bi
import matplotlib.pyplot as plt
import mysql.connector
from sqlalchemy import create_engine, inspect, DateTime
from PIL import Image
from flask_cors import CORS  # Import CORS from flask_cors
import os
from statsmodels.stats.weightstats import ttest_ind
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True)


DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "rivercast"
}

def get_db():
    if 'db' not in g:
        g.db = mysql.connector.connect(**DB_CONFIG)
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.before_request
def before_request():
    g.db = get_db()

@app.teardown_request
def teardown_request(e=None):
    close_db()

engine = create_engine("mysql+pymysql://" + "root" + ":" + "1234" + "@" + "localhost" + "/" + "rivercast")

# RIVERCAST APIs

def save_plot(data, filename, directory, text_color='white', inner_box_border_color='white'):
    # Assuming 'data' is your data to be plotted
    fig, ax = plt.subplots(facecolor='#0e1318')
    ax.set_facecolor('#0e1318')  # Set the background color of the inner area
    ax.plot(data)

    # Customize the color of tick labels
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)

    # Set the color of the inner box border
    for spine in ax.spines.values():
        spine.set_edgecolor(inner_box_border_color)

    # Create the full path for saving the image
    full_path = os.path.join(directory, filename)

    # Save the plot to the specified directory
    plt.savefig(full_path, format='png')
    
    # Clear the plot to prevent it from being displayed in the response
    plt.clf()
    plt.close()

    return full_path

# Endpoint for raw data plot
@app.route('/RC_raw_data_plot', methods=['GET'])
def rc_raw_data_plot():
    
    rd = initiate_model_instance.rawData['2022-01-01':]

    save_directory = '../client/src/assets/rivercastImages'

    image_stream = save_plot(rd, 'RawData.png', save_directory)

    return send_file(image_stream, mimetype='image/png')

# Endpoint for clean data plot
@app.route('/RC_clean_data_plot', methods=['GET'])
def rc_clean_data_plot():
    # Assuming initiate_model_instance.cleanData is already defined
    cd = initiate_model_instance.cleanData['2022-01-01':]
    
    # Specify the directory where you want to save the image
    save_directory = '../client/src/assets/rivercastImages'
    
    # Save the plot and get the full path of the saved image
    image_path = save_plot(cd, 'cleanData.png', save_directory)
    
    # Send the saved image as a response
    return send_file(image_path, mimetype='image/png')



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
    df2 = getRiverCastMAE()[0]

    # Assuming 'date_time' is the column name in your DataFrame
    df2.set_index('Datetime', inplace=True)

    # Use 'DateTime' as the index label in the database with a specified key length
    df2.to_sql(
        name='rivercast_df_with_mae',
        con=engine,
        index=True,
        index_label='Datetime',
        if_exists='replace',
        method='multi',
        dtype={'Datetime': DateTime(50)}  # Specify the key length as needed
    )

        # Call forecast function
    df3 = getRiverCastMAE()[1]

    # Assuming 'date_time' is the column name in your DataFrame
    df3.set_index('cnt', inplace=True)

    # Use 'DateTime' as the index label in the database with a specified key length
    df3.to_sql(
        name='rivercast_overall_maes',
        con=engine,
        index=True,
        index_label='cnt',
        if_exists='replace',
        method='multi',
    )

    

    return jsonify("Add RC MAE to DB initiated")

@app.route('/updateModelData', methods=['GET'])
def updateModelData():
    update_result = updateMainData()

    if update_result[1] != "Data are up-to-date":
        dftosql = update_result[0]
        engine = create_engine("mysql+pymysql://" + "root" + ":" + "1234" + "@" + "localhost" + "/" + "rivercast")

        # Create an inspector and check if the table 'modelData' already exists in the database
        inspector = inspect(engine)
        table_exists = inspector.has_table("modeldata")

        # If the table exists, append data without duplicates
        if table_exists:
            dftosql.to_sql(name="modeldata", con=engine, index=False, if_exists="append", index_label=False, method="multi", chunksize=1000)
        else:
            # If the table doesn't exist, create it and insert the data
            dftosql.to_sql(name="modeldata", con=engine, index=False, if_exists="replace", index_label=False)

        return jsonify("Model Data updated!")
    else:
        return jsonify("Data are up-to-date")


@app.route('/rc_updateDateRangeData', methods=['GET'])
def rc_updateDRdata():
    df = getForecastforDateRangeFunction_bi()

    df.set_index('Datetime', inplace=True)
    df.to_sql(name='rivercast_daterange_data', con=engine, index=True, index_label='Datetime', if_exists='replace', method='multi', dtype={'Datetime': DateTime(50)})

    return jsonify("DateRange Data Updated")


# BIDIRECTIONAL APIs

# Endpoint for raw data plot
@app.route('/bidirectional_raw_data_plot', methods=['GET'])
def bi_raw_data_plot():
    
    rd = initiate_model_instance.rawData['2022-01-01':]

    save_directory = '../client/src/assets/biimages'

    image_stream = save_plot(rd, 'rawDataBidirectional-removebg-preview.png', save_directory)

    return send_file(image_stream, mimetype='image/png')

# Endpoint for clean data plot
@app.route('/bidirectional_clean_data_plot', methods=['GET'])
def bi_clean_data_plot():
    
    cd = initiate_model_instance_bi.cleanData['2022-01-01':]

    save_directory = '../client/src/assets/biimages'

    image_stream = save_plot(cd, 'cleanDataBidirectional-removebg-preview.png', save_directory)

    return send_file(image_stream, mimetype='image/png')


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
    df2 = getBidirectionalMAE()[0]

    # Assuming 'date_time' is the column name in your DataFrame
    df2.set_index('Datetime', inplace=True)

    # Use 'DateTime' as the index label in the database with a specified key length
    df2.to_sql(
        name='bidirectional_df_with_mae',
        con=engine,
        index=True,
        index_label='Datetime',
        if_exists='replace',
        method='multi',
        dtype={'Datetime': DateTime(50)}  # Specify the key length as needed
    )

        # Call forecast function
    df3 = getBidirectionalMAE()[1]

    # Assuming 'date_time' is the column name in your DataFrame
    df3.set_index('cnt', inplace=True)

    # Use 'DateTime' as the index label in the database with a specified key length
    df3.to_sql(
        name='bidirectional_overall_maes',
        con=engine,
        index=True,
        index_label='cnt',
        if_exists='replace',
        method='multi',
    )

    return jsonify("Add Bi MAE to DB initiated")

@app.route('/bi_updateDateRangeData', methods=['GET'])
def bi_updateDRdata():
    df = getForecastforDateRangeFunction()
    
    df.set_index('Datetime', inplace=True)
    df.to_sql(name='bidirectional_daterange_data', con=engine, index=True, index_label='Datetime', if_exists='replace', method='multi', dtype={'Datetime': DateTime(50)})

    return jsonify("DateRange Data Updated")

@app.route('/getPValue', methods=['GET'])
def pVal():
    # Call forecast function
    rc = pd.read_csv('rivercast_results.csv')
    bi = pd.read_csv('bidirectional_results.csv')
    pVal = []

    t_stat, p_value, degrees_of_freedom = ttest_ind(rc['MAE'], bi['MAE'], alternative='two-sided', usevar='pooled', weights=(None, None), value=0)

    p_val = p_value

    pVal.append(p_val)

    pValDF = pd.DataFrame(np.array(pVal), columns = ['pValue'])

    pValDF['cnt'] = pValDF.index

    pValDF.set_index('cnt', inplace=True)

    pValDF.to_sql(
        name='pvalue',
        con=engine,
        index=True,
        index_label='cnt',
        if_exists='replace',
        method='multi',
    )

    return jsonify("Added pValue")

@app.route('/tempupdates', methods=['GET'])
def tempupdates():
    df = pd.read_csv('bidirectional_results_daterange.csv')
    
    df.set_index('Datetime', inplace=True)
    df.to_sql(name='bidirectional_daterange_data', con=engine, index=True, index_label='Datetime', if_exists='replace', method='multi', dtype={'Datetime': DateTime(50)})

    return jsonify("DateRange Data Updated")


@app.route('/get_parameters', methods=['GET'])
def get_params():
    df = get_parameters()
    
    df.to_sql(name='parameters', con=engine, index=False, if_exists='replace', method='multi')

    return jsonify("Parameters updated")

@app.route('/get_added_parameters', methods=['GET'])
def added_params():
    df = get_added_params()
    
    df.to_sql(name='added_parameters', con=engine, index=False, if_exists='replace', method='multi')

    return jsonify("Added Parameters updated")


if __name__ == '__main__':
    app.run(debug=True)


