# Importing essential libraries
from flask import Flask, render_template, request
import os
import numpy as np
from wsgiref import simple_server
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import pickle
import webbrowser
import statsmodels.api as sm

# Load the Linear Regression model
filename = 'UCI-prediction-rfc-model.pkl'
lr = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        airtemp = float(request.form['Air temperature [K]'])
        pteamp = float(request.form['Process temperature [K]'])
        rspeed = float(request.form['Rotational speed [rpm]'])
        torque = float(request.form['Torque [Nm]'])
        toolw = float(request.form['Tool wear [min]'])

        data = np.array([[airtemp, pteamp, rspeed, torque, toolw]])
        my_prediction = lr.predict(data)
        return render_template('result.html', prediction=my_prediction)


@app.route('/report', methods=['POST'])
def report():
    if request.method == 'POST':
        webbrowser.open(
            'file://C:/Users/salhat/Documents/DataScience/Ineuron/ML Project/UCI-Prediction_new/UCI_profile_report.html',
            new=2)

        return render_template('index.html')


if __name__ == '__main__':
    port = int(os.getenv("PORT"))
    host = '0.0.0.0'
    httpd = simple_server.make_server(host=host,port=port, app=app)
    httpd.serve_forever()