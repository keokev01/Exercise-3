from flask import Flask, request
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pickle
import os

app = Flask(__name__)
CORS(app)

@app.route("/api/predict/", methods=['GET'])
def predict():
    bfs_number = int(request.args['bfs_number'])
    area = float(request.args['area'])
    rooms = float(request.args['rooms'])
    
    df_bfs_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bfs_municipality_and_tax_data.csv'),
                            sep=',', encoding='utf-8')
    df_bfs_data['tax_income'] = df_bfs_data['tax_income'].str.replace("'", "")
    df = df_bfs_data[df_bfs_data['bfs_number']==bfs_number]

    randomforest_model = RandomForestRegressor()
    model_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "randomforest_regression.pkl")
    with open(model_filename, 'rb') as f:
        randomforest_model = pickle.load(f)

    # ['rooms' 'area' 'pop' 'pop_dens' 'frg_pct' 'emp' 'tax_income' 'm2_per_rooms']
    prediction = randomforest_model.predict([[rooms, area, df['pop'].iloc[0], df['pop_dens'].iloc[0], df['frg_pct'].iloc[0], df['emp'].iloc[0], df['tax_income'].iloc[0], area/rooms]])
    return str(round(prediction[0],2))

@app.route("/")
def hello_world():

    print(request.args)

    return "<p>Hello, World!</p>"
    