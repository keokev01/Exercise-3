from flask import Flask, request
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pickle
import os

app = Flask(__name__)
CORS(app)

@app.route("/api/predict/", methods=['GET'])
def predict():
    bfs_number = int(request.args['bfs_number'])
    area = float(request.args['area'])
    rooms = float(request.args['rooms'])
    model_type = request.args.get('model_type', 'randomforest')  # Standardmäßig wird Random Forest verwendet

    # Laden der BFS-Daten
    df_bfs_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bfs_municipality_and_tax_data.csv'),
                              sep=',', encoding='utf-8')
    df_bfs_data['tax_income'] = df_bfs_data['tax_income'].str.replace("'", "").astype(int)
    df = df_bfs_data[df_bfs_data['bfs_number'] == bfs_number]

    if model_type == 'polynomial':
        # Laden des polynomialen Modells
        model_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polynomial_model.pkl")
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
    else:
        # Laden des Random Forest Modells
        model_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "randomforest_regression.pkl")
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)

    # Vorhersage
    prediction = model.predict([[rooms, area, df['pop'].iloc[0], df['pop_dens'].iloc[0], df['frg_pct'].iloc[0], df['emp'].iloc[0], df['tax_income'].iloc[0], area/rooms]])
    return str(round(prediction[0], 2))

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(debug=True)
