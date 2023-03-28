

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, json, render_template, jsonify


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('glod.html')

@app.route('/predict', methods=['POST', 'GET'])

def predict():

    spx = float(request.form['spx'])
    uso = str(request.form['uso'])
    slv = float(request.form['slv'])
    Eur_Usd = str(request.form['Eur_Usd'])


    X = np.array([[spx,uso,slv,Eur_Usd]])

    my_prediction = model.predict(X)
    r=my_prediction[0]

    return render_template("glod.html", r=r)



if __name__ == "__main__":
    app.run(debug=True,port=7895)
