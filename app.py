import pickle
import numpy as np
from flask import Flask, request, app, jsonify, url_for, render_templates
app = Flask(__name__)

# Load the model
model = pickle.load(open('StockPrediction.pkl', 'rb'))
countvector = pickle.load(open('countvector.pkl', 'rb'))

@app.route('/')
def home():
    return render_templates('home.html')

@app.route('/predict_api', methods = ['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = countvector.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug = True)

