from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model/RandomForest.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    N = (request.form.get('N'))
    P = (request.form.get('P'))
    K = (request.form.get('K'))
    temperature = (request.form.get('temperature'))
    humidity = (request.form.get('humidity'))
    ph = (request.form.get('ph'))

    # prediction
    result = model.predict(np.array([[N, P, K, temperature, humidity, ph]]))

    return render_template('predictionresult.html', result=str(result))


if __name__ == '__main__':
    app.run(debug=True)