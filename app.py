import numpy as np
import pickle
from flask import Flask, request, render_template

# Load ML model
model = pickle.load(open('heart_disease_model.pickle', 'rb'))

# Create application
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]

    # Convert features to array
    array_features = [np.array(features)]

    # Predict features
    prediction = model.predict(array_features)
    print(prediction)
    output = prediction

    # Check the output values and retrieve the result with html tag based on the value
    if output == 0:
        return render_template('index.html',
                               result='Heart disease - Unlikely')
    else:
        return render_template('index.html',
                               result='Heart disease - Likely')


if __name__ == '__main__':
    # Run the application
    app.run()
