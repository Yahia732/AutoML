from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.simplefilter("ignore")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        with open('calibrated_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        # Parse JSON request
        request_data = request.get_json()



        # Get dataset ID and values from the request
        data = pd.DataFrame([request_data])




        prediction = model.predict_proba(data)


        # Return the prediction in the response body
        response_body = {'prediction': prediction[:,1].tolist()}
        return jsonify(response_body)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
