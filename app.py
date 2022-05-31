from flask import Flask, Response, request, jsonify
from flask_restful import reqparse, abort, Api, Resource
from xgboost import XGBClassifier
import pickle
import numpy as np

app = Flask(__name__)
api = Api(app)

model = XGBClassifier()

# load fitted model
clf_path = 'XGBClassifier.pkl'
with open(clf_path, 'rb') as f:
	model= pickle.load(f)


@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_values = np.array(list(data.values())).astype(float)
    prediction = model.predict(data_values.reshape(1,-1))
    
    # create JSON object
    output = str(prediction[0])

   # output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
