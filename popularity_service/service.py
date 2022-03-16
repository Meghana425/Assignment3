import numpy as np
from flask import Flask, jsonify, make_response, request, Response
from joblib import load

MODEL_PATH = 'movie_popularity.joblib'

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict() -> Response:
	request_data = request.json
	features = extract_features(request_data)
	model_output = model_predictions(features)
	response_data = jsonify({'predicted_popularity': model_output, 'model_info': str(model)})
	return make_response(response_data)


def extract_features(request_data):
	features = np.array([
		request_data['budget'],
		request_data['revenue'],
		request_data['runtime'],
		request_data['vote_average'],
		request_data['vote_count']],
	)
	return np.array([features])


def model_predictions(features: np.ndarray):
	predictions = float(model.predict(features)[0])
	return predictions


if __name__ == '__main__':
	model = load(MODEL_PATH)
	print(f'loaded model={model}')
	print(f'starting API server')
	app.run(host='0.0.0.0', port=5000)
