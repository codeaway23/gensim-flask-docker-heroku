import json
import pickle

from flask import Blueprint, request, jsonify

from api.MLC.predict import predict_top_n 

prediction_app = Blueprint('prediction_app', __name__)

@prediction_app.route('/health', methods = ['GET'])
def check_health():
	if request.method == 'GET':
		response = {
			'status': 200,
			'message': 'Success'
		}
		return jsonify(response), 200

@prediction_app.route('/prediction', methods = ['POST'])
def predict():
	if request.method == 'POST':
		if not request.data:
			response = {
				'status': 400,
				'message': 'Bad request. Missing payload.'
				}
			return jsonify(response), 400

		if not request.is_json:
			response = {
				'status': 400,
				'message': 'Bad request. Only JSON payload is valid.'
				}
			return jsonify(response), 400

		data = request.get_json()
		
		if not 'text' in data:
			response = {
				'status': 400,
				'message': 'Bad request. Payload missing required parameters.'
				}
			return jsonify(response), 400

		text = data['text']

		if 'steps' in data:
			steps = data['steps']
		else:
			steps = 40

		if not 'top_n' in data:
			top_n = 5
		else:
			top_n = data['top_n']

		proba = predict_top_n(text, steps, top_n)
		response = {
			'status': 200,
			'message': 'Success',
			'text': text,
			'top_n': top_n,
			'inference_steps': steps,
			'prediction': proba
			}
		return jsonify(response), 200

if __name__ == '__main__':
    prediction_app.run(debug=True)
