import json
import pytest

import sys
sys.path.append('..')
from api.app import create_app

@pytest.fixture(scope='module')
def test_client():
	flask_app = create_app(config_name='test')
	testing_client = flask_app.test_client()
	ctx = flask_app.app_context()
	ctx.push() 
	yield testing_client
	ctx.pop()

def test_health_get_request(test_client):
	response = test_client.get('/health')
	assert response.status_code == 200

def test_post_request_no_header_no_payload(test_client):
	response = test_client.post('/prediction')
	assert response.status_code == 400

def test_post_request_necessary_param_missing(test_client):
	head = {
		"Content-Type": "application/json"
		}
	data = {
		"top_n": 5, 
		"steps": 20
		}
	data = json.dumps(data)
	response = test_client.post('/prediction', data=data, headers=head)
	assert response.status_code == 400

@pytest.mark.filterwarnings("ignore: logreg")
def test_post_request_necessary_param(test_client):
	head = {
		"Content-Type": "application/json"
		}
	data = {
		"text": "sample customer review"
		}
	data = json.dumps(data)
	response = test_client.post('/prediction', data=data, headers=head)
	assert response.status_code == 200

def test_post_request_all_params(test_client):
	head = {
		"Content-Type": "application/json"
		}
	data = {
		"text": "sample customer review",
		"top_n": 3, 
		"steps": 10
		}
	data = json.dumps(data)
	response = test_client.post('/prediction', data=data, headers=head)
	assert response.status_code == 200

if __name__ == '__main__':
    test_client.run()