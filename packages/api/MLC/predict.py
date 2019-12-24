import os
import pickle
import argparse
import gensim

this_dir, this_filename = os.path.split(__file__)
DBOW_PATH = os.path.join(this_dir, "models", "dbow_model.pkl")
LOGREG_PATH = os.path.join(this_dir, "models", "log_reg_model.pkl")

with open(DBOW_PATH,'rb') as file:
	model_dbow = pickle.load(file)
with open(LOGREG_PATH,'rb') as file:
	logreg_model = pickle.load(file)

def predict_top_n(test_text, inference_steps=40, top_n=5):
	regressor = model_dbow.infer_vector(test_text.split(), steps=inference_steps)
	y_pred = logreg_model.predict_proba(regressor.reshape(1,-1))
	classes = logreg_model.classes_
	proba = dict(zip(classes, y_pred[0]))	
	return sorted(proba.items(), key=lambda x: x[1], reverse=True)[:top_n]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--text",
						required=True,
						type=str,
	                    help="enter text for classification")
	parser.add_argument("--inference_steps",
						required=False,
						type=int,
						default=40,
	                    help="enter number of inference steps")
	parser.add_argument("--top_n", 
						required=False,
						type=float,
						default=5,
	                    help="enter how many top predictions and their scores you want to output")
	args = parser.parse_args()
	print(predict_top_n(args.text, args.inference_steps, args.top_n))
