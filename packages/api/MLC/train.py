import pickle
import gensim
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from MLC.preprocessing import *

def tag_docs_df(df):
	return df.apply(lambda r: TaggedDocument(words=r['text'].split(), tags=r.label), axis=1)

def vec_for_learning(model, tagged_docs):
	sents = tagged_docs.values
	targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=inference_steps)) for doc in sents])
	return targets, regressors

def Doc2VecModel(text_df, no_epochs, val_split_ratio):
	## splitting dataframe into training and validation frames
	train_df, val_df = train_test_split(text_df, test_size=val_split_ratio)	
	## creating tagged documents
	train_tagged = tag_docs_df(train_df)
	val_tagged = tag_docs_df(train_df)
	## building a distributed bag of words model 
	cores = multiprocessing.cpu_count()
	print("Building the Doc2Vec model vocab...")
	model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=2, workers=cores)
	model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
	## training the model
	print("Training the Doc2Vec model for ", no_epochs, "number of epochs" )
	for epoch in range(no_epochs):
		model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), 
				total_examples=len(train_tagged.values), epochs=1)
	## preparing document vectors for learning
	y_train, X_train = vec_for_learning(model_dbow, train_tagged)
	y_val, X_val = vec_for_learning(model_dbow, val_tagged)
	## training a logistic regression model
	print("Training the logistic regression model...")
	logreg = LogisticRegression(solver='lbfgs', multi_class='auto')
	logreg.fit(X_train, y_train)
	## making predictions on the training set
	print("Prediction numbers:")
	train_binary = logreg.predict(X_train)
	print('Accuracy on the training set : %s' % accuracy_score(y_train, train_binary))
	print('F1 score on the training set : {}'.format(f1_score(y_train, train_binary, average='weighted')))
	## making predictions on the validation set
	val_binary = logreg.predict(X_val)
	print('Accuracy on the validation set : %s' % accuracy_score(y_val, val_binary))
	print('F1 score on the validation set : {}'.format(f1_score(y_val, val_binary, average='weighted')))
	return model_dbow, logreg

def train(data_file, label_file, no_epochs, val_split_ratio):
	## preparing training data
	data_df = pd.read_csv(data_file)
	label_df = pd.read_csv(label_file)
	text_df = prepare_training_data(data_file, label_file)
	## building the document vector model
	model_dbow, logreg = Doc2VecModel(text_df, no_epochs, val_split_ratio)
	print("Pickling the models...")
	with open('models/dbow_model.pkl','wb') as file:
		pickle.dump(model_dbow, file)
	with open('models/log_reg_model.pkl','wb') as file:
		pickle.dump(logreg, file)	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file",
						required=True,
						type=str,
	                    help="enter data file path")
	parser.add_argument("--label_file", 
						required=True,
						type=str,
	                    help="enter label file path")
	parser.add_argument("--epochs",
						required=False,
						type=int,
						default=60,
	                    help="enter number of epochs for model training")
	parser.add_argument("--train_val_split", 
						required=False,
						type=float,
						default=0.2,
	                    help="enter train val split")

	args = parser.parse_args()
	if args.train_val_split > 1 or args.train_val_split < 0:
		raise Exception('invalid test-val split. number must be in the range (0, 1]')
	if args.epochs < 0:
		raise Exception('Epochs must be greater than 0')

	train(args.data_file, args.label_file, args.epochs, args.train_val_split)
