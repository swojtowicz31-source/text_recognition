import os
import re
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r"(-?1|0)\t(.+)", line.strip())
            if match:
                labels.append(int(match.group(1)))
                data.append(match.group(2))
    return data, labels


def train_model(data, labels):
    pipeline = Pipeline([ 
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), 
        ('clf', MultinomialNB()) 
    ])
    pipeline.fit(data, labels)
    return pipeline


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def evaluate_model(model, data, labels):
    preds = model.predict(data)
    print("\nClassification Report:\n", classification_report(labels, preds))
    print("Accuracy:", accuracy_score(labels, preds))


def cross_validate_model(model, data, labels):
    scores = cross_val_score(model, data, labels, cv=5)
    print("\nCross-validation accuracy scores:", scores)
    print("Mean CV accuracy:", scores.mean())


def predict_model(model, data, output=None):
    preds = model.predict(data)
    if output:
        with open(output, 'w') as f:
            for p in preds:
                f.write(f"{p}\n")
    else:
        for p in preds:
            print(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'crossval'], required=True)
    parser.add_argument('--input', help='Input file path')
    parser.add_argument('--model', default='brexit_model.joblib', help='Model file path')
    parser.add_argument('--output', help='Output file for predictions')
    args = parser.parse_args()

    if args.mode == 'train':
        data, labels = load_data(args.input)
        model = train_model(data, labels)
        save_model(model, args.model)
        print("\nModel trained and saved.")
        evaluate_model(model, data, labels)

    elif args.mode == 'predict':
        model = load_model(args.model)
        data, _ = load_data(args.input)
        print("\nPrediction results:")
        predict_model(model, data, args.output)
        
        

    elif args.mode == 'crossval':
        data, labels = load_data(args.input)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', MultinomialNB())
        ])
        print("\nCross-validation results:")
        cross_validate_model(pipeline, data, labels)


if __name__ == '__main__':
    main()
