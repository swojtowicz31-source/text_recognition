# Text Classification with Naive Bayes and TF-IDF

This project implements a simple text classification pipeline using the Multinomial Naive Bayes algorithm combined with TF-IDF vectorization.

## Features

- Train a model on labeled text data  
- Perform predictions on new data  
- Evaluate model performance with classification report and accuracy  
- Cross-validate the model with 5-fold CV  

## Usage

The script supports three modes:

- `train`: Train the model and save it to a file  
- `predict`: Load a saved model and predict labels for input data  
- `crossval`: Perform 5-fold cross-validation on the dataset  

## Command line arguments

- `--mode`: Operation mode (`train`, `predict`, or `crossval`) **(required)**  
- `--input`: Path to input text file (format: `<label>\t<text>` per line)  
- `--model`: Path to save/load the model file (default: `brexit_model.joblib`)  
- `--output`: Output file to save predictions (optional)  

## Input data format

Each line should contain a label and the corresponding text separated by a tab character, e.g.:

