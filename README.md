# Naive Bayes Classifier to Filter Spam Emails

This repository contains a project implementing a Naive Bayes Classifier to filter spam emails. The classifier is designed to distinguish between spam and non-spam (ham) emails using the probabilistic Naive Bayes approach. This project provides a simple yet powerful introduction to machine learning and email filtering.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Results](#results)
- [Limitations](#limitations)

## Introduction

Email spam filtering is one of the classic problems in machine learning and data classification. This project focuses on building a Naive Bayes classifier to filter spam emails. The goal is to classify an email as either "spam" or "ham" based on its content.

The Naive Bayes approach is based on Bayes' Theorem, which provides a way to calculate the probability of a label given some evidence. It is called "naive" because it assumes that all features are independent, which is rarely true in practice but still works well for many problems.

## Features

- Implements a probabilistic Naive Bayes classifier.
- Preprocesses email data (e.g., tokenization, stop-word removal).
- Trains the model using labeled email datasets.
- Evaluates the model's accuracy on test data.
- Visualizes results and provides performance metrics.

## Installation

To use this project, you need Python and Jupyter Notebook installed. Follow the steps below to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/LovedeepRajpurohit/Naive-Bayes-Classifier-to-Filter-Spam-Emails.git
   cd Naive-Bayes-Classifier-to-Filter-Spam-Emails
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Navigate to the `.ipynb` file and open it in your browser.

## Usage

1. Ensure the dataset folder contains the spam and ham email files.
2. Run the Jupyter Notebook to preprocess the data, train the model, and evaluate its performance.
3. Customize the code as needed to experiment with different datasets or configurations.

## Dataset

The project uses a labeled dataset containing spam and ham emails. You can use publicly available datasets such as the [Enron Email Dataset](https://www.cs.cmu.edu/~enron/) or the [SpamAssassin Public Corpus](https://spamassassin.apache.org/publiccorpus/).

Ensure that the dataset is organized into two folders:
- `spam/` - Contains spam emails.
- `ham/` - Contains non-spam emails.

Place these folders in the appropriate directory as specified in the code.

## How It Works

1. **Data Preprocessing**:
   - Emails are tokenized into words.
   - Stop words and special characters are removed.
   - Words are converted to lowercase.

2. **Training**:
   - The Naive Bayes classifier is trained using the frequency of words in the `spam` and `ham` datasets.
   - Probabilities for each word being spam or ham are calculated.

3. **Prediction**:
   - For a given email, the probabilities of it being spam or ham are computed.
   - The label with the highest probability is assigned to the email.

4. **Evaluation**:
   - Accuracy, precision, recall, and F1-score are calculated on the test dataset.

## Results

The Naive Bayes classifier achieves high accuracy in filtering spam emails. The exact performance metrics may vary depending on the dataset used. Example metrics include:

- **Accuracy**: ~95%
- **Precision**: ~92%
- **Recall**: ~91%
- **F1-Score**: ~91%

Performance visualization is available in the Jupyter Notebook.

## Limitations

- **Independence Assumption**: The assumption that all features are independent may not hold in practice.
- **Text Preprocessing**: The quality of predictions heavily depends on the preprocessing of the data.
- **Dataset Quality**: The classifier's performance is as good as the dataset used for training and testing.

---
Happy coding!
