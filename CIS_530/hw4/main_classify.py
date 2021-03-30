# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:03:03 2021

@author: imacd_0odruq3
"""

#main_classify.py
import codecs
import math
import random
import string
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score

'''
Don't change these constants for the classification task.
You may use different copies for the sentence generation model.
'''
languages = ["af", "cn", "de", "fi", "fr", "in", "ir", "pk", "za"]
all_letters = string.ascii_letters + " .,;'"
device = 'cpu'
'''
Returns the words of the language specified by reading it from the data folder
Returns the validation data if train is false and the train data otherwise.
Return: A nx1 array containing the words of the specified language
'''
def getWords(baseDir, lang, train = True):  
    if train:
        data_type = 'train'
    else:
        data_type = 'val'

    read_location = '{}/{}/{}.txt'.format(baseDir, data_type, lang)
    word_array = open(read_location, encoding='iso-8859-1').read().strip().split('\n')

    return np.array(word_array)

'''
Returns a label corresponding to the language
For example it returns an array of 0s for af
Return: A nx1 array as integers containing index of the specified language in the "languages" array
'''
def getLabels(lang, length):
    label = languages.index(lang)
    labels = [label] * length

    return np.array(labels)

'''
Returns all the laguages and labels after reading it from the file
Returns the validation data if train is false and the train data otherwise.
You may assume that the files exist in baseDir and have the same names.
Return: X, y where X is nx1 and y is nx1
'''
def readData(baseDir, train=True):
    data = np.empty(0)
    labels = np.empty(0)

    for lang in languages:
        lang_words = getWords(baseDir, lang, train)
        lang_len = len(lang_words)
        data = np.append(data, lang_words)
        labels = np.append(labels, getLabels(lang, lang_len))

    return data, labels

'''
Convert a line/word to a pytorch tensor of numbers
Refer the tutorial in the spec
Return: A tensor corresponding to the given line
'''
def line_to_tensor(line):
    line_tensor = torch.zeros(len(line), 1, len(all_letters))
    for i, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        line_tensor[i][0][letter_index] = 1 
    return line_tensor

'''
Returns the category/class of the output from the neural network
Input: Output of the neural networks (class probabilities)
Return: A tuple with (language, language_index)
        language: "af", "cn", etc.
        language_index: 0, 1, etc.
'''
def category_from_output(output):
    category_index = output.topk(1)[1][0].item()
    return languages[category_index], category_index

'''
Get a random input output pair to be used for training 
Refer the tutorial in the spec
'''
def random_training_pair(X, y):
    rand_index = random.randint(0, X.shape[0]-1)

    # line_tensor = line_to_tensor(X[rand_index])
    # category_tensor = torch.tensor([y[rand_index]], dtype = torch.long)
    # return line_tensor, category_tensor

    line = X[rand_index]
    category = y[rand_index]
    return line, category

'''
Input: trained model, a list of words, a list of class labels as integers
Output: a list of class labels as integers
'''
def predict(model, X, y):
    predictions = []
    for i, data in enumerate(X):
        line_tensor = line_to_tensor(data).to(device)
        with torch.no_grad():
            hidden = model.init_hidden().to(device)

            for j in range(0, line_tensor.shape[0]):
                output, hidden = model(line_tensor[j], hidden)
            
            predicted_class, predicted_class_index = category_from_output(output)

            predictions.append(predicted_class_index)

    return np.array(predictions)
'''
Input: trained model, a list of words, a list of class labels as integers
Output: The accuracy of the given model on the given input X and target y
'''
def calculateAccuracy(model, X, y):
    predicted_class_labels = predict(model, X, y)

    acc = ((predicted_class_labels == y).sum())/len(y)

    return acc

'''
Train the model for one epoch/one training word.
Ensure that it runs within 3 seconds.
Input: X and y are lists of words as strings and classes as integers respectively
Returns: You may return anything
'''
def trainOneEpoch(model, criterion, optimizer, X, y):
    total_loss = 0

    for i, data in enumerate(X):
        start_time = time.time()
        line_tensor = line_to_tensor(data).to(device)
        category_tensor = torch.tensor(y, dtype=torch.long).to(device)

        hidden = model.init_hidden().to(device)
        optimizer.zero_grad()
        # model.zero_grad()

        for j in range(0, line_tensor.shape[0]):
            output, hidden = model(line_tensor[j], hidden)
            
        pred = category_from_output(output)
            
        loss = criterion(output, category_tensor)
        total_loss += loss
        loss.backward()
        optimizer.step()

        # for p in model.parameters():
        #     p.data.add_(p.grad.data, alpha = -learning_rate)

        end_time = time.time()

        if end_time-start_time >= 3:
            print('Runtime Over 3 Seconds')

    return total_loss, pred
'''
Use this to train and save your classification model. 
Save your model with the filename "model_classify"
'''
def run():
    model = CharRNNClassify().to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = .002)
    current_loss_train = 0
    all_losses_train = []
    current_loss_val = 0
    all_losses_val = []
    all_acc_train = []
    all_acc_val = []
    iterations = 10

    X_train, y_train = readData('/content')
    X_val, y_val = readData('/content', train=False)

    for iter in range(1, iterations+1):
        data, label = random_training_pair(X_train, y_train)
        data = np.array([data])
        label = [label]
        loss_train, pred_train = trainOneEpoch(model, criterion, optimizer, data, label)
        current_loss_train += loss_train.item()

        data, label = random_training_pair(X_val, y_val)
        data = np.array([data])
        label = torch.tensor([label], dtype=torch.long).to(device)
        pred_val = torch.tensor(predict(model, data, label)).to(device)
        loss_val = 0
        current_loss_val += loss_val

        if iter % 1000 == 0:
            print(iter)

        if iter % 10000 == 0:
            all_losses_train.append(current_loss_train/10)
            current_loss_train = 0 
            all_losses_val.append(current_loss_val/10)
            current_loss_val = 0 
            all_acc_train.append(calculateAccuracy(model, X_train, y_train))
            all_acc_val.append(calculateAccuracy(model, X_val, y_val))


    torch.save(model.state_dict(), '/content/model_ian.pt', _use_new_zipfile_serialization=False)
