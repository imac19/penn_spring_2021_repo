#############################################################
## ASSIGNMENT 2 CODE SKELETON
## RELEASED: 2/2/2020
## DUE: 2/12/2020
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
import matplotlib.pyplot as plt
import numpy as np

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    
    correct=0
    total=0
    for i in range(0, len(y_pred)):
        if y_pred[i]==1:
            if y_true[i]==1:
                correct+=1
                total+=1
            else:
                total+=1
    
    precision = correct/total

    return precision
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    
    correct = 0
    total = 0
    for i in range(0, len(y_pred)):
        if y_true[i]==1:
            if y_pred[i]==1:
                correct+=1
                total+=1
            else:
                total+=1
                
    recall = correct/total

    return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    
    recall = get_recall(y_pred, y_true)
    
    precision = get_precision(y_pred, y_true)
    
    fscore = (2*precision*recall)/(precision+recall)

    return fscore

## Prints out accuracy metrics

def test_predictions(acc_list):
    
# =============================================================================
#     precision = get_precision(y_pred, y_true)
#     recall = get_precision(y_pred, y_true)
#     fscore = get_fscore(y_pred, y_true)
# =============================================================================

    precision = acc_list[0]
    recall = acc_list[1]
    fscore = acc_list[2]
    
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Fscore: {}".format(fscore))

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

### 2.1: A very simple baseline

## Makes feature matrix for all complex
def all_complex_feature(words):
    return [1] * len(words)

## Labels every word complex
def all_complex(data_file):
    
    words, y_true = load_file(data_file)
    y_pred = all_complex_feature(words)
    
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    performance = [precision, recall, fscore]
    test_predictions(performance)
    
    return performance


### 2.2: Word length thresholding

## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    
    y_pred = []
    
    for i in range (0, len(words)):
        if len(words[i])<threshold:
            y_pred.append(0)
        else:
            y_pred.append(1)
            
    return y_pred

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    
    train_words, train_y_true = load_file(training_file)
    dev_words, dev_y_true = load_file(development_file)
    
    baseline_f = 0
    best_threshold = 0
    precisions = []
    recalls = []
    
    for i in range(3,10):
        
        train_y_pred = length_threshold_feature(train_words, i)
        f = get_fscore(train_y_pred, train_y_true)
        precisions.append(get_precision(train_y_pred, train_y_true))
        recalls.append(get_recall(train_y_pred, train_y_true))
        
        print('Word Length Threshold: {}, Fscore: {}'.format(i, f))
        
        if f>baseline_f:
            baseline_f = f
            best_threshold = i
            
    train_y_pred = length_threshold_feature(train_words, best_threshold)
    dev_y_pred = length_threshold_feature(dev_words, best_threshold)
    
    tprecision = get_precision(train_y_pred, train_y_true)
    trecall = get_recall(train_y_pred, train_y_true)
    tfscore = get_fscore(train_y_pred, train_y_true)
    
    dprecision = get_precision(dev_y_pred, dev_y_true)
    drecall = get_recall(dev_y_pred, dev_y_true)
    dfscore = get_fscore(dev_y_pred, dev_y_true)
    
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    
    print()
    plt.figure()
    plt.plot(precisions, recalls)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs. Recall For Various Word Lengths')
    plt.show()
    
    print('Chosen Word Length Threshold: {}'.format(best_threshold))
    print()
    print('Training Performance:')
    test_predictions(training_performance)
    print()
    print('Development Performance:')
    test_predictions(development_performance)
    
    return training_performance, development_performance

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt', errors='ignore', encoding='utf-8') as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    
    y_pred = []
    
    for i in range(0, len(words)):
        
        if counts[words[i]] >= threshold:
            y_pred.append(0)
        else:
            y_pred.append(1)
            
    return y_pred

def word_frequency_threshold(training_file, development_file, counts):
    
    train_words, train_y_true = load_file(training_file)
    dev_words, dev_y_true = load_file(development_file)
    
    baseline_f = 0
    best_threshold = 0
    precisions = []
    recalls = []
    
    avg_freq = np.mean(list(counts.values()))
    std_freq = np.std(list(counts.values()))
    threshold_values = np.linspace(avg_freq, avg_freq+(std_freq), 1000)
    
    for i in threshold_values:
        
        train_y_pred = frequency_threshold_feature(train_words, i, counts)
        f = get_fscore(train_y_pred, train_y_true)
        precisions.append(get_precision(train_y_pred, train_y_true))
        recalls.append(get_recall(train_y_pred, train_y_true))
        
        if f>baseline_f:
            baseline_f = f
            best_threshold = i
            print('Word Frequency Threshold: {}, Fscore: {}'.format(i, f))
    
    train_y_pred = frequency_threshold_feature(train_words, best_threshold, counts)
    dev_y_pred = frequency_threshold_feature(dev_words, best_threshold, counts)
    
    tprecision = get_precision(train_y_pred, train_y_true)
    trecall = get_recall(train_y_pred, train_y_true)
    tfscore = get_fscore(train_y_pred, train_y_true)
    
    dprecision = get_precision(dev_y_pred, dev_y_true)
    drecall = get_recall(dev_y_pred, dev_y_true)
    dfscore = get_fscore(dev_y_pred, dev_y_true)
    
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    
    print()
    plt.figure()
    plt.plot(precisions, recalls)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs. Recall For Various Word Frequencies')
    plt.show()
    
    print('Chosen Word Frequency Threshold: {}'.format(best_threshold))
    print()
    print('Training Performance:')
    test_predictions(training_performance)
    print()
    print('Development Performance:')
    test_predictions(development_performance)

    return training_performance, development_performance

### 2.4: Naive Bayes
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE    
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
    
    print('All Complex Baseline (Training):')
    all_complex(training_file)
    print('All Complex Baseline (Development):')
    all_complex(development_file)
    print()
    print('Word Length Baseline:')
    word_length_threshold(training_file, development_file)
    print()
    print('Word Frequency Baseline:')
    word_frequency_threshold(training_file, development_file, counts)
    
    
    
    
    
    