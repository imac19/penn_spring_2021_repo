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
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from syllables import count_syllables
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from nltk.corpus import wordnet as wn

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

def load_file_test(data_file):
    words = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
            i += 1
    return words

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
    
    return training_performance, development_performance, precisions, recalls

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
    threshold_values = np.linspace(avg_freq+(std_freq*.1), avg_freq+(3*std_freq), 1000)
    
    for i in threshold_values:
        
        train_y_pred = frequency_threshold_feature(train_words, i, counts)
        f = get_fscore(train_y_pred, train_y_true)
        precisions.append(get_precision(train_y_pred, train_y_true))
        recalls.append(get_recall(train_y_pred, train_y_true))
        
        if f>baseline_f:
            baseline_f = f
            best_threshold = i
#            print('Word Frequency Threshold: {}, Fscore: {}'.format(i, f))
    
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

    return training_performance, development_performance, precisions, recalls

### 2.4: Naive Bayes
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    
    train_words, train_y_true = load_file(training_file)
    dev_words, dev_y_true = load_file(development_file)
    
    train_X = np.empty((len(train_words), 2))
    train_y = np.array(train_y_true)
    dev_X = np.empty((len(dev_words),2))
    
    word_lens = [len(word) for word in train_words]
    word_freqs = [counts[word] for word in train_words]
    len_mean = np.mean(word_lens)
    len_std = np.std(word_lens)
    freq_mean = np.mean(word_freqs)
    freq_std = np.std(word_freqs)
    
    for i in range(0, train_X.shape[0]):
        train_X[i,0] = (len(train_words[i]) - len_mean)/len_std
        train_X[i,1] = (counts[train_words[i]] - freq_mean)/freq_std
    
    for j in range(0, dev_X.shape[0]):
        dev_X[j,0] = (len(dev_words[j]) - len_mean)/len_std
        dev_X[j,1] = (counts[dev_words[j]] - freq_mean)/freq_std
        
    clf = GaussianNB()
    clf.fit(train_X, train_y)
    train_y_pred = clf.predict(train_X)
    dev_y_pred = clf.predict(dev_X)
    
    tprecision = get_precision(train_y_pred, train_y_true)
    trecall = get_recall(train_y_pred, train_y_true)
    tfscore = get_fscore(train_y_pred, train_y_true)
    
    dprecision = get_precision(dev_y_pred, dev_y_true)
    drecall = get_recall(dev_y_pred, dev_y_true)
    dfscore = get_fscore(dev_y_pred, dev_y_true)
    
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    
    print('Training Performance:')
    test_predictions(training_performance)
    print()
    print('Development Performance:')
    test_predictions(development_performance)
    
    return development_performance

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    
    train_words, train_y_true = load_file(training_file)
    dev_words, dev_y_true = load_file(development_file)
    
    train_X = np.empty((len(train_words), 2))
    train_y = np.array(train_y_true)
    dev_X = np.empty((len(dev_words),2))
    
    word_lens = [len(word) for word in train_words]
    word_freqs = [counts[word] for word in train_words]
    len_mean = np.mean(word_lens)
    len_std = np.std(word_lens)
    freq_mean = np.mean(word_freqs)
    freq_std = np.std(word_freqs)
    
    for i in range(0, train_X.shape[0]):
        train_X[i,0] = (len(train_words[i]) - len_mean)/len_std
        train_X[i,1] = (counts[train_words[i]] - freq_mean)/freq_std
    
    for j in range(0, dev_X.shape[0]):
        dev_X[j,0] = (len(dev_words[j]) - len_mean)/len_std
        dev_X[j,1] = (counts[dev_words[j]] - freq_mean)/freq_std
        
    clf = LogisticRegression()
    clf.fit(train_X, train_y)
    train_y_pred = clf.predict(train_X)
    dev_y_pred = clf.predict(dev_X)
    
    tprecision = get_precision(train_y_pred, train_y_true)
    trecall = get_recall(train_y_pred, train_y_true)
    tfscore = get_fscore(train_y_pred, train_y_true)
    
    dprecision = get_precision(dev_y_pred, dev_y_true)
    drecall = get_recall(dev_y_pred, dev_y_true)
    dfscore = get_fscore(dev_y_pred, dev_y_true)
    
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    
    print('Training Performance:')
    test_predictions(training_performance)
    print()
    print('Development Performance:')
    test_predictions(development_performance)
    
    return development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE

def letter_freq_stats(train_words, letter_freq_dict):
    word_lens = [len(word) for word in train_words]
    word_lens = np.unique(word_lens)
    len_stats = {}
     
    for length in word_lens:
        vals = []
        for word in train_words:
            if(len(word) == length):
                prob = 1
                for char in word:
                    if char in letter_freq_dict.keys():
                        prob *= letter_freq_dict[char]
                vals.append(prob)
        len_stats[length] = [np.mean(vals), np.std(vals)]
         
    return len_stats

def ians_classifier(training_file, development_file, counts, test_file, classifier, best=0):
    
    train_words, train_y_true = load_file(training_file)
    dev_words, dev_y_true = load_file(development_file)
    test_words = load_file_test(test_file)
    
    train_X = np.empty((len(train_words), 9))
    train_y = np.array(train_y_true)
    dev_X = np.empty((len(dev_words), 9))
    dev_y = np.array(dev_y_true)
    test_X = np.empty((len(test_words), 9))
    
    word_lens = [len(word) for word in train_words]
    word_freqs = [counts[word] for word in train_words]
    word_syls = [count_syllables(word) for word in train_words]
    len_mean = np.mean(word_lens)
    len_std = np.std(word_lens)
    freq_mean = np.mean(word_freqs)
    freq_std = np.std(word_freqs)
    syls_mean = np.mean(word_syls)
    syls_std = np.std(word_syls)
    
    letter_freq_dict = {'a': .082, 'b': .015, 'c': .028, 'd': .043, 'e': .13,
                        'f': .022, 'g': .02, 'h': .061, 'i': .07, 'j': .0015,
                        'k': .0077, 'l': .04, 'm': .024, 'n': .067, 'o': .075,
                        'p': .019, 'q': .00095, 'r': .06, 's':.063, 't': .091,
                        'u': .028, 'v': .0098, 'w': .024, 'x': .0015, 'y': .02,
                        'z': .00074}
    letter_stats_dict = letter_freq_stats(train_words, letter_freq_dict)
    
    # n, v, s, a, r  
    
    for i in range(0, train_X.shape[0]):
        train_X[i,0] = (len(train_words[i]) - len_mean)/len_std
        train_X[i,1] = (counts[train_words[i]] - freq_mean)/freq_std
        prob = 1
        for char in train_words[i]:
            if char in letter_freq_dict.keys():
                prob *= letter_freq_dict[char]
        train_X[i,2] = (prob - letter_stats_dict[len(train_words[i])][0])/letter_stats_dict[len(train_words[i])][1]
        train_X[i,3] = (count_syllables(train_words[i]) - syls_mean)/syls_std
        pos_tags = []
        for x in wn.synsets(train_words[i]):
            pos_tags.append(x.name().split('.')[1])
        if 'n' in pos_tags:
            train_X[i,4] = 1
        else:
            train_X[i,4] = 0
        if 'v' in pos_tags:
            train_X[i,5] = 1
        else:
            train_X[i,5] = 0
        if 's' in pos_tags:
            train_X[i,6] = 1
        else:
            train_X[i,6] = 0
        if 'a' in pos_tags:
            train_X[i,7] = 1
        else:
            train_X[i,7] = 0
        if 'r' in pos_tags:
            train_X[i,8] = 1
        else:
            train_X[i,8] = 0
    
    for j in range(0, dev_X.shape[0]):
        dev_X[j,0] = (len(dev_words[j]) - len_mean)/len_std
        dev_X[j,1] = (counts[dev_words[j]] - freq_mean)/freq_std
        prob = 1
        for char in dev_words[j]:
            if char in letter_freq_dict.keys():
                prob *= letter_freq_dict[char]
        dev_X[j,2] = (prob - letter_stats_dict[len(dev_words[j])][0])/letter_stats_dict[len(dev_words[j])][1]
        dev_X[j,3] = (count_syllables(dev_words[j]) - syls_mean)/syls_std
        pos_tags = []
        for x in wn.synsets(dev_words[j]):
            pos_tags.append(x.name().split('.')[1])
        if 'n' in pos_tags:
            dev_X[j,4] = 1
        else:
            dev_X[j,4] = 0
        if 'v' in pos_tags:
            dev_X[j,5] = 1
        else:
            dev_X[j,5] = 0
        if 's' in pos_tags:
            dev_X[j,6] = 1
        else:
            dev_X[j,6] = 0
        if 'a' in pos_tags:
            dev_X[j,7] = 1
        else:
            dev_X[j,7] = 0
        if 'r' in pos_tags:
            dev_X[j,8] = 1
        else:
            dev_X[j,8] = 0
        
    for k in range(0, test_X.shape[0]):
        test_X[k,0] = (len(test_words[k]) - len_mean)/len_std
        test_X[k,1] = (counts[test_words[k]] - freq_mean)/freq_std
        prob = 1
        for char in test_words[k]:
            if char in letter_freq_dict.keys():
                prob *= letter_freq_dict[char]
        test_X[k,2] = (prob - letter_stats_dict[len(test_words[k])][0])/letter_stats_dict[len(test_words[k])][1]
        test_X[k,3] = (count_syllables(test_words[k]) - syls_mean)/syls_std
        pos_tags = []
        for x in wn.synsets(test_words[k]):
            pos_tags.append(x.name().split('.')[1])
        if 'n' in pos_tags:
            test_X[k,4] = 1
        else:
            test_X[k,4] = 0
        if 'v' in pos_tags:
            test_X[k,5] = 1
        else:
            test_X[k,5] = 0
        if 's' in pos_tags:
            test_X[k,6] = 1
        else:
            test_X[k,6] = 0
        if 'a' in pos_tags:
            test_X[k,7] = 1
        else:
            test_X[k,7] = 0
        if 'r' in pos_tags:
            test_X[k,8] = 1
        else:
            test_X[k,8] = 0

        
    clf = classifier
    clf.fit(train_X, train_y)
    train_y_pred = clf.predict(train_X)
    dev_y_pred = clf.predict(dev_X)
    
    tprecision = get_precision(train_y_pred, train_y_true)
    trecall = get_recall(train_y_pred, train_y_true)
    tfscore = get_fscore(train_y_pred, train_y_true)
    
    dprecision = get_precision(dev_y_pred, dev_y_true)
    drecall = get_recall(dev_y_pred, dev_y_true)
    dfscore = get_fscore(dev_y_pred, dev_y_true)
    
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    
    print('Training Performance:')
    test_predictions(training_performance)
    print()
    print('Development Performance:')
    test_predictions(development_performance)
    
    if best==1:
    
        clf = classifier
        clf.fit(np.concatenate((train_X, dev_X)), np.concatenate((train_y, dev_y)))
        test_preds = clf.predict(test_X)
        
        file = open('test_labels.txt', 'w')
        for label in test_preds:
            file.write('{}\n'.format(label))
        file.close()
        
        for i in range(0, len(dev_y_pred)):
            
            print('Word: {}, Prediction: {}, Actual: {}'.format(dev_words[i], dev_y_pred[i], dev_y_true[i]))
    
    return development_performance


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
    a = word_length_threshold(training_file, development_file)
    length_precisions = a[2]
    length_recalls = a [3]
    print()
    print('Word Frequency Baseline:')
    b = word_frequency_threshold(training_file, development_file, counts)
    freq_precisions = b[2]
    freq_recalls = b[3]
    print()
    print('Naive Bayes Classifier:')
    naive_bayes(training_file, development_file, counts)
    print()
    print('Logistic Regression:')
    logistic_regression(training_file, development_file, counts)
    print()
    print('Ians Classifier (Random Forest):')
    ians_classifier(training_file, development_file, counts, test_file, RandomForestClassifier(max_depth=2, n_estimators=200))
    print()
    print('Ians Classifier (AdaBoost):')
    ians_classifier(training_file, development_file, counts, test_file, AdaBoostClassifier(), best=1)
    print()
    print('Ians Classifier (SVM):')
    ians_classifier(training_file, development_file, counts, test_file, SVC())
    print()
    
    
    
    print()
    plt.figure()
    plt.plot(length_precisions, length_recalls, color = 'r', label='Word Length')
    plt.plot(freq_precisions, freq_recalls, color = 'b', label='Word Frequencies')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs. Recall Curve For Word Length/Frequencies')
    plt.legend()
    plt.show()
    
# =============================================================================
#     for i in range(len(train_data[0])):
#         print('Word: {}, Complexity: {}'.format(train_data[0][i],train_data[1][i]))
# =============================================================================
    
    
# =============================================================================
# letter_freq_dict = {'a': .082, 'b': .015, 'c': .028, 'd': .043, 'e': .13,
#                         'f': .022, 'g': .02, 'h': .061, 'i': .07, 'j': .0015,
#                         'k': .0077, 'l': .04, 'm': .024, 'n': .067, 'o': .075,
#                         'p': .019, 'q': .00095, 'r': .06, 's':.063, 't': .091,
#                         'u': .028, 'v': .0098, 'w': .024, 'x': .0015, 'y': .02,
#                         'z': .00074}
# 
# 
# word = 'word'
# for char in word:
#     print(letter_freq_dict[char])
# =============================================================================
