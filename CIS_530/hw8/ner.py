from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import pickle 

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

def first_letter_capitalized(word):
    if word.istitle():
        return 1
    else:
        return 0
    
def all_letters_capitalized(word):
    if word.isupper():
        return 1
    else:
        return 0

def has_top_prefix(word, most_common_prefixes_top):
    pre = word[0:4]
    if pre in most_common_prefixes_top:
        return 1 
    else:
        return 0
    
def has_top_suffix(word, most_common_suffixes_top):
    suf = word[-4:]
    if suf in most_common_suffixes_top:
        return 1 
    else:
        return 0
    
def is_single_character(word):
    if len(word) == 1:
        return 1 
    else:
        return 0 
    
def has_number(word):
    for char in word:
        if char.isdigit():
            return 1
    return 0
        
def all_numbers(word):
    if word.isdigit():
        return 1 
    else:
        return 0 
        
def getfeats(word, o, pos_tag, most_common_prefixes_top, most_common_suffixes_top):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    features = [
        (o + 'word', word),
        # TODO: add more features here.
        (o + 'isfirstcapital', first_letter_capitalized(word)),
        (o + 'isallcapital', all_letters_capitalized(word)),
# =============================================================================
#         (o + 'haspre', has_top_prefix(word, most_common_prefixes_top))
# =============================================================================
        (o + 'hassuf', has_top_suffix(word, most_common_suffixes_top)),
# =============================================================================
#         (o + 'wordlen', len(word))
# =============================================================================
# =============================================================================
#         (o + 'wordlenis1', is_single_character(word))
# =============================================================================
# =============================================================================
#         (o + 'allnum', all_numbers(word))
# =============================================================================
    ]
    return features
    

def word2features(sent, i, most_common_prefixes_top, most_common_suffixes_top):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in list(range(-3,3)):
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            pos_tag = sent[i+o][1]
            featlist = getfeats(word, o, pos_tag, most_common_prefixes_top, most_common_suffixes_top)
            features.extend(featlist)
    
    return dict(features)

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))
    
    train_feats = []
    train_labels = []
    
    most_common_prefixes = dict()
    most_common_suffixes = dict()
    num_single_char_named_entities = 0
    num_single_char = 0
    
    for sent in train_sents:
        for i in range(len(sent)): 
            if sent[i][-1] != 'O' and len(sent[i][0])==1:
                num_single_char_named_entities += 1
            if len(sent[i][0])==1:
                num_single_char += 1
            if sent[i][-1] != 'O' and len(sent[i][0])>4:
                word = sent[i][0]
                pre = word[0:4]
                suf = word[-4:]
                if pre in most_common_prefixes.keys():
                    most_common_prefixes[pre] += 1
                else:
                    most_common_prefixes[pre]=1
                    
                if suf in most_common_suffixes.keys():
                    most_common_suffixes[suf] += 1
                else:
                    most_common_suffixes[suf]=1
                    
    most_common_prefixes_top = sorted(most_common_prefixes, key=most_common_prefixes.get, reverse=True)[0:150]
    most_common_suffixes_top = sorted(most_common_suffixes, key=most_common_suffixes.get, reverse=True)[0:150]
    
# =============================================================================
#     print(most_common_prefixes_top)
#     print(most_common_suffixes_top)
# =============================================================================
# =============================================================================
#     print(num_single_char)
#     print(num_single_char_named_entities)
#     print(num_single_char_named_entities/num_single_char)
# =============================================================================
    

    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i,most_common_prefixes_top, most_common_suffixes_top)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    
    # TODO: play with other models
    #model = Perceptron(verbose=1)
    #model = LogisticRegression(verbose=1, max_iter=1000, C=7, solver='liblinear')
    #model = svm.SVC(verbose=1)
    #model = RandomForestClassifier(verbose=1, random_state=19, n_estimators=40, max_features=.5, criterion='entropy')
    #model = AdaBoostClassifier(random_state=19, n_estimators=200)
    #model = XGBClassifier(random_state=19, n_estimators=1000, max_depth=10, eta=.2, objective='multi:softmax', num_classes=9)
    #model.fit(X_train, train_labels)
    
    #pickle.dump(model, open('model', 'wb'))
    
    loaded_model = pickle.load(open('model', 'rb'))
    model=loaded_model
    
    y_pred = model.predict(X_train)
    j = 0
    print("Writing to results_train.txt")
    # format is: word gold pred
    with open("train_results.txt", "w") as out:
        for sent in train_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in dev_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i,most_common_prefixes_top, most_common_suffixes_top)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)
    y_pred_dev = model.predict(X_test)
    dev_labels = test_labels

    j = 0
    print("Writing to results_dev.txt")
    # format is: word gold pred
    with open("dev_results.txt", "w") as out:
        for sent in dev_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")
        
        
    test_feats = []
    test_labels = []
    
    for sent in test_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i,most_common_prefixes_top, most_common_suffixes_top)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results_test.txt")
    # format is: word gold pred
    with open("test_results.txt", "w") as out:
        for sent in test_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")



    y_pred_dev = list(y_pred_dev)
    dev_labels = list(dev_labels)
    
    label_list = list(set(dev_labels))
    label_list.sort()
    print(label_list)
    
    y_pred_dev
    dev_labels
    
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot as plt
    from seaborn import heatmap
    
    cm = confusion_matrix(y_pred_dev, dev_labels, labels=label_list)
    for i in range(0,9):
        cm[i][i]=0
    
    print(cm)
    
    total_errors = sum(sum(cm))
    print(total_errors)

    plt.figure()
    heatmap(cm/2177, annot=False, xticklabels=label_list, yticklabels=label_list)
    plt.title('Confusion Matrix')
    plt.show()
    
    label_order = ['OVR', 'LOC', 'MISC', 'ORG', 'PER']
    precisions = [67.43, 63.20, 43.15, 67.84, 78.69]
    recalls = [70.90, 79.57, 37.53, 68.24, 79.79]
    fs = [69.12, 70.45, 40.14, 68.04, 79.24]
    
    import numpy as np

    barWidth = 0.25
     
    # Set position of bar on X axis
    r1 = np.arange(len(precisions))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
     
    # Make the plot
    plt.bar(r1, precisions, width=barWidth, edgecolor='white', label='Precision')
    plt.bar(r2, recalls, width=barWidth, edgecolor='white', label='Recall')
    plt.bar(r3, fs, width=barWidth, edgecolor='white', label='F1 Score')
     
    # Add xticks on the middle of the group bars
    plt.xlabel('Entity')
    plt.ylabel('Score')
    plt.xticks([r + barWidth for r in range(len(precisions))], label_order)
    plt.title('Model Performance By Entity')
     
    # Create legend & Show graphic
    plt.legend()
    plt.show()
    
    loc_miss = {'ORG':0, 'MISC':0, 'PER':0, 'O':0}
    org_miss = {'LOC':0, 'MISC':0, 'PER':0, 'O':0}
    misc_miss = {'ORG':0, 'LOC':0, 'PER':0, 'O':0}
    o_miss = {'ORG':0, 'MISC':0, 'PER':0, 'LOC':0}
    per_miss = {'ORG':0, 'MISC':0, 'O':0, 'LOC':0}
    
    missed_indices = []
    
    
    for i in range(0, len(y_pred_dev)):
            
        if y_pred_dev[i] in ['O']:
            pred = 'O'
        elif y_pred_dev[i] in ['B-ORG', 'I-ORG']:
            pred = 'ORG'
        elif y_pred_dev[i] in ['B-LOC', 'I-LOC']:
            pred = 'LOC'
        elif y_pred_dev[i] in ['B-MISC', 'I-MISC']:
            pred = 'MISC'
        elif y_pred_dev[i] in ['B-PER', 'I-PER']:
            pred = 'PER'

        if dev_labels[i] in ['O'] and pred!='O':
            o_miss[pred]+=1
            missed_indices.append(i)
        if dev_labels[i] in ['B-ORG', 'I-ORG'] and pred!='ORG':
            org_miss[pred]+=1
            missed_indices.append(i)
        if dev_labels[i] in ['B-LOC', 'I-LOC'] and pred!='LOC':
            loc_miss[pred]+=1
            missed_indices.append(i)
        if dev_labels[i] in ['B-MISC', 'I-MISC'] and pred!='MISC':
            misc_miss[pred]+=1
            missed_indices.append(i)
        if dev_labels[i] in ['B-PER', 'I-PER'] and pred!='PER':
            per_miss[pred]+=1
            missed_indices.append(i)
            
    miss_dicts = [loc_miss, org_miss, misc_miss, o_miss, per_miss]
    
    for d in miss_dicts:
        sum_d = sum(d.values())
        
        for key in d.keys():
            d[key] = round(d[key]/sum_d, 2)
            
    
    print(loc_miss)
    print(org_miss)
    print(misc_miss)
    print(o_miss)
    print(per_miss)
    
    plot_titles = ['Percent of Location Misclassifications by Entity',
                   'Percent of Organization Misclassifications by Entity',
                   'Percent of Miscellaneous Misclassifications by Entity',
                   'Percent of Object Misclassifications by Entity',
                   'Percent of Person Misclassifications by Entity']
    counter=0
    for d in miss_dicts:
        plt.figure()
        plt.title(plot_titles[counter])
        plt.bar(range(len(d)), list(d.values()), align='center')
        plt.xticks(range(len(d)), list(d.keys()))
        plt.xlabel('Misclassified as Entity')
        plt.ylabel('Percent of Misclassifications')
        plt.show()
        counter+=1

    dev_sents_flat = [item for sublist in dev_sents for item in sublist]
    print(dev_sents_flat)
    for index in missed_indices[1000:2000]:
        print('Actual: {}'.format(dev_labels[index]))
        print('Predicted: {}'.format(y_pred_dev[index]))
        print('Word: {}'.format(dev_sents_flat[index][0]))
        print('Index: {}'.format(index))
        print()
    
    