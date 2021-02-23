import math, random
import numpy as np
import os

#
# UPDATED : 2/16/21
#

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(c):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * c

def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    text_padded = start_pad(c) + text
    ngrams = []
    
    for i in range(0, len(text_padded)-c):
        context = text_padded[i:i+c]
        char = text_padded[i+c]
        ngrams.append((context, char))
    
    return ngrams
    

def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

def create_models(model_class, path, c_param, k_param, lambdas):
    models = []
    path = os.getcwd() + '/cities_train/train'
    for file in os.listdir(path):
        filepath = path + '/' + file
        m=create_ngram_model_lines(model_class, filepath, c_param, k_param)
        m.change_lambdas(lambdas)
        models.append(m)
        
    return models

def get_prediction(models, city):
    city_pred_index = None
    city_perplexity = float('inf')
    COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']
    
    for i in range(0, len(models)):
        if models[i].perplexity(city) < city_perplexity:
            city_perplexity = models[i].perplexity(city)
            city_pred_index = i
    
    return COUNTRY_CODES[city_pred_index]

def models_accuracy(actual, predicted):
    totals = [0,0,0,0,0,0,0,0,0]
    correct = [0,0,0,0,0,0,0,0,0]
    COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']
    
    for i in range(0, len(actual)):
        ind = COUNTRY_CODES.index(actual[i])
        totals[ind] += 1
        if actual[i] == predicted[i]:
            correct[ind] += 1
            
    total_accuracy = sum(correct)/sum(totals)
    accuracy_by_country = []
    for j in range(0, len(COUNTRY_CODES)):
        country_acc = correct[j]/totals[j]
        accuracy_by_country.append(country_acc)


        print('Accuracy for {}: {}'.format(COUNTRY_CODES[j], country_acc))
    print()
    print('Overall Accuracy: {}'.format(total_accuracy))


    
    return total_accuracy

    
    

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        self.c = c
        self.k = k
        self.vocab = set()
        self.ngrams = dict()

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        ngram_list = ngrams(self.c, text)
        for context, char in ngram_list:
            self.vocab.add(char)
            if context in self.ngrams.keys():
                if char in self.ngrams[context].keys():
                    self.ngrams[context][char] +=1
                else:
                    self.ngrams[context][char] = 1
            else:
                self.ngrams[context] = dict()
                self.ngrams[context][char] = 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        
        if context not in self.ngrams.keys():
            return 1.0/len(self.vocab)
        else:
            total = sum(self.ngrams[context].values())
            try:
                return (self.ngrams[context][char] + self.k)/(total + len(self.vocab)*self.k)
            except:
                return (self.k)/(total + len(self.vocab)*self.k)

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        rand_num = random.random()
        cumulative_prob = 0
        for char in sorted(self.vocab):
            cumulative_prob += self.prob(context, char)
            if cumulative_prob >= rand_num:
                return char

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        return_string = ''
        context = self.c * '~'
        for i in range(0, length):
            char = self.random_char(context)
            return_string += char
            context = context[1:] + char
            
        return return_string

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''

        p = 0
        padded_text = start_pad(self.c) + text
        for i in range(0, len(text)):
            context = padded_text[i:i+self.c]
            char = text[i]
            x = self.prob(context, char)
            if x>0:
                p += np.log(1/x)
            else:
                return np.float('inf')
        p = p * (1/len(text))
        p = np.exp(p) 
        
        return p

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''
    

    def __init__(self, c, k):
        self.c = c
        self.k = k
        self.vocab = set()
        self.ngrams = dict()
        self.lambdas = []
        for i in range(0, c+1):
            self.lambdas.append(1/(c+1))

    def get_vocab(self):
        return self.vocab

    def update(self, text):
        for i in range(0, self.c+1):
            ngram_list = ngrams(i, text)
            for context, char in ngram_list:
                self.vocab.add(char)
                if context in self.ngrams.keys():
                    if char in self.ngrams[context].keys():
                        self.ngrams[context][char] +=1
                    else:
                        self.ngrams[context][char] = 1
                else:
                    self.ngrams[context] = dict()
                    self.ngrams[context][char] = 1

    def prob(self, context, char):
        
        interpolation_prob = 0
        usage_increment = 0

        for lamb in self.lambdas:
            
            context_check = context[usage_increment:self.c]
        
            if context_check not in self.ngrams.keys():
                interpolation_prob +=  (1.0/len(self.vocab)) * lamb
            else:
                total = sum(self.ngrams[context_check].values())
                try:
                    interpolation_prob += ((self.ngrams[context_check][char] 
                                            + self.k)/(total + 
                                                       len(self.vocab)*self.k)) * lamb
                except:
                    interpolation_prob += ((self.k)/(total + len(self.vocab)*self.k)) * lamb
            usage_increment += 1
            
        return interpolation_prob
    
    def change_lambdas(self, lambdas_change):
        self.lambdas = lambdas_change

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    
    random.seed(19)
    
    os.chdir('C:/Users/imacd_0odruq3/Documents/Penn_Spring_2021_Semester/penn_spring_2021_repo/CIS_530/hw3')

# =============================================================================
#     print('Shakespeare Generated Text: c=2, k=0')
#     m = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=2, k=0)
#     print(m.random_text(250))
#     print()
# 
#     print('Shakespeare Generated Text: c=3, k=0')
#     m = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=3, k=0)
#     print(m.random_text(250))
#     print()
# 
#     print('Shakespeare Generated Text: c=4, k=0')
#     m = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=4, k=0)
#     print(m.random_text(250))
#     print()
#     
#     print('Shakespeare Generated Text: c=5, k=0')
#     m = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=5, k=0)
#     print(m.random_text(250))
#     print()
#     
#     print('Shakespeare Generated Text: c=6, k=0')
#     m = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=6, k=0)
#     print(m.random_text(250))
#     print()
#     
#     print('Shakespeare Generated Text: c=7, k=0')
#     m = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=7, k=0)
#     print(m.random_text(250))
#     print()
# =============================================================================

       
# =============================================================================
#     path = os.getcwd() + '/test_data/shakespeare_sonnets.txt'
#     f = open(path)
#     shakespeare_sonnets = f.read()
#     f.close()
# #    shakespeare_sonnets = shakespeare_sonnets.replace('\n\n', ' ').replace('\n', ' ').replace('   ', ' ').replace('  ', ' ')
#     
#     path = os.getcwd() + '/test_data/nytimes_article.txt'
#     f = open(path)
#     ny_times = f.read()
#     f.close()
# #    ny_times = ny_times.replace('\n\n', ' ').replace('\n', ' ')
#     
# #    print(shakespeare_sonnets)
# #    print(ny_times)
# =============================================================================


# =============================================================================
#     for k_param in [.1, .5, 1, 2]:
#         m = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=4, k=k_param)
#         print('Perplexity (Uninterpolated, c=4, k={}) on Shakespeare Sonnets:'.format(k_param))
#         print(m.perplexity(shakespeare_sonnets))
#         print()
#         print('Perplexity (Uninterpolated, c=4, k={}) on NY Times Article:'.format(k_param))
#         print(m.perplexity(ny_times))
#         print()
#         m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', c=4, k=k_param)
#         print('Perplexity (Interpolated, c=4, k={}) on Shakespeare Sonnets:'.format(k_param)) 
#         print(m.perplexity(shakespeare_sonnets))
#         print()
#         print('Perplexity (Interpolated, c=4, k={}) on NY Times Article:'.format(k_param))
#         print(m.perplexity(ny_times))           
#         print()
# 
#     for c_param in [2,3,4,5,6,7]:
#         m = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=c_param, k=.1)
#         print('Perplexity (Uninterpolated, c={}, k=.1) on Shakespeare Sonnets:'.format(c_param))
#         print(m.perplexity(shakespeare_sonnets))
#         print()
#         print('Perplexity (Uninterpolated, c={}, k=.1) on NY Times Article:'.format(c_param))
#         print(m.perplexity(ny_times))
#         print()
#         m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', c=c_param, k=.1)
#         print('Perplexity (Interpolated, c={}, k=.1) on Shakespeare Sonnets:'.format(c_param)) 
#         print(m.perplexity(shakespeare_sonnets))
#         print()
#         print('Perplexity (Interpolated, c={}, k=.1) on NY Times Article:'.format(c_param))
#         print(m.perplexity(ny_times))           
#         print()     
# 
#     for lambda_list in [[.25, .25, .25, .25], [.1, .2, .3, .4], [.4, .3, .2, .1]]:
#         m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', c=4, k=.1)
#         m.change_lambdas(lambda_list)
#         print('Perplexity (Interpolated, c=4, k=.1) on Shakespeare Sonnets:') 
#         print('Lambdas: {}'.format(lambda_list))
#         print(m.perplexity(shakespeare_sonnets))
#         print()
#         print('Perplexity (Interpolated, c=4, k=.1) on NY Times Article:')
#         print('Lambdas: {}'.format(lambda_list))
#         print(m.perplexity(ny_times))           
#         print()               
# =============================================================================


    
    val_data = dict()
    path = os.getcwd() + '/cities_val/val'
    for file in os.listdir(path):
        l = []
        f = open(path + '/' + file, encoding='ISO-8859-1')
        for line in f:
            l.append(line[0:-1])
        val_data[file] = l
        f.close()
    

# COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']
    
    path = os.getcwd()
    
    #Uninterpolated

# =============================================================================
#     c_vals = [2,3,4,5,6,7]
#     k_vals = [.1, .5, 1, 2]
#     best_c = None
#     best_k = None
#     best_acc = 0
#     
#     for c_val in c_vals:
#         for k_val in k_vals:
#             actual = []
#             predicted = []
#             models = create_models(NgramModel, path, c_val, k_val)
#     
#             for key, cities in val_data.items():
#                 for city in cities:
#                     actual.append(key[0:2])
#                     pred = get_prediction(models, city)
#                     predicted.append(pred)
# 
#             print('Accuracy: {}'.format(models_accuracy(actual, predicted)))
#             print('c: {}'.format(c_val))
#             print('k: {}'.format(k_val))
#             print()
#             if models_accuracy(actual, predicted) > best_acc:
#                 best_acc = models_accuracy(actual, predicted)
#                 best_c = c_val
#                 best_k = k_val
#     
#     print('Overall Best Accuracy: {}'.format(best_acc))
#     print('Overall Best c: {}'.format(best_c))
#     print('Overall Best k: {}'.format(best_k))
# =============================================================================
    
    #Interpolated
    
# =============================================================================
#     c_vals = [2,3]
#     k_vals = [.1,.5]
#     lambdas_two = [[.9, .1], [.75, .25], [.6, .4]]
#     lambdas_three = [[.8, .15, .05], [.7, .2, .1], [.5, .3, .2]]
#     best_c = None
#     best_k = None
#     best_lambdas = None
#     best_acc = 0
#     
#     for c_val in c_vals:
#         for k_val in k_vals:
#             if c_val==2:
#                 for lambda_val in lambdas_two:
#                     actual = []
#                     predicted = []
#                     models = create_models(NgramModelWithInterpolation, path, c_val, k_val, lambda_val)
#                     
#             
#                     for key, cities in val_data.items():
#                         for city in cities:
#                             actual.append(key[0:2])
#                             pred = get_prediction(models, city)
#                             predicted.append(pred)
#         
#                     print('Accuracy: {}'.format(models_accuracy(actual, predicted)))
#                     print('c: {}'.format(c_val))
#                     print('k: {}'.format(k_val))
#                     print('lambda: {}'.format(lambda_val))
#                     print()
#                     if models_accuracy(actual, predicted) > best_acc:
#                         best_acc = models_accuracy(actual, predicted)
#                         best_c = c_val
#                         best_k = k_val
#                         best_lambda = lambda_val
#                         
#             if c_val==3:
#                 for lambda_val in lambdas_three:
#                     actual = []
#                     predicted = []
#                     models = create_models(NgramModelWithInterpolation, path, c_val, k_val, lambda_val)
#                     
#             
#                     for key, cities in val_data.items():
#                         for city in cities:
#                             actual.append(key[0:2])
#                             pred = get_prediction(models, city)
#                             predicted.append(pred)
#         
#                     print('Accuracy: {}'.format(models_accuracy(actual, predicted)))
#                     print('c: {}'.format(c_val))
#                     print('k: {}'.format(k_val))
#                     print('lambda: {}'.format(lambda_val))
#                     print()
#                     if models_accuracy(actual, predicted) > best_acc:
#                         best_acc = models_accuracy(actual, predicted)
#                         best_c = c_val
#                         best_k = k_val
#                         best_lambda = lambda_val
#     
#     print('Overall Best Accuracy: {}'.format(best_acc))
#     print('Overall Best c: {}'.format(best_c))
#     print('Overall Best k: {}'.format(best_k))
#     print('Overall Best k: {}'.format(best_lambda))
# =============================================================================
    
    actual = []
    predicted = []
    models = create_models(NgramModelWithInterpolation, path, 3, .5, [.5, .3, .2])
    cities_list = []
    

       
    for key, cities in val_data.items():
        for city in cities:
            actual.append(key[0:2])
            pred = get_prediction(models, city)
            predicted.append(pred)
            cities_list.append(city)
            
    models_accuracy(actual, predicted)
    
    for i in range(0, len(actual)):
        if actual[i] != predicted[i]:
            print('Incorrect - City Name: {}, Actual Country: {}, Predicted Country: {}'.format(cities_list[i], actual[i], predicted[i]))

    
    test_data = []
    path = os.getcwd() + '/cities_test.txt'
    f = open(path)
    for line in f:
        test_data.append(line[0:-1])
    f.close()
    
    test_models = create_models(NgramModelWithInterpolation, path, 3, .5, [.5, .3, .2])
    test_predicted = []
    
    for city in test_data:
        pred = get_prediction(test_models, city)
        test_predicted.append(pred)
    
    file = open('test_labels.txt', 'w')
    for label in test_predicted:
        file.write('{}\n'.format(label))
    file.close()