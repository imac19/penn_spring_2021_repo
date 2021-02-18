import math, random
import numpy as np

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
            try:
                total = sum(self.ngrams[context].values())
                return self.ngrams[context][char]/total
            except:
                return 0.0

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
        
        p = np.exp(p)
        p = p ** (1/len(text))
        
        return p 

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        pass

    def get_vocab(self):
        pass

    def update(self, text):
        pass

    def prob(self, context, char):
        pass

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
# =============================================================================
#     m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
#     print(m.random_text(250))
# 
#     m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
#     print(m.random_text(250))
# 
#     m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
#     print(m.random_text(250))
# 
#     m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
#     print(m.random_text(250))
#     
# =============================================================================
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    print(m.perplexity('abcd'))
#1.189207115002721
    print(m.perplexity('abca'))
#inf
    print(m.perplexity('abcda'))
#1.515716566510398
