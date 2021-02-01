'''Homework 1 Python Questions

This is an individual homework
Implement the following functions.

Do not add any more import lines to this file than the ones
already here without asking for permission on Piazza.
Use the regular expression tools built into Python; do NOT use bash.
'''

import re

def check_for_foo_or_bar(text):
   '''Checks whether the input string meets the following condition.

   The string must have both the word 'foo' and the word 'bar' in it,
   whitespace- or punctuation-delimited from other words.
   (not, e.g., words like 'foobar' or 'bart' that merely contain
    the word 'bar');

   See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#match-objects

   Return:
     True if the condition is met, false otherwise.
   '''
   
   word_list = re.split('\W+', text)
   
   if 'foo' in word_list:
       if 'bar' in word_list:
           return True
       
   return False


def replace_rgb(text):
   '''Replaces all RGB or hex colors with the word 'COLOR'
   
   Possible formats for a color string:
   #0f0
   #0b013a
   #37EfaA
   rgb(1, 1, 1)
   rgb(255,19,32)
   rgb(00,01, 18)
   rgb(0.1, 0.5,1.0)

   There is no need to try to recognize rgba or other formats not listed 
   above. There is also no need to validate the ranges of the rgb values.

   However, you should make sure all numbers are indeed valid numbers.
   For example, '#xyzxyz' should return false as these are not valid hex digits.
   Similarly, 'rgb(c00l, 255, 255)' should return false.

   Only replace matching colors which are at the beginning or end of the line,
   or are space separated from the text around them. For example, due to the 
   trailing period:

   'I like rgb(1, 2, 3) and rgb(2, 3, 4).' becomes 'I like COLOR and rgb(2, 3, 4).'

   # See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#re.sub

   Returns:
     The text with all RGB or hex colors replaces with the word 'COLOR'
   '''
   
   rgb_case_one = ' #[A-F0-9a-f]{3,6} '
   rgb_case_two = '^#[A-F0-9a-f]{3,6} '
   rgb_case_three = ' #[A-F0-9a-f]{3,6}$'
   rgb_case_four = ' rgb\([ ]*?[0-9]*[.]?[0-9]*[ ]*[,][ ]*[0-9]*[.]?[0-9]*[ ]*[,][ ]*[0-9]*[.]?[0-9]*[ ]*[)] '
   rgb_case_five = '^rgb\([ ]*?[0-9]*[.]?[0-9]*[ ]*[,][ ]*[0-9]*[.]?[0-9]*[ ]*[,][ ]*[0-9]*[.]?[0-9]*[ ]*[)] '
   rgb_case_six = ' rgb\([ ]*?[0-9]*[.]?[0-9]*[ ]*[,][ ]*[0-9]*[.]?[0-9]*[ ]*[,][ ]*[0-9]*[.]?[0-9]*[ ]*[)]$'
   rgb_case_seven = '^#[A-F0-9a-f]{3,6}$'
   rgb_case_eight = '^rgb\([ ]*?[0-9]*[.]?[0-9]*[ ]*[,][ ]*[0-9]*[.]?[0-9]*[ ]*[,][ ]*[0-9]*[.]?[0-9]*[ ]*[)]$'

   rgb_case_list_one = [rgb_case_one, rgb_case_four]
   rgb_case_list_two = [rgb_case_two, rgb_case_five]
   rgb_case_list_three = [rgb_case_three, rgb_case_six]
   rgb_case_list_four = [rgb_case_seven, rgb_case_eight]
   replace_text = text
   
   for case in rgb_case_list_one:
       replace_text = re.sub(case, ' COLOR ', replace_text)
       
   for case in rgb_case_list_two:
       replace_text = re.sub(case, 'COLOR ', replace_text)
       
   for case in rgb_case_list_three:
       replace_text = re.sub(case, ' COLOR', replace_text)
       
   for case in rgb_case_list_four:
       replace_text = re.sub(case, 'COLOR', replace_text)
       
   return replace_text


def edit_distance(str1, str2):
    '''Computes the minimum edit distance between the two strings.

    Use a cost of 1 for all operations.

    See Section 2.4 in Jurafsky and Martin for algorithm details.
    Do NOT use recursion.

    Returns:
      An integer representing the string edit distance
      between str1 and str2
    '''
  
    len_str1 = len(str1)
    len_str2 = len(str2)
  
    dist_mat = []

    for i in range(0, len_str2+1):
        dist_mat.append([0] * (len_str1+1))
  
    for i in range(1, len_str2+1):
        dist_mat[i][0] = i 
      
    for j in range(1, len_str1+1):
        dist_mat[0][j] = j
  
    for i in range(1, len_str2+1):
        for j in range(1, len_str1+1):
            add_del_cost = 1
            sub_cost = 0 if str2[i-1]==str1[j-1] else 1
            
            dist_one = dist_mat[i-1][j] + add_del_cost
            dist_two = dist_mat[i][j-1] + add_del_cost
            dist_three = dist_mat[i-1][j-1] + sub_cost
            
            dist_mat[i][j] = min([dist_one, dist_two, dist_three])
            
    return dist_mat[len_str2][len_str1]
      

def wine_text_processing(wine_file_path, stopwords_file_path):
    '''Process the two files to answer the following questions and output results to stdout.

    1. What is the distribution over star ratings?
    2. What are the 10 most common words used across all of the reviews, and how many times
       is each used?
    3. How many times does the word 'a' appear?
    4. How many times does the word 'fruit' appear?
    5. How many times does the word 'mineral' appear?
    6. Common words (like 'a') are not as interesting as uncommon words (like 'mineral').
       In natural language processing, we call these common words "stop words" and often
       remove them before we process text. stopwords.txt gives you a list of some very
       common words. Remove these stopwords from your reviews. Also, try converting all the
       words to lower case (since we probably don't want to count 'fruit' and 'Fruit' as two
       different words). Now what are the 10 most common words across all of the reviews,
       and how many times is each used?
    7. You should continue to use the preprocessed reviews for the following questions
       (lower-cased, no stopwords).  What are the 10 most used words among the 5 star
       reviews, and how many times is each used? 
    8. What are the 10 most used words among the 1 star reviews, and how many times is
       each used? 
    9. Gather two sets of reviews: 1) Those that use the word "red" and 2) those that use the word
       "white". What are the 10 most frequent words in the "red" reviews which do NOT appear in the
       "white" reviews?
    10. What are the 10 most frequent words in the "white" reviews which do NOT appear in the "red"
        reviews?

    No return value.
    '''

    with open(wine_file_path, encoding='utf-8', errors='ignore') as f:
        wine_file = f.readlines()
        f.close()
    
    with open(stopwords_file_path, encoding='utf-8', errors='ignore') as f:
        stopwords_file = f.readlines()
        f.close()
    
    # Question 1
    star_dict = {6:0,5:0,4:0,3:0,2:0,1:0}
    for review in wine_file:
        stars = re.sub('\n', '', re.split('\t', review)[-1])
        star_dict[len(stars)]+=1
    
    for stars in star_dict.keys():
        print('*'*stars + '\t' + str(star_dict.get(stars)))
        
    # Question 2,3,4,5
    print()
    word_dict = {}
    for review in wine_file:
        for word in re.split(' ', review):
            word = re.sub('\n|\t|\*', '', word)
            if word in word_dict.keys():
                word_dict[word]+=1
            else:
                word_dict[word]=1
    word_dict_sorted = sorted(word_dict.items(), key= lambda item: (-item[1], item[0]), reverse=False)
    
    for item in word_dict_sorted[0:10]:
        print(item[0] + '\t' + str(item[1]))
    
    print()
    print(str(word_dict.get('a')))
    print()
    print(str(word_dict.get('fruit')))
    print()
    print(str(word_dict.get('mineral')))
    
    # Question 6
    print()
    stopwords_set = set()
    for word in stopwords_file:
        stopwords_set.add(re.sub('\n', '', word))
    word_dict = {}
    for review in wine_file:
        for word in re.split(' ', review):
            word = re.sub('\n|\t|\*', '', word.lower())
            if word not in stopwords_set:
                if word in word_dict.keys():
                    word_dict[word]+=1
                else:
                    word_dict[word]=1
    word_dict_sorted = sorted(word_dict.items(), key= lambda item: (-item[1], item[0]), reverse=False)
    
    for item in word_dict_sorted[0:10]:
        print(item[0] + '\t' + str(item[1]))
    
    # Question 7, 8
    print()
    five_star_reviews = []
    one_star_reviews = []
    for review in wine_file:
        stars = re.sub('\n', '', re.split('\t', review)[-1])
        if len(stars)==5:
            five_star_reviews.append(review)
        if len(stars)==1:
            one_star_reviews.append(review)
            
    stopwords_set = set()
    for word in stopwords_file:
        stopwords_set.add(re.sub('\n', '', word))
    word_dict = {}
    for review in five_star_reviews:
        for word in re.split(' ', review):
            word = re.sub('\n|\t|\*', '', word.lower())
            if word not in stopwords_set:
                if word in word_dict.keys():
                    word_dict[word]+=1
                else:
                    word_dict[word]=1
    word_dict_sorted = sorted(word_dict.items(), key= lambda item: (-item[1], item[0]), reverse=False)
    
    for item in word_dict_sorted[0:10]:
        print(item[0] + '\t' + str(item[1]))
        
    print()
    word_dict = {}
    for review in one_star_reviews:
        for word in re.split(' ', review):
            word = re.sub('\n|\t|\*', '', word.lower())
            if word not in stopwords_set:
                if word in word_dict.keys():
                    word_dict[word]+=1
                else:
                    word_dict[word]=1
    word_dict_sorted = sorted(word_dict.items(), key= lambda item: (-item[1], item[0]), reverse=False)
    
    for item in word_dict_sorted[0:10]:
        print(item[0] + '\t' + str(item[1]))
    
    # Question 9,10
    
    print()
    red_reviews = []
    white_reviews = []
    for review in wine_file:
        word_list = []
        for word in re.split(' ', review):
            word = re.sub('\n|\t|\*', '', word.lower())
            word_list.append(word)
        if 'red' in word_list:
            red_reviews.append(review)
        if 'white' in word_list:
            white_reviews.append(review)
            
    red_word_dict = {}
    red_word_list = []
    for review in red_reviews:
        for word in re.split(' ', review):
            word = re.sub('\n|\t|\*', '', word.lower())
            if word not in stopwords_set:
                if word in red_word_dict.keys():
                    red_word_dict[word]+=1
                else:
                    red_word_dict[word]=1
                    red_word_list.append(word)
    red_word_dict_sorted = sorted(red_word_dict.items(), key= lambda item: (-item[1], item[0]), reverse=False)
    
    white_word_dict = {}
    white_word_list = []
    for review in white_reviews:
        for word in re.split(' ', review):
            word = re.sub('\n|\t|\*', '', word.lower())
            if word not in stopwords_set:
                if word in white_word_dict.keys():
                    white_word_dict[word]+=1
                else:
                    white_word_dict[word]=1
                    white_word_list.append(word)
    white_word_dict_sorted = sorted(white_word_dict.items(), key= lambda item: (-item[1], item[0]), reverse=False)
    
    red_printcount = 0
    iterator=0
    while red_printcount<10:
        if red_word_dict_sorted[iterator][0] not in white_word_list:
            print(red_word_dict_sorted[iterator][0] + '\t' + str(red_word_dict_sorted[iterator][1]))
            red_printcount+=1
        iterator+=1
            
    print()
    white_printcount = 0
    iterator=0
    while white_printcount<10:
        if white_word_dict_sorted[iterator][0] not in red_word_list:
            print(white_word_dict_sorted[iterator][0] + '\t' + str(white_word_dict_sorted[iterator][1]))
            white_printcount+=1
        iterator+=1
    
    
    
    
        
    
# Testing functions

# =============================================================================
# # check_for_foo_or_bar test 
# print('check_for_foo_or_bar test')
# # Should be true
# print(check_for_foo_or_bar('jsdf ndsfnb foo fdklnf bar adnf'))
# # Should be false
# print(check_for_foo_or_bar('jsdf ndsfnb foofg fdklnf bar adnf'))
# # Should be false
# print(check_for_foo_or_bar('jsdf ndsfnb foo fdklnf barfg adnf'))
# # Should be false
# print(check_for_foo_or_bar('jsdf ndsfnb foo fdklnf bar adnf'))
# print()
# 
# 
# # replace_rgb test 
# print('replace_rgb test')
# # Should become 'I like COLOR and rgb(2, 3, 4).'
# print(replace_rgb('I like rgb(1, 2, 3) and rgb(2, 3, 4).'))
# # Should become 'I like COLOR and COLOR'
# print(replace_rgb('I like rgb(1, 2, 3) and #F43D45'))
# # Should become 'I like COLOR and COLOR'
# print(replace_rgb('I like rgb(145345, 4352, 3.564) and #F45'))
# # Should become 'I like rgb(c00l, 255, 255) and #F56L45'
# print(replace_rgb('I like rgb(c00l, 255, 255) and #F56L45'))
# print()
# 
# # edit_distance test
# print('edit_distance test')
# # Should print 4
# print(edit_distance('malignant', 'malificent'))
# print()
# =============================================================================

# wine_text_processing test 
# print('wine_text_processing test')
# wine_text_processing('data/wine.txt', 'data/stopwords.txt')
