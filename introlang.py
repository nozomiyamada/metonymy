"""
@author: Nozomi
"""
import numpy as np
from gensim.models import KeyedVectors
nor = np.linalg.norm
import warnings
warnings.filterwarnings('ignore')

# load vector data : both cbow and skip-gram
model1 = KeyedVectors.load_word2vec_format('./cbow.bin', unicode_errors='ignore', binary=True)
model2 = KeyedVectors.load_word2vec_format('./skip.bin', unicode_errors='ignore', binary=True)


# function for calculating the similarity of 2 words

# compare results of cbow and skip-gram, and take the better one
def result(result_cbow, result_skip):
    return max(result_cbow, result_skip)

def sim(word1, word2):
    if word1 not in model1.wv.vocab:
        print('{} not in the vocabulary'.format(word1))
        return
    if word2 not in model1.wv.vocab:
        print('{} not in the vocabulary'.format(word2))
        return
    vec1_c = model1.wv[word1]  # cbow
    vec2_c = model1.wv[word2]
    vec1_s = model2.wv[word1]  # skip-gram
    vec2_s = model2.wv[word2]
    result_c = round(float(np.dot(vec1_c, vec2_c)) / (nor(vec1_c) * nor(vec2_c)), 4)
    result_s = round(float(np.dot(vec1_s, vec2_s)) / (nor(vec1_s) * nor(vec2_s)), 4)
    return result(result_c, result_s)


### extra function ###

# function for searching similar words
def most_sim(word, n=3):
    if word not in model1.wv.vocab:
        print('{} not in the vocabulary'.format(word))
        return

    results1 = model1.wv.most_similar(positive=[word], topn=n)
    print('model 1')
    for result in results1:
        print(result[0], round(result[1], 4))
    results2 = model2.wv.most_similar(positive=[word], topn=n)
    print('\nmodel 2')
    for result in results2:
        print(result[0], round(result[1], 4))


# function for arithmetic
def plus(pos1, pos2, n=3):
    """
    calculate: pos1 + pos2
    
    plus('หนุ่ม', 'ภรรยา')
    > 'สามี'
    """
    if pos1 not in model1.wv.vocab:
        print('{} not in the vocabulary'.format(pos1))
        return
    if pos2 not in model1.wv.vocab:
        print('{} not in the vocabulary'.format(pos2))
        return

    results1 = model1.wv.most_similar(positive=[pos1, pos2], topn=n)
    print('model 1')
    for result in results1:
        print(result[0], round(result[1], 4))
    results2 = model2.wv.most_similar(positive=[pos1, pos2], topn=n)
    print('\nmodel 2')
    for result in results2:
        print(result[0], round(result[1], 4))
        
def minus(pos, neg, n=3):
    """
    calculate: pos - neg
    """
    if pos not in model1.wv.vocab:
        print('{} not in the vocabulary'.format(pos))
        return
    if neg not in model1.wv.vocab:
        print('{} not in the vocabulary'.format(neg))
        return

    results1 = model1.wv.most_similar(positive=[pos], negative=[neg], topn=n)
    print('model 1')
    for result in results1:
        print(result[0], round(result[1], 4))
    results2 = model2.wv.most_similar(positive=[pos], negative=[neg], topn=n)
    print('\nmodel 2')
    for result in results2:
        print(result[0], round(result[1], 4))

def calc(pos1, neg1, pos2, n=3):
    """
    calculate: pos1 - neg1 + pos2
    
    calc('โตเกียว', 'ญี่ปุ่น', 'จีน')
    > 'ปักกิ่ง'
    """
    if pos1 not in model1.wv.vocab:
        print('{} not in the vocabulary'.format(pos1))
        return
    if neg1 not in model1.wv.vocab:
        print('{} not in the vocabulary'.format(neg1))
        return
    if pos2 not in model1.wv.vocab:
        print('{} not in the vocabulary'.format(pos2))
        return


    results1 = model1.wv.most_similar(positive=[pos1, pos2], negative=[neg1], topn=n)
    print('model 1')
    for result in results1:
        print(result[0], round(result[1], 4))
    results2 = model2.wv.most_similar(positive=[pos1, pos2], negative=[neg1], topn=n)
    print('\nmodel 2')
    for result in results2:
        print(result[0], round(result[1], 4))