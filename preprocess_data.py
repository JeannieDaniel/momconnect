"""
Functions for preprocessing data for training purposes
Author: Jeanne Elizabeth Daniel
April 2019

"""

import pandas as pd
import numpy as np
import gensim
 
def preprocess(text, min_token_length = 0, join = False):
    
    """ Method for preprocessing text
    
    Args:
        text: string of text
        min_token_length: integer value indicating min number of characters in a token
        join: boolean indicating if function should join the list of tokens into the string or not
    
    Returns:
        list of cleaned words or joined string
    """
    
    if type(text) != str:
        return []
    
    result = []
    for token in gensim.utils.simple_preprocess(text, min_len = min_token_length):
        if len(token) > 2:
            result.append(token)

    if join:
        return ' '.join(result)
    return result
    
def create_dictionary(train_data, no_below = 1, no_above = 0.25, keep_n = 95000, min_token_length = 0):
    
    """ Create dictionary of all words in our dataset that adhere to the following conditions:

    Args:
        train_data: dataframe with questions
        no_below: integer = minimum number of occurrences in the dataset
        no_above: float between 0 and 1 - proportion of sentences containing word
        keep_n: max number of words in our vocabulary
        min_token_length: minimum number of characters a token must have
    
    Returns:
        dictionary of words found in training set in "dict" format
    
    """
    
    documents = train_data[['helpdesk_question']]
    documents['index'] = documents.index
    processed_docs = documents['helpdesk_question'].apply(preprocess, args = [min_token_length])
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary_of_words = pd.DataFrame(pd.Series(dict(dictionary)))
    dictionary_of_words['index'] = dictionary_of_words.index 

    return dictionary_of_words.set_index(0)['index'].to_dict()


def filter_words(text_list, dictionary):
    
    """ Filter sentences to remove any words from that does not appear in our dictionary
    
    Args:
        text_list: list of words in a sentence
        dictionary: dictionary of words in training set
    
    Returns:
        Filtered list of words in a sentence
    """
    
    result = []
    for t in text_list:
        if dictionary.get(t) != None:
            result.append(t)
    return result


def preprocess_question(question, dictionary, minimum_token_length):
    
    """ Create list of cleaned and filtered words for each sentence
    
    Args:
        question: string text
        dictionary: dictionary of words in training set
    
    Return:
        Cleaned and filtered list of words
    """
    
    return filter_words(preprocess(question, minimum_token_length), dictionary)

def create_lookup_tables(unique_words):
    
    """ Create lookup tables for word_to_id and id_to_word
    
    Args:
        unique_words: dictionary of words in training set
    
    Return:
        word_to_id: dict with words as keys and corresponding ids as values
        id_to_word: dict with ids as keys and corresponding words as values
    """
    
    word_to_id = {}  # word->id lookup
    id_to_word = {}  # id->word lookup
    for index, word in enumerate(sorted(list(unique_words))):
        word_to_id[word] = index + 1
        id_to_word[index + 1] = word
    return word_to_id, id_to_word

def transform_sequence_to_word_ids(seq, word_to_id):
    
    """ Create list of word IDs for sequence of words, padded with zeroes and truncated to a fixed length
    
    Args:
        seq: list of words
        word_to_id: dict with words as keys and corresponding ids as values
        
    Return:
        list of word IDs padded and truncated to length of 30 items
    """
    
    seq_word_ids = []
    for word in seq:
        seq_word_ids.append([word_to_id[word]])
    
    # pad sequence with zeros
    for i in range(30 - len(seq_word_ids)):
        seq_word_ids.append([0])
        
    return seq_word_ids[:30]

def create_one_hot_vector_for_reply(reply, all_responses):
    
    """ Constructs a one-hot vector for replies
    
    Args:
        reply: query item 
        all_responses: dict containing all the template responses with their corresponding IDs
    
    Return:
        a one-hot vector where the corresponding ID of the reply is the one-hot index
    
    """
    
    Y = np.zeros(len(all_responses), dtype = int)
    Y[all_responses[reply]] += 1
    return Y 


def label_preprocess(entry, responses):
    
    """ Returns integer ID corresponding to response for easy comparison and classification
    
    Args:
        entry: query item 
        responses: dict containing all the template responses with their corresponding IDs
        
    Return: 
        integer corresponding to each response     
        
    """
    
    if responses.get(entry) != None:
        return responses[entry]
    else:
        return len(responses) #default unknown class


def sample_pairs_offline(df, sample_size = 10):
    
    """ Offline sampling for sentence pairs
    
    Args:
        df: dataframe of questions and answers
        sample_size: number of positive/negative samples per sentence
    
    Returns:
        a data frame of positive and negative pairs
    
    """
    
    sentences_1 = []
    sentences_2 = []
    labels = []
    sample_size = sample_size

    df['helpdesk_question_clean'] = df['helpdesk_question'].apply(preprocess_data.preprocess, args = [0, True])

    for group in df.groupby('helpdesk_reply'):
        questions = list(group[1]['helpdesk_question_clean'])
        low_resource = list(group[1]['low_resource'])

        for i in range(len(questions)):
            q = questions[i]
            if len(preprocess_data.preprocess(q, 0)) > 0:
                for s in list(group[1]['helpdesk_question_clean'].sample(sample_size)):
                    if s != q and len(preprocess_data.preprocess(s, 0)) > 0:
                        if s > q:
                            sentences_1.append(s)
                            sentences_2.append(q)
                            labels.append(1) # positive
                        else:
                            sentences_1.append(q)
                            sentences_2.append(s)
                            labels.append(1) # positive

                #sample negatives
                negatives = df.loc[df['helpdesk_reply'] != group[0]]
                samples = negatives['helpdesk_question_clean'].sample(sample_size)

                if samples.shape[0] > 0:
                    for s in list(samples):
                        if len(preprocess_data.preprocess(s, 0)) > 0:
                            if s > q:
                                sentences_1.append(s)
                                sentences_2.append(q)
                                labels.append(0) # negative
                            else:
                                sentences_1.append(q)
                                sentences_2.append(s)
                                labels.append(0) #negative
                                
    data_pairs = pd.concat([pd.Series(sentences_1), pd.Series(sentences_2), pd.Series(labels)], axis = 1)               
    del sentences_1, sentences_2, labels
    return data_pairs.drop_duplicates()


