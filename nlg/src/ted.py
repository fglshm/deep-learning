import numpy as np
from ted_dataset import load_data

def data(speech_type):
    words, word_to_id, id_to_word = load_data(speech_type)
    corpus = np.load('ted.{}.corpus.npy'.format(speech_type))
    
    return corpus, word_to_id, id_to_word
