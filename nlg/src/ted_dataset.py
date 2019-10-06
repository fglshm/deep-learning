import pickle

def load_data(speech_type):
    pkl_filename = 'ted.{}.vocab.pkl'.format(speech_type)
    with open(pkl_filename, 'rb') as f:
        vocab = pickle.load(f)
    
    all_words, word_to_id, id_to_word = vocab
    return all_words, word_to_id, id_to_word