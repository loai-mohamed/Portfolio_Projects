import unicodedata
import re
import nltk


                                                #### preprocessing ####
    
    # some code acquired from pytorch.org/tutorials/beginner/chatbot_tutorial.html

    
MAX_LENGTH = 10  # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def prepare_text(sentence):
    
    '''

    Our text needs to be cleaned with a tokenizer. This function will perform that task.
    https://www.nltk.org/api/nltk.tokenize.html

    '''
    sentence = normalizeString(sentence)
    tokens = nltk.tokenize.word_tokenize(sentence)
    return tokens


