import nltk
from nltk.tag import pos_tag
from simple import *
import json
import random
from tokenize_str import tokenize

def count_nltk_accuracy():

    test_sents = json.load(open('data/pos.test.json'))
    test_words = flatten(test_sents)
    test_tokens = [t[0].lower() for t in test_words]
    test_tags = [t[1] for t in test_words]
    predicted_tags = [t[1] for t in pos_tag(test_tokens)]
    accuracy = avg([(1 if strip_tag(a) == strip_tag(b) else 0) for a,b in zip(test_tags, predicted_tags)])

    print "NLTK accuracy:", accuracy * 100

    # NLTK accuracy: 64.290304904

def strip_tag(tag):
    return tag.split('-')[0].split('+')[0]

def process_sent(sent):
    return [(word, strip_tag(tag)) for word, tag in sent]

def write_nltk_data():
    sents = list(nltk.corpus.brown.tagged_sents())
    sents = [process_sent(s) for s in sents]
    random.shuffle(sents)
    cutoff = 50000
    train = sents[:cutoff]
    test = sents[cutoff:]
    open('data/pos.train.json', 'w').write(json.dumps(train))
    open('data/pos.test.json', 'w').write(json.dumps(test))

def interactive_tagger():
    while True:
        text = raw_input(' > ')
        print pos_tag(tokenize(text))

if __name__ == '__main__':
    # write_nltk_data()
    # count_nltk_accuracy()
    interactive_tagger()
