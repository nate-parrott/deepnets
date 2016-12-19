from wiki_crawl import load_paragraphs
from wiki_hmm import hmm_model

def null_model(training):
    def null(tokens):
        return [0 for _ in tokens]
    return null

def uppercase_links(words, states):
    return u" ".join([(w.upper() if state else w.lower()) for w, state in zip(words, states)])

if __name__ == '__main__':
    training = load_paragraphs(limit=100)
    model = hmm_model(training)
    for para, states in training:
        print "REAL:"
        print uppercase_links(para, states)
        print "PREDICTED:"
        print uppercase_links(para, model(para))
        print "\n"
        # quit()
