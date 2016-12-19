import re

def tokenize(text):
    for punct in './"\'!?-:;':
        text = text.replace(punct, " " + punct + " ")
    text = re.sub(r'([./"\'!?\-:;])', r' \1 ', text)
    tokens = re.split(r'\s+', text.lower())
    return [t for t in tokens if len(t)]

if __name__ == '__main__':
    print tokenize('hey there!')
