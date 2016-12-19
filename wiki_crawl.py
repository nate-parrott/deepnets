import wikipedia
from bs4 import BeautifulSoup, Tag, NavigableString
import hashlib
import json
import os
import random
import urllib
import re

def linkify(soup):
    if soup.name == 'a' and soup.has_attr('href') and soup['href'].startswith('/wiki/'):
        link = urllib.unquote(soup['href'][len('/wiki/'):].replace('_', ' '))
        return [[link, soup.text]]
    else:
        out = []
        for child in soup:
            if isinstance(child, Tag):
                out += linkify(child)
            elif isinstance(child, NavigableString):
                out.append(unicode(child))
        return out

def crawl(title):
    try:
        page = wikipedia.page(title)
    except wikipedia.PageError:
        return None
    except wikipedia.exceptions.DisambiguationError:
        return None
    if page:
        html = page.html()
        soup = BeautifulSoup(html, 'lxml')
        paragraphs = []
        for p in soup.find_all('p'):
            paragraphs.append(linkify(p))
        print paragraphs
    # write paragraphs:
    name = hashlib.md5(title.encode('ascii', 'ignore')).hexdigest()
    with open('wikipedia/' + name + '.json', 'w') as f:
        info = {
            "title": title,
            "paragraphs": paragraphs
        }
        f.write(json.dumps(info))
        return info

def crawl_loop(initial_articles=[]):
    initial_articles = initial_articles[:]
    
    crawl_state_path = 'wikipedia/crawl.json'
    if os.path.exists(crawl_state_path):
        state = json.load(open(crawl_state_path))
        queue = set(state['queue'])
        crawled = set(state['crawled'])
    else:
        queue = set(initial_articles)
        crawled = set()
    
    while len(queue):
        if len(initial_articles):
            title = initial_articles.pop()
        else:
            title = queue.pop()
        if title not in crawled:
            print 'Crawling', title
            results = crawl(title)
            if results:
                for para in results['paragraphs']:
                    for item in para:
                        if isinstance(item, list):
                            linked = item[0]
                            if linked not in queue and linked not in crawled:
                                queue.add(linked)
            else:
                print 'Failed to crawl', title
            crawled.add(title)
        # save state:
        with open(crawl_state_path, 'w') as f:
            f.write(json.dumps({'queue': list(queue), 'crawled': list(crawled)})) 
    
    print 'Done crawling..?!?!'

def flatten(iters):
    return [item for iter in iters for item in iter]

def load_paragraphs(limit=None):
    docs = []
    for name in os.listdir('wikipedia'):
        if name.endswith('.json') and name != 'crawl.json':
            path = 'wikipedia/' + name
            content = json.load(open(path))
            docs.append(content)
            if limit is not None and len(docs) >= limit:
                break
    paragraphs = flatten((doc['paragraphs'] for doc in docs))
    paragraphs = [p for p in paragraphs if len(p) > 0]
    data = map(tokenize_and_mask, paragraphs)
    return [d for d in data if len(d[0]) > 0]

def tokenize(text):
    tokens = re.split(r'\s+', re.sub(r'[,.?!]+', '', text.lower()))
    return [t for t in tokens if len(t)]

def tokenize_and_mask(para):
    tokens = []
    mask = []
    for item in para:
        if isinstance(item, list):
            new_tokens = tokenize(item[1])
            tokens += new_tokens
            mask += [1] * len(new_tokens)
        else:
            new_tokens = tokenize(item)
            tokens += new_tokens
            mask += [0] * len(new_tokens)
    return tokens, mask

if __name__ == '__main__':
    crawl_loop(['samsung', 'snapchat', 'bernie sanders', 'climate change', 'kanye west'])
