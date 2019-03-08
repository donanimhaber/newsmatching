import numpy as np
import logging
import html
import re
from pymongo import MongoClient
import jpype
from threading import Thread
import multiprocessing
from multiprocessing import Process
import time
import os
import nltk
import string

tr_stop = []
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'stopwords_tr.txt'), 'r') as f:
    for word in f.readlines():
        tr_stop.append(word.strip())
        
classpath = "zemberek-full.jar"
#jvmpath = "/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/libjvm.so"

if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" %classpath)

SentenceWordAnalysis = jpype.JClass('zemberek.morphology.analysis.SentenceWordAnalysis')
TurkishTokenizer = jpype.JClass('zemberek.tokenization.TurkishTokenizer').DEFAULT
TurkishMorphology = jpype.JClass('zemberek.morphology.TurkishMorphology')
turkishMorphology = TurkishMorphology.createWithDefaults()

def lower_tr(data):
    data = data.replace(u'İ',u'i')
    data = data.replace(u'I',u'ı')
    return data.lower()

def remove_newline(text):

    text = text.replace(r'\n', ' ')  # removes line feed from message
    text = text.replace(r'\r', ' ')
    text = text.replace('\\n', ' ')  # unicode
    text = text.replace('\\r', ' ')  #  unicode

    return text

def remove_punc(raw):

    raw = raw.replace('“', ' ')
    raw = raw.replace('”', ' ')
    raw = raw.replace('’', ' ')
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    raw = regex.sub(' ', raw)

    return raw

def remove_stops_tr(raw):

    tokens = nltk.word_tokenize(raw)

    words = [word for word in tokens if not word in tr_stop]

    return ' '.join(words)

r_punct = re.compile('[%s]' % re.escape(string.punctuation+'’'))
clear_html = re.compile('<.*?>')  # cleans html tags from message

def preprocess_ft(text):
    text.replace('’', ' ')
    text = text.replace('\\n',' ')
    text = text.replace('\\r',' ')
    text = text.replace(r'\n',' ')
    text = text.replace(r'\r',' ')

    text = html.unescape(text)
    text = re.sub(clear_html, ' ', text)
    text = lower_tr(text)

    text = r_punct.sub(' ', text)

    # clean and tokenize document string
    tokens = nltk.word_tokenize(text)

    # remove stop words from tokens
    # stopped_tokens = [i for i in tokens if not i in tr_stop]

    body = ' '.join(tokens)

    return body

token_pattern_compiled = re.compile('(?u)\\b[\\w\\d][\\w\\d]*\\b')

def preprocess02(text, tokenize=False):

    if text is None:
        return None

    if len(text)/(text.count(' ')+1) > 2100:
        logging.warning("ABUSE DETECTED! " + text[:100])
        return None

    text = html.unescape(text)

    text = re.sub(clear_html, ' ', text)  # clean html tags from message

    text = lower_tr(text)  # lowerize

    text = text.replace('_', ' ')

    tokens = token_pattern_compiled.findall(text) # for clearing punctuations hard way

    if tokenize:
        return tokens
    else:
        return ' '.join(tokens)


def process_lemma_stops(text, tokenize=False):

    if text is None:
        return None

    if len(text) / (text.count(' ') + 1) > 2100:
        logging.warning("ABUSE DETECTED! " + text[:100])
        return None

    text = html.unescape(text)

    text = re.sub(clear_html, ' ', text)  # clean html tags from message

    text = lower_tr(text)  # lowerize

    text = text.replace('_', ' ')

    cleaned_tokens = []

    sentences = nltk.sent_tokenize(text)

    for sentence in sentences:
        body_words = []
        try:
            if sentence:
                e = turkishMorphology.analyzeAndDisambiguate(sentence).bestAnalysis()
                for x in e:
                    lemma = max(x.getLemmas(), key=len)
                    if not lemma == "UNK":
                        body_words.append(lemma)
                    else:
                        body_words.append(x.getStem())
                    lemma = None
            cleaned_tokens.append(' '.join(body_words))
        except:
            cleaned_tokens.append(' '.join(sentence.split()))

    text = ' '.join(cleaned_tokens)

    tokens = token_pattern_compiled.findall(text)  # for clearing punctuations hard way

    cleaned_tokens = [i for i in tokens if not i in tr_stop]

    if tokenize:
        return cleaned_tokens
    else:
        return ' '.join(cleaned_tokens)