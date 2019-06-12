import pandas as pd
import nltk
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import json


def lemmatize(campaign_list):
    """
    Return the campaign list after being lemmatized
    :param text_list:
    :return:
    """
    result = []
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    for campaign in campaign_list:
        token_list = []
        tokenization = nltk.word_tokenize(campaign, language='english')
        word_pos = nltk.pos_tag(tokenization)
        for word, tag in word_pos:
            print("word %s, %s" % (word, tag))
        for word, tag in word_pos:
            print(wordnet_lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)))
        for w in tokenization:
            token_list.append(wordnet_lemmatizer.lemmatize(w.lower(), pos='n'))
        result.append(" ".join(token_list))
    print(result)


def get_wordnet_pos(treebank_tag):  # get part of speech from tag
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def covert2urls():
    df = pd.read_csv('projects.csv', thousands=',')
    urls = {}
    for i, link in enumerate(df['ProjectLink'].values):
        urls[i] = link
    with open('urls.json', 'w') as fp:
        json.dump(urls, fp)
        fp.close()


def filter_words(tokens, stop_words):
    """
    filter the word by nltk stopwords and length
    """
    return [w for w in tokens if w not in stop_words and len(w) > 3]


def convert_result(file_name):
    line_num = 1
    result = []
    with open(file_name, 'r') as fp:
        for line in fp:
            if line_num % 11 == 0:
                result.append(int(line.split(":")[3].strip(" ").rstrip("\n")))
            line_num += 1
    plt.plot(range(10, 1000, 10), result)
    plt.xlabel("Topic_Num")
    plt.ylabel("Log Likelihood")
    plt.show()


if __name__ == '__main__':
    a = [{'a1' : 1}, {'a2': 2}]
    b = [{'b1' : 1}, {'b2' : 2}]
    print(a + b)