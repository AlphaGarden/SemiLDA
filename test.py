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

def convert_ocr_data(filename):
    output_data = {}
    with open(filename, 'r') as fp:
        ocr_data = json.load(fp)
        fp.close()
    if ocr_data:
        for key, text in ocr_data.items():
            pid = str(key).split("/")[2]
            if pid not in output_data.keys():
                output_data[pid] = list()
            output_data[pid].append(ocr_data[key])
    with open('ocr_results.json', 'w') as fp:
        json.dump(output_data, fp)
        fp.close()

if __name__ == '__main__':
    convert_ocr_data('/Users/garden/PycharmProjects/KickstarterLDA/data/ocr/ocr_results_518_610.json')