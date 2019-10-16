import guidedlda
import json
import numpy as np
import glob
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from stanfordcorenlp import StanfordCoreNLP
from corextopic import corextopic as ct
import scipy.sparse as ss
from scipy import spatial

TOPIC_NUMS_TUPLE = (50,)
ITERATION_NUMS = 200
TOP_K_WORDS = 25
TOP_K_DOC = 10
SEED_CONFIDENCE = 0.99
CACHE = True
DATA_PATH = 'data/'
WORD = 'word/'
CAMPAIGNS = 'campaigns/'
TRANSCRIPTIONS = 'transcriptions/'
OCR = 'ocr/'
CLEAN_DATA = 'clean_data/'
COMBINE_DATA = 'combine_data/'
KS_PROJECT = 'ksproject/'

KS_PROJECT_FILE_PATH = DATA_PATH + KS_PROJECT + 'ksProject.xlsx'

STOP_WORDS_FILE_PATH = DATA_PATH + WORD + 'stop_word_list.txt'
SEED_WORDS_FILE_PATH = DATA_PATH + WORD + 'seed_word.txt'

CAMPAIGN_FILE_PATH = DATA_PATH + CAMPAIGNS
CAMPAIGN_CACHE_FILE = DATA_PATH + CLEAN_DATA + '_campaign.txt'
RAW_CAMPAIGN_CACHE_FILE = DATA_PATH + CLEAN_DATA + '_raw_campaign.json'

TRANSCRIPTION_FILE_PATH = DATA_PATH + TRANSCRIPTIONS
TRANSCRIPTION_CACHE_FILE = DATA_PATH + CLEAN_DATA + '_transcription.txt'
RAW_TRANSCRIPTION_CACHE_FILE = DATA_PATH + CLEAN_DATA + '_raw_transcription.json'

COMBINATION_DATA_CACHE_FILE = DATA_PATH + CLEAN_DATA + '_combination.txt'
RAW_COMBINATION_DATA_CACHE_FILE = DATA_PATH + CLEAN_DATA + '_raw_combination.json'

OCR_FILE_PATH = DATA_PATH + OCR


TRANSCRIPTION_COLUMN = ['NarTopicTeamLoading', 'NarTopicProductLoading', 'NarTopicMotivationLoading', 'NarTopicRewardLoading']
LOADING_COLUMN = ['DesTopicTeamLoading', 'DesTopicProductLoading', 'DesTopicMotivationLoading', 'DesTopicRewardLoading',
                  'NarTopicTeamLoading', 'NarTopicProductLoading', 'NarTopicMotivationLoading', 'NarTopicRewardLoading',
                  'Consistency']


# combine_data the campaign and narratives together and re-run the model
# compute the cosine similarity for the document_topic between the campaign and narrative

def main():
    X, word2id, vocab = load_data('transcription')
    # X, word2id, vocab = load_data('transcription')
    seed_topic_list = load_seed_words()
    loglikelihoods = []
    # seed_word_count(X, word2id, seed_topic_list)
    top_k_word(X, 30, vocab)
    for topic_num in TOPIC_NUMS_TUPLE:
        model = guided_analysis(X, word2id, topic_num, SEED_CONFIDENCE, seed_topic_list)
        # model = non_guided_analysis(X, topic_num)
        loglikelihoods.append(model.loglikelihood())
        retrieve_words_from(model, vocab, topic_num, TOP_K_WORDS)
        # calculate_loading(model, topic_num, seed_topic_list)
        # export_seeds_assignment(model, word2id, topic_num, seed_topic_list)
    plt.plot(list(TOPIC_NUMS_TUPLE), loglikelihoods)
    plt.show()


def run_corextopic():
    X, word2id, vocab = load_data('campaign')
    seed_topic_list = load_seed_words()
    model = corextopic_analysis(X, vocab, word2id, 50, seed_topic_list)
    doc_topic = model.p_y_given_x
    for doc_id in range(10):
        row = []
        for topic_id in range(4):
            row.append(doc_topic[doc_id][topic_id])
        print(row)
    topics = model.get_topics(topic=0, n_words=25)
    print(topics)


def guided_analysis(X, word2id, topic_num, confidence, seed_topic_list):
    """
    Guided Analysis on the given dtm
    """
    model = guidedlda.GuidedLDA(n_topics=topic_num, n_iter=ITERATION_NUMS, random_state=7, refresh=20)
    model.fit(X, seed_topics=load_seed_topics(word2id, seed_topic_list), seed_confidence=confidence)
    return model


def non_guided_analysis(X, topic_num):
    """
    Non_guided Analysis on the given dtm
    """
    model = guidedlda.GuidedLDA(n_topics=topic_num, n_iter=ITERATION_NUMS, random_state=7, refresh=20)
    model.fit(X)
    return model


def corextopic_analysis(X, vocab, word2id, topic_num, seed_topic_list):
    model = ct.Corex(n_hidden=topic_num, max_iter=200)
    seed_words = [[word for word in words if word in word2id.keys()] for words in seed_topic_list.values()]
    model.fit(ss.csr_matrix(X), words=vocab, anchors=seed_words, anchor_strength=4)
    return model


def calculate_loading(model, topic_num, seed_topic_list, n_top_docs=TOP_K_DOC, human_readable=False, sorting=False):
    """
    Export the loading for the model
    """
    doc_topic = model.doc_topic_
    doc_num = doc_topic.shape[0]
    result = []
    seed_topic_num = len(seed_topic_list.keys())
    if sorting:
        sorted_index = np.argsort(doc_topic, axis=0)  # sort the data with index row by row
        loading_set = range(doc_num - 1, doc_num - n_top_docs - 1, -1)
        if human_readable:
            data_list = read_json(RAW_CAMPAIGN_CACHE_FILE)
            campaigns = [' '.join(project['ProjectCampaign']).replace('\n', '') for project in data_list]
        else:
            campaigns = read_file(CAMPAIGN_CACHE_FILE)
        for topic_id in range(seed_topic_num):
            result.append("--------- Topic {} ---------".format(topic_id))
            for ranking in loading_set:
                doc_id = sorted_index[ranking, topic_id]
                result.append(
                    "Document {} {} : {}".format(doc_id, format_digit(doc_topic[doc_id, topic_id]), campaigns[doc_id]))
    else:
        result.append("              {}".format(" ".join(["Topic " + str(i) for i in range(seed_topic_num)])))
        for doc_id in range(10):
            result.append("Document {} : {}".format(str(doc_id), " ".join(
                ["  " + str(format_digit(doc_topic[doc_id][topic_id])) for topic_id in range(seed_topic_num)])))
    save_file(result, topic_num, 'document_loading')


def seed_word_count(X, word2id, seed_topic_list):
    word_frequence = np.sum(X, axis=0)
    result = []
    for tid, words in enumerate(seed_topic_list.values()):
        word_count = {}
        for word in words:
            if word in word2id.keys():
                word_count[word] = word_frequence[word2id[word]]
            else:
                word_count[word] = 0
        result.append(pretty_print_loading(tid, word_count.keys(), word_count.values()))
    save_file(result, '', 'seed_wordcount')


def top_k_word(X, top_k, vocab):
    word_frequence = np.sum(X, axis=0)
    word_index = np.argsort(word_frequence)[:-(top_k + 1): -1]
    word_count = word_frequence[word_index]
    word_list = [vocab[i] for i in word_index]
    print(
        'Top-k frequent word: {}'.format(','.join(
            list(map(lambda x, y: str(x) + '(' + str(format_digit(y)) + ')', word_list, word_count)))))


def retrieve_words_from(model, vocab, topic_num, n_top_words):
    """
    Retrieve the top k topics
    """
    topic_word = model.topic_word_
    result = []
    for tid, topic_dist in enumerate(topic_word):
        word_index = np.argsort(topic_dist)[:-(n_top_words + 1): -1]
        topic_words = np.array(vocab)[word_index]
        topic_words_assignment = topic_dist[word_index]
        result.append(pretty_print_loading(tid, topic_words, topic_words_assignment))
    save_file(result, topic_num, 'topic_words_matrix')


def pretty_print_loading(tid, topic_words, topic_words_assignment):
    return 'Topic {} : {}'.format(tid, ','.join(
        list(map(lambda x, y: str(x) + '(' + str(format_digit(y)) + ')', topic_words, topic_words_assignment))))


def export_seeds_assignment(model, word2id, topic_num, seed_topic_list):
    topic_word = model.topic_word_
    result = []
    for tid, seeds in enumerate(seed_topic_list.values()):
        loading = {}
        for word in seeds:
            if word in word2id.keys():
                loading[word] = topic_word[tid][word2id[word]]
            else:
                loading[word] = 0
        result.append(pretty_print_loading(tid, loading.keys(), loading.values()))
    save_file(result, topic_num, 'seeds_loading')


def export_loading(type):
    seed_topic_list = load_seed_words()
    X, word2id, vocab = load_data(type)
    if type == 'combination':
        model = guided_analysis(X, word2id, 150, SEED_CONFIDENCE, seed_topic_list)
        calculate_loading(model, 150, seed_topic_list)
        return export_loading_to_excel(model, seed_topic_list)
    elif type == 'transcription':
        model = guided_analysis(X, word2id, 50, SEED_CONFIDENCE, seed_topic_list)
        calculate_loading(model, 50, seed_topic_list)
        return export_transcription_loading(model, seed_topic_list)


def export_transcription_loading(model, seed_topic_list):
    data_list = read_json(RAW_TRANSCRIPTION_CACHE_FILE)
    nar_doc_topic = model.doc_topic_
    output_data = []
    keys = ['ProjectID'] + TRANSCRIPTION_COLUMN
    seed_topic_num = len(seed_topic_list.keys())
    for doc_id in range(len(data_list)):
        project_id = data_list[doc_id]['id']
        row = [project_id]
        for topic_id in range(seed_topic_num):
            row.append(format_digit(nar_doc_topic[doc_id][topic_id]))
        output_data.append(row)
    pd.DataFrame(output_data).to_excel('nar_loading.xlsx', header=keys, index=False)


def match_KsProject():
    seed_topic_list = load_seed_words()
    X, word2id, vocab = load_data('transcription')
    model = guided_analysis(X, word2id, 50, SEED_CONFIDENCE, seed_topic_list)
    calculate_loading(model, 50, seed_topic_list)
    ks_projects_list = read_excel(KS_PROJECT_FILE_PATH, 'Sheet1')
    ks_pids = ks_projects_list['ProjectId'].values
    data_list = read_json(RAW_TRANSCRIPTION_CACHE_FILE)
    nar_doc_topic = model.doc_topic_
    pid2row = {}
    for i, project in enumerate(data_list):
        pid2row[project['id']] = i
    keys = ['ProjectID'] + TRANSCRIPTION_COLUMN
    seed_topic_num = len(seed_topic_list.keys())
    output_data = []
    for pid in ks_pids:
        row = [pid]
        pid = str(pid)
        if pid in pid2row.keys():
            for topic_id in range(seed_topic_num):
                row.append(format_digit(nar_doc_topic[pid2row[pid]][topic_id]))
        else:
            row.extend([0] * seed_topic_num)
        output_data.append(row)
    pd.DataFrame(output_data).to_excel('matched_nar_loading.xlsx', header=keys, index=False)

def match_KsProject_order_loading(filename, outout_data):
    keys = ['ProjectID'] + LOADING_COLUMN
    new_output_data = []
    ks_projects_list = read_excel(KS_PROJECT_FILE_PATH, 'Sheet1')
    ks_pids = ks_projects_list['ProjectId'].values
    for pid in ks_pids:
        pid = str(pid)
        new_output_data.append(outout_data[pid])
    pd.DataFrame(new_output_data).to_excel('matched_ksPoject_loading.xlsx', header=keys, index=False)


def export_loading_to_excel(model, seed_topic_list):
    data_list = read_campaigns_from_path(CAMPAIGN_FILE_PATH)

    des_data_list = read_json(RAW_CAMPAIGN_CACHE_FILE)
    nar_data_list = read_json(RAW_TRANSCRIPTION_CACHE_FILE)

    des_pid2doc = {}
    nar_pid2doc = {}

    des_doc_num = len(des_data_list)
    des_doc_topic = model.doc_topic_[0: des_doc_num, :]
    nar_doc_topic = model.doc_topic_[des_doc_num: , :]
    column_num = len(LOADING_COLUMN)
    seed_topic_num = len(seed_topic_list.keys())
    output_data = {}
    keys = ['ProjectID'] + LOADING_COLUMN
    # initialize rows for excel
    for doc_id in range(len(data_list)):
        project_id = data_list[doc_id]['ProjectId']
        row = [project_id] + [0.0] * column_num
        output_data[project_id] = row
    i = 0
    # setting campaign loading
    for doc_id in range(des_doc_topic.shape[0]):
        project_id = des_data_list[doc_id]['ProjectId']
        des_pid2doc[project_id] = doc_id
        if project_id in output_data.keys():
            i += 1
            for topic_id in range(seed_topic_num):
                output_data[project_id][topic_id + 1] = float(format_digit(des_doc_topic[doc_id][topic_id]))
    print("Total descriptions: " + str(i))
    i = 0
    # setting transcription loading and consistency
    for doc_id in range(nar_doc_topic.shape[0]):
        project_id = nar_data_list[doc_id]['id']
        nar_pid2doc[project_id] = doc_id
        if project_id in output_data.keys():
            i += 1
            for topic_id in range(seed_topic_num):
                output_data[project_id][5 + topic_id] = float(format_digit(nar_doc_topic[doc_id][topic_id]))

    # setting the similarity
    for doc_id in range(len(data_list)):
        project_id = data_list[doc_id]['ProjectId']
        if project_id in nar_pid2doc.keys() and project_id in des_pid2doc.keys():
            des_loading = des_doc_topic[des_pid2doc[project_id]]
            nar_loading = nar_doc_topic[nar_pid2doc[project_id]]
            output_data[project_id][9] = float(
                format_digit(1 - spatial.distance.cosine(des_loading, nar_loading)))
    print("Total transcriptions: " + str(i))
    pd.DataFrame(output_data.values()).to_excel('loading.xlsx', header=keys, index=False)
    return output_data


def dt_matrix(model, n_top_docs, topic_num):
    """
    Export the document - topic matrix
    """
    doc_topic = model.doc_topic_
    result = []
    for i, docs_dist in enumerate(doc_topic):
        doc_topic_assignment = np.sort(docs_dist)[: -(n_top_docs + 1): -1]
        result.append('Document {} : {}'.format(i, ','.join(map(str, doc_topic_assignment))))
    save_file(result, topic_num, 'document_topic_matrix')


def tw_matrix(model, n_top_words, topic_num):
    """
    Export the topic- word matrix
    """
    topic_word = model.topic_word_
    result = []
    for i, word_dist in enumerate(topic_word):
        topic_word_assignment = np.sort(word_dist)[: -(n_top_words + 1): - 1]
        result.append('Topic {} : {}'.format(i, ','.join(map(str, topic_word_assignment))))
    save_file(result, topic_num, 'topic_word_matrix')


def unique_words(model, vocab, n_top_words, topic_num):
    """
    Find out the unique words for the topics
    """
    word_topic = model.word_topic_
    result = []
    for i, topic_dist in enumerate(word_topic):
        beta_dist = np.array(list(map(lambda x: float(x) / (1 - x), topic_dist)))
        # pick n_top beta for the words in the topic
        sorted_index = np.argsort(beta_dist)[: -(n_top_words + 1): - 1]
        sorted_beta = beta_dist[sorted_index]
        result.append(('{} : {}'.format(vocab[i], ','.join(
            list(map(lambda x, y: str(x) + '(' + str(y) + ')', sorted_index, sorted_beta))))))
    save_file(result, topic_num, 'word_uniqueness_matrix')


def load_seed_topics(word2id, seed_topic_list):
    """
    Construct the seeds_topic dictionary
    :param word2id:
    :return:
    """
    seed_topics = {}
    for tid, seeds in enumerate(seed_topic_list.values()):
        for word in seeds:
            lower_word = word.lower()
            if lower_word in word2id.keys():
                seed_topics[word2id[lower_word]] = tid
    return seed_topics


def format_digit(input):
    return '{0:.3f}'.format(input)


def get_wordnet_pos(treebank_tag):
    """
    get part of speech from tree bank ag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def load_stopwords(filepath):
    """
    Load the stop words
    """
    stop_words = set(stopwords.words('english'))
    with open(filepath) as fp:
        for line in fp:
            stop_words.add(line.rstrip('\n'))
    return stop_words


def load_seed_words():
    """
    Load the seed words
    """
    word_list = read_file(SEED_WORDS_FILE_PATH)
    word_dict = {}
    for tid, words in enumerate(word_list):
        word_dict[tid] = [word.strip() for word in words.split(',')]
    return word_dict


def filter_words(tokens, stop_words):
    """
    filter the word by nltk stopwords and length
    """
    return [w for w in tokens if w not in stop_words and len(w) > 3]


def clean_text(text):
    text = re.sub(r"\S*@\S*", " ", text)  # remove email address
    text = re.sub(r"((:?http|https)://)?[-./?:@_=#\w]+\.(?:[a-zA-Z]){2,6}(?:[-\w.&/?:@_=#()])*", " ",
                  text)  # remove urls
    text = re.sub(r"[-!?=~|#$+%*&@:/(){}\[\],\"\n'._]", " ", text)  # remove punctuations
    text = re.sub(r"\d+", " ", text)  # remove digits
    text = re.sub(r"\b(\w)\1+\b", " ", text)  # remove meaningless word composed

    return text


def nltk_lemmatize(campaign_list):
    """
    Return the campaign list after being lemmatized
    """
    result = []
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    stop_words = load_stopwords(STOP_WORDS_FILE_PATH)
    for campaign in campaign_list:
        token_list = []
        tokens = filter_words(nltk.word_tokenize(campaign, language='english'), stop_words)
        pos_tags = nltk.pos_tag(tokens)
        for word, tag in pos_tags:
            token_list.append(wordnet_lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)))
        result.append(" ".join(token_list))
    return result


def stanford_lemmatize(campaign_list):
    """
    Return the campaign list after being lemmatized by stanford nlp
    """
    result = []
    nlp = StanfordCoreNLP(path_or_host='http://localhost', port=9000, timeout=30000)
    stop_words = load_stopwords(STOP_WORDS_FILE_PATH)

    # Define properties needed to get lemma
    props = {'annotators': 'pos,lemma',
             'pipelineLanguage': 'en',
             'outputFormat': 'json'}
    for campaign in campaign_list:
        parsed_str = nlp.annotate(campaign, properties=props)
        parsed_dict = json.loads(parsed_str)
        sen_len = len(parsed_dict['sentences'])
        if sen_len == 0:
            result.append("")
        elif sen_len == 1:
            lemma_tokens = [token['lemma'] for token in parsed_dict['sentences'][0]['tokens']]
            filtered_list = filter_words(lemma_tokens, stop_words)
            result.append(' '.join(filtered_list))
        else:
            print("Length of sentence of lemmatization is greater than 1 ")
    return result


# * -------------- Utils function ------------ *


# def save_file(data, topic_num, filename):
#     with open('%s_%s.txt' % (str(topic_num), filename), 'w') as fp:
#         for item in data:
#             fp.write(item + "\n")
#         fp.close()


def save_file(data, topic_num, filename):
    for item in data:
        print(item)


def save_cache_file(data, filename):
    with open(filename, 'w') as fp:
        for item in data:
            fp.write(item + "\n");
        fp.close()


def read_file(filename):
    with open(filename, 'r') as fp:
        data_list = [line.rstrip('\n') for line in fp]
        fp.close()
    return data_list


def read_json(filename):
    with open(filename, 'r') as fp:
        data_list = list(json.load(fp))
        fp.close()
    return data_list


def read_campaigns_from_path(path):
    duplicate_data = []
    for file_name in [f for f in glob.glob(path + "**/*.json", recursive=True)]:
        duplicate_data.extend(read_json(file_name))
    # apply filtering for the data
    data_list = {}
    for item in duplicate_data:
        pid = item['ProjectId']
        if pid not in data_list.keys():
            data_list[pid] = item
    return list(data_list.values())


def dump_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)
        fp.close()


def format_str(input):
    return re.sub(r'[^\x00-\x7F]+', ' ', str(input.encode('utf-8')).strip())


def read_transcription(filename):
    df = pd.read_excel(filename, sheet_name='results')
    data = []
    ids = df['id'].values
    transcriptions = df['transcription'].values
    for index in range(df.shape[0]):
        project = {'id': str(ids[index]),
                   'transcription': '' if pd.isna(transcriptions[index]) else format_str(str(transcriptions[index]))}
        data.append(project)
    return data


def read_transcriptions_from_path(path):
    pid_set = set()
    data = []
    for file_name in [f for f in glob.glob(path + "**/*.xlsx", recursive=True)]:
        for project in read_transcription(file_name):
            if project['id'] not in pid_set:
                data.append(project)
                pid_set.add(project['id'])
    return data


def read_ocr_text(filename):
    data_list = []
    with open(filename, 'r') as fp:
        data_json = json.load(fp)
        for pid in data_json.keys():
            data_list.append({'id': pid, 'ocr_text': " ".join(data_json[pid])})
    return data_list


def read_ocr_texts_from_path(path):
    data_list = []
    pid_set = set()
    for file_name in [f for f in glob.glob(path + "**/*.json", recursive=True)]:
        for project in read_ocr_text(file_name):
            if project['id'] not in pid_set:
                data_list.append(project)
                pid_set.add(project['id'])
    return data_list

def format_compaign():
    data_list = read_json(RAW_COMBINATION_DATA_CACHE_FILE)
    for project in data_list:
        if 'ProjectCampaign' in project.keys():
            project['ProjectCampaign'] = (" ".join(project['ProjectCampaign'])).replace('\n', '')
    dump_json(data_list, 'formatted_campaign.json')


def read_excel(path, sheet_name = None):
    df = pd.read_excel(path, sheet_name)
    return df


def combine(campaign_list, ocr_text_list):
    pid2index = {}
    for index, project in enumerate(campaign_list):
        pid2index[project['ProjectId']] = index
    for project in ocr_text_list:
        pid = project['id']
        if pid in pid2index.keys():
            campaign_list[pid2index[pid]]['ProjectCampaign'].append(project['ocr_text'])
        else:
            print("ERROR: ocr text with id doesn't exist in the campaign")
    return campaign_list


def get_cleaned_campaign(data_list):
    return [clean_text((" ".join(item['ProjectCampaign'])).lower()) for item in data_list]


def get_cleaned_transcription(data_list):
    return [clean_text(item['transcription'].lower()) for item in data_list]


def load_data(doc_type):
    """
    1. Load the data from file
    2. clean the text
    3. lemmatize the text
    4. extract the vocabulary from the documents
    5. convert the data to document - term matrix
    """
    document_list = []
    raw_document_list = []
    if doc_type == 'campaign':
        if CACHE:
            document_list = read_file(CAMPAIGN_CACHE_FILE)
        else:
            campaign_data_list = read_campaigns_from_path(CAMPAIGN_FILE_PATH)
            ocr_data_list = read_ocr_texts_from_path(OCR_FILE_PATH)
            data_list = combine(campaign_data_list, ocr_data_list)
            if data_list:
                cleaned_list = get_cleaned_campaign(data_list)
                lemmatized_list = stanford_lemmatize(cleaned_list)
                # lemmatized_list = nltk_lemmatize(cleaned_list)
                for i, campaign in enumerate(lemmatized_list):
                    if len(campaign) > 0:
                        document_list.append(campaign)
                        raw_document_list.append(data_list[i])
                save_cache_file(document_list, CAMPAIGN_CACHE_FILE)
                dump_json(raw_document_list, RAW_CAMPAIGN_CACHE_FILE)
    elif doc_type == 'transcription':
        if CACHE:
            document_list = read_file(TRANSCRIPTION_CACHE_FILE)
        else:
            data_list = read_transcriptions_from_path(TRANSCRIPTION_FILE_PATH)
            cleaned_list = get_cleaned_transcription(data_list)
            lemmatized_list = stanford_lemmatize(cleaned_list)
            for i, transcription in enumerate(lemmatized_list):
                if len(transcription) > 0:
                    document_list.append(transcription)
                    raw_document_list.append(data_list[i])
            save_cache_file(document_list, TRANSCRIPTION_CACHE_FILE)
            dump_json(raw_document_list, RAW_TRANSCRIPTION_CACHE_FILE)
    elif doc_type == 'combination':
        if CACHE:
            document_list = read_file(COMBINATION_DATA_CACHE_FILE)
        else:
            campaign_data_list = combine(read_campaigns_from_path(CAMPAIGN_FILE_PATH),
                                         read_ocr_texts_from_path(OCR_FILE_PATH))
            transcription_data_list = read_transcriptions_from_path(TRANSCRIPTION_FILE_PATH)
            data_list = campaign_data_list + transcription_data_list
            cleaned_list = get_cleaned_campaign(campaign_data_list) + get_cleaned_transcription(transcription_data_list)
            lemmatized_list = stanford_lemmatize(cleaned_list)
            for i, document in enumerate(lemmatized_list):
                if len(document) > 0:
                    document_list.append(document)
                    raw_document_list.append(data_list[i])
            save_cache_file(document_list, COMBINATION_DATA_CACHE_FILE)
            dump_json(raw_document_list, RAW_COMBINATION_DATA_CACHE_FILE)
    vectorizer = CountVectorizer(stop_words=None, ngram_range=(1, 1),
                                 lowercase=True, analyzer='word')
    X = vectorizer.fit_transform(document_list).toarray()
    word2id = vectorizer.vocabulary_
    vocab = vectorizer.get_feature_names()
    return X, word2id, vocab


if __name__ == '__main__':
    # format_compaign()
    match_KsProject_order_loading(KS_PROJECT_FILE_PATH, export_loading('combination'))