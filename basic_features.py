import pandas as pd
import re
import difflib
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import pickle
import numpy as np
import gensim


train_data = pd.read_csv("Data/train.csv", encoding = "ISO-8859-1")
train_data["train_ind"] = 1
train_data = train_data.drop(["qid1","qid2"], axis = 1)
test_data = pd.read_csv("Data/test.csv", encoding = "ISO-8859-1")
test_data.columns = ['id', 'question1', 'question2']
test_data["train_ind"] = 0
test_data["is_duplicate"] = -1
len(train_data)
len(test_data)

train_data.columns
test_data.columns
full_data = pd.concat([train_data, test_data], axis = 0).reset_index(drop = True)
len(full_data)
full_data.columns
""" Functions to be used """
### Stemming

stemmer = SnowballStemmer('english')
def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])



# str_stemmer("Adana Gallery Suri Square Hijab – Light Pink#^#<ul><li>Material : Non sheer shimmer chiffon<")
def count_sentences(input_string):
    sentences = re.split("[,.!?]+",input_string)
    words_per_sentence = [len(k.split(" ")) for k in sentences]
    return len(sentences), np.mean(words_per_sentence), max(words_per_sentence) 

def get_num_similar_words_in_str(input_string):
    s1,s2 = input_string.split("#^#")
    s1w = re.findall('\w+', s1.lower())
    s2w = re.findall('\w+', s2.lower())
    common_ratio = difflib.SequenceMatcher(None, s1w, s2w).ratio()
    unique_s1w = list(set(s1w))
    unique_s2w = list(set(s2w))
    unique_common_ratio = difflib.SequenceMatcher(None, unique_s1w, unique_s2w).ratio()

    stop = set(stopwords.words('english'))

    non_stop_s1w = [word for word in s1w if word not in stop]
    non_stop_s2w = [word for word in s2w if word not in stop]
    non_stop_common_ratio = difflib.SequenceMatcher(None, non_stop_s1w, non_stop_s2w).ratio()
    unique_non_stop_s1w = list(set(non_stop_s1w))
    unique_non_stop_s2w = list(set(non_stop_s2w))
    unique_non_stop_common_ratio = difflib.SequenceMatcher(None, unique_non_stop_s1w, unique_non_stop_s2w).ratio()

    return len(s1w), len(s2w), len(unique_s1w), len(unique_s2w), common_ratio, unique_common_ratio, len(non_stop_s1w), len(non_stop_s2w), len(unique_non_stop_s1w), len(unique_non_stop_s2w), non_stop_common_ratio, unique_non_stop_common_ratio


def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)



full_data.loc[full_data["question1"].isnull(), "question1"] = " "
full_data.loc[full_data["question2"].isnull(), "question2"] = " "

full_data["len_q1"] = full_data["question1"].map(lambda x: len(x))
full_data["len_q2"] = full_data["question2"].map(lambda x: len(x))

## Count number of sentences by counting number of periods "."
full_data["num_sen_q1"], full_data["wrd_per_sen_q1"], full_data["max_wrd_sen_q1"] = zip(*full_data["question1"].map(lambda x: count_sentences(x)))
full_data["num_sen_q2"], full_data["wrd_per_sen_q2"], full_data["max_wrd_sen_q2"] = zip(*full_data["question2"].map(lambda x: count_sentences(x)))

## Number of common words in title and description
full_data["q1_n_q2"] = full_data["question1"] + "#^#" + full_data["question2"]


full_data["wrd_in_q1"], full_data["wrd_in_q2"], full_data["unq_wrd_in_q1"], full_data["unq_wrd_in_q2"],  full_data["cmn_wrd_q1_q2"], full_data["unq_cmn_wrd_q1_q2"], full_data["non_stopwrd_q1"], full_data["non_stopwrd_q2"], full_data["unq_non_stopwrd_q1"], full_data["unq_non_stopwrd_q2"], full_data["cmn_non_stopwrd_q1_q2"], full_data["unq_cmn_non_stopwrd_q1_q2"]  = zip(*full_data["q1_n_q2"].map(lambda x: get_num_similar_words_in_str(x)) )
print("Similar words over")

## Derived features
full_data["stopwrd_q1"]  = full_data["wrd_in_q1"] - full_data["non_stopwrd_q1"]
full_data["stopwrd_ratio_q1"]  = full_data["non_stopwrd_q1"]/full_data["wrd_in_q1"]
full_data.loc[full_data["wrd_in_q1"] == 0, "stopwrd_ratio_q1"] = 0
full_data["stopwrd_q2"]  = full_data["wrd_in_q2"] - full_data["non_stopwrd_q2"]

full_data["stopwrd_ratio_q2"]  = full_data["non_stopwrd_q2"]/full_data["wrd_in_q2"]
full_data.loc[full_data.wrd_in_q2 ==0,"stopwrd_ratio_q2"] = 0


full_data["q1_unq_ratio"] = full_data["unq_non_stopwrd_q1"]/ full_data["non_stopwrd_q1"]
full_data.loc[full_data.non_stopwrd_q1 ==0,"q1_unq_ratio"] = 0

full_data["q2_unq_ratio"] = full_data["unq_non_stopwrd_q2"]/ full_data["non_stopwrd_q2"]
full_data.loc[full_data.non_stopwrd_q2 ==0,"q2_unq_ratio"] = 0

full_data["q1_q2_non_stopwrd_ratio"] = full_data["non_stopwrd_q1"]/full_data["non_stopwrd_q2"]
full_data.loc[full_data.non_stopwrd_q2 ==0,"q1_q2_non_stopwrd_ratio"] = 0

full_data["q1_q2_unq_non_stopwrd_ratio"] = full_data["unq_non_stopwrd_q1"]/full_data["unq_non_stopwrd_q2"]
full_data.loc[full_data.unq_non_stopwrd_q2 ==0,"q1_q2_unq_non_stopwrd_ratio"] = 0

full_data.describe()

print("Getting fuzzy ratios")
## Now get similarity scores between title and description
full_data['fuzz_qratio'] = full_data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
full_data['fuzz_WRatio'] = full_data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
full_data['fuzz_partial_ratio'] = full_data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
full_data['fuzz_partial_token_set_ratio'] = full_data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
full_data['fuzz_partial_token_sort_ratio'] = full_data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
full_data['fuzz_token_set_ratio'] = full_data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
full_data['fuzz_token_sort_ratio'] = full_data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

print(full_data.describe())

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# sample usage
save_object(full_data, './Data/Feature sets/Basic_features_wo_stem.pkl')



model = gensim.models.KeyedVectors.load_word2vec_format('./Data/GoogleNews-vectors-negative300.bin.gz', binary=True)
full_data['wmd'] = full_data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
print("First W2Vec done")
del(model)
norm_model = gensim.models.KeyedVectors.load_word2vec_format('./Data/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)
full_data['norm_wmd'] = full_data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
print("Second W2Vec done")


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# sample usage
save_object(full_data, './Data/Feature sets/Basic_features_wo_stem_W2vec.pkl')


