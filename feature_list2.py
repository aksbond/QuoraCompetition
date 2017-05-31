import pandas as pd
import networkx as nx
import numpy as np
from nltk.corpus import stopwords
import pickle

train_data = pd.read_csv("Data/train.csv", encoding = "ISO-8859-1", nrows = 1000)
train_data["train_ind"] = 1
train_data = train_data.drop(["qid1","qid2"], axis = 1)
test_data = pd.read_csv("Data/test.csv", encoding = "ISO-8859-1", nrows = 1000)
test_data.columns = ['id', 'question1', 'question2']
test_data["train_ind"] = 0
test_data["is_duplicate"] = -1
len(train_data)
len(test_data)

train_data.columns
test_data.columns
full_data = pd.concat([train_data, test_data], axis = 0).reset_index(drop = True)
len(full_data)

g = nx.Graph()
g.add_nodes_from(full_data.question1)
g.add_nodes_from(full_data.question2)
edges = list(full_data[['question1', 'question2']].to_records(index=False))
g.add_edges_from(edges)


def get_intersection_count(row):
    return(len(set(g.neighbors(row.question1)).intersection(set(g.neighbors(row.question2)))))

full_data['intersection_count'] = full_data.apply(lambda row: get_intersection_count(row), axis=1)




train_qs = pd.Series(train_data['question1'].tolist() + train_data['question2'].tolist()).astype(str)
test_qs = pd.Series(test_data['question1'].tolist() + test_data['question2'].tolist()).astype(str)

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=3)
tfidf_matrix = tfidf_vectorizer.fit_transform(train_qs)
feature_names = tfidf_vectorizer.get_feature_names()
dense = tfidf_matrix.todense()
word_index_dict = dict((j, i) for i,j in enumerate(feature_names))


stops = set(stopwords.words("english"))
def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    q1_tfidf = tfidf_vectorizer.transform([" ".join(q1words.keys())])
    q2_tfidf = tfidf_vectorizer.transform([" ".join(q2words.keys())])
    inter = np.intersect1d(q1_tfidf.indices, q2_tfidf.indices)
    shared_weights = 0
    for word_index in inter:
        shared_weights += (q1_tfidf[0, word_index] + q2_tfidf[0, word_index])
    total_weights = q1_tfidf.sum() + q2_tfidf.sum()
    return np.sum(shared_weights) / np.sum(total_weights)

full_data["tfidf_wrd_match"] = full_data.apply(tfidf_word_match_share, axis=1, raw=True)


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# sample usage
save_object(full_data, './Data/Feature sets/tfidf_feature.pkl')



#####################################################################

df1 = train_data[['question1']].copy()
df2 = train_data[['question2']].copy()
df1_test = test_data[['question1']].copy()
df2_test = test_data[['question2']].copy()

df2.rename(columns = {'question2':'question1'},inplace=True)
df2_test.rename(columns = {'question2':'question1'},inplace=True)

train_questions = df1.append(df2)
train_questions = train_questions.append(df1_test)
train_questions = train_questions.append(df2_test)
#train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
train_questions.drop_duplicates(subset = ['question1'],inplace=True)

train_questions.reset_index(inplace=True,drop=True)
questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
train_cp = train_data.copy()
test_cp = test_data.copy()
#train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

test_cp['is_duplicate'] = -1
test_cp.rename(columns={'test_id':'id'},inplace=True)
comb = pd.concat([train_cp,test_cp])

comb['q1_hash'] = comb['question1'].map(questions_dict)
comb['q2_hash'] = comb['question2'].map(questions_dict)

q1_vc = comb.q1_hash.value_counts().to_dict()
q2_vc = comb.q2_hash.value_counts().to_dict()


def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0
#map to frequency space
comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','is_duplicate']]
test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq']]

len(train_comb)
len(test_comb)


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# sample usage
save_object(train_comb, './Data/Feature sets/train_comb.pkl')
save_object(test_comb, './Data/Feature sets/test_comb.pkl')


