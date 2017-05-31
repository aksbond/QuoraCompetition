import nltk
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import re

tagset = ['CC',	'CD',	'DT',	'EX',	'FW',	'IN',	'JJ',	'JJR',	'JJS',	'LS',	'MD',	'NN',	'NNS',	'NNP',	'NNPS',	'PDT',	'POS',	'PRP',	'PRP$',	'RB',	'RBR',	'RBS',	'RP',	'SYM',	'TO',	'UH',	'VB',	'VBD',	'VBG',	'VBN',	'VBP',	'VBZ',	'WDT',	'WP',	'WP$',	'WRB', ".", "$", "''"]
tagset = {k:[] for k in tagset}


def get_pos_tags(input_string):
  s1,s2 = input_string.split("#^#")
  s1w = re.findall('\w+', s1)
  s2w = re.findall('\w+', s2)
  s1w_tags = nltk.pos_tag(s1w)
  s2w_tags = nltk.pos_tag(s2w)

  s1_dict = deepcopy(tagset)
  for k, v in s1w_tags:
    s1_dict[v].append(k.lower())

  s2_dict = deepcopy(tagset)
  for k, v in s2w_tags:
    s2_dict[v].append(k.lower())

  common_keys = s1_dict.keys() & s2_dict.keys()

  shared_items = [len(list(filter(lambda x: x in set(s2_dict[k]), s1_dict[k]))) for k in common_keys]
  shared_items_by_s1 = [0 if len(set(s1_dict[k])) == 0 else len(list(filter(lambda x: x in set(s2_dict[k]), s1_dict[k]))) / len(set(s1_dict[k])) for k in common_keys]
  shared_items_by_s2 = [0 if len(set(s2_dict[k])) == 0 else len(list(filter(lambda x: x in set(s2_dict[k]), s1_dict[k]))) / len(set(s2_dict[k])) for k in common_keys]

  return np.array((shared_items + shared_items_by_s1 + shared_items_by_s2))

with open('./Data/Feature sets/clean_data.pkl', 'rb') as input:
    full_data = pickle.load(input)


a = ['CC',	'CD',	'DT',	'EX',	'FW',	'IN',	'JJ',	'JJR',	'JJS',	'LS',	'MD',	'NN',	'NNS',	'NNP',	'NNPS',	'PDT',	'POS',	'PRP',	'PRP_dlr',	'RB',	'RBR',	'RBS',	'RP',	'SYM',	'TO',	'UH',	'VB',	'VBD',	'VBG',	'VBN',	'VBP',	'VBZ',	'WDT',	'WP',	'WP_dlr',	'WRB', "dot", "dlr", "colon"]*3
pos = ["pos"]*len(a)
b = range(0,len(a))
cols = ["{}{}{}".format(pos_,a_,b_) for pos_,a_, b_ in zip(pos,a, b)]

pos_tags = pd.DataFrame(np.row_stack(full_data["q1_n_q2"].map(lambda x: get_pos_tags(x))),columns=cols)
print(pos_tags.shape)

del(full_data)
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object(pos_tags, './Data/Feature sets/pos_tags_cleandata30may.pkl')

