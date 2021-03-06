!pip -qq install stanza

import stanza
import os
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,pos', verbose=False)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import stanza


class Preprocess:
    def __init__(self, mode, data):
        self.data = data
        self.mode = mode

    def tokenize(self, line):
        doc = nlp(line)
        text, pos = [], []
        for sentence in doc.sentences:
            for word in sentence.words:
                text.append(word.text)
                pos.append(word.xpos)
        return text, pos

    def tag_prep(self, dict, text):
        line, cause, effect = dict['sentence'], dict['cause'], dict['effect']
        d = defaultdict(list)
        index = 0
        for idx, w in enumerate(text):
            index = line.find(w, index)
            if not index == -1:
                d[idx].append([w, index])
                index += len(w)

        d_ = defaultdict(list)

        def cut_space(init_t):
            for s_idx, s in enumerate(line[init_t:]):
                if s != ' ':
                    init_t += s_idx
                    return init_t

        init_c = cut_space(line.find(cause))
        init_e = cut_space(line.find(effect))

        cause_list, _ = self.tokenize(cause) if (cause != '' and effect != '') else ([], False)
        effect_list, _ = self.tokenize(effect) if (cause != '' and effect != '') else ([], False)
        # print(line)
        # print('cause:', cause_list)
        # print('effect:', effect_list)
        # print(d)
        for idx in d:
            d_[idx].append([tuple([d[idx][0][0], '_']), d[idx][0][1]])
            init_c = cut_space(line.find(cause.strip()))
            init_e = cut_space(line.find(effect.strip()))
            for (c_idx, c) in enumerate(cause_list):
                start = line.find(c, init_c)
                stop = line.find(c, init_c) + len(c)
                word = line[start:stop]
                if int(start) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    if c_idx == 0:
                        und_[idx].append([tuple([word, 'B-C']), line.find(word, init_c)])
                    else:
                        und_[idx].append([tuple([word, 'I-C']), line.find(word, init_c)])
                    d_[idx] = und_[idx]
                    break
                init_c = cut_space(init_c + len(word))
            for (e_idx, e) in enumerate(effect_list):
                start = line.find(e, init_e)
                stop = line.find(e, init_e) + len(e)
                word = line[start:stop]
                # print(start, stop, word)
                if int(start) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    if e_idx == 0:
                        und_[idx].append([tuple([word, 'B-E']), line.find(word, init_e)])
                    else:
                        und_[idx].append([tuple([word, 'I-E']), line.find(word, init_e)])
                    d_[idx] = und_[idx]
                    break
                init_e = cut_space(init_e + len(word))
        tag = []
        for idx in d_:
            tag.append(d_[idx][0][0][1])
        # print('tags:', tag)
        return tag

    def write_file(self, filename, data):
        mode = self.mode
        with open(filename, 'w', encoding='utf-8') as f:
            if mode == 'test':
                for t, p in zip(data[0], data[1]):
                    for i in range(len(t)):
                        f.write(t[i] + ' ' + p[i] + '\n')
                    f.write('\n')
            elif mode == 'train':
                for t, p, ta in zip(data[0], data[1], data[2]):
                    for i in range(len(t)):
                        f.write(t[i] + ' ' + p[i] + ' ' + ta[i] + '\n')
                    f.write('\n')
        return filename

    def prepare(self):
        data = self.data.copy()
        mode = self.mode
        data = data.values

        lodict, multi_ = [], {}
        for idx, row in enumerate(data):
            if isinstance(row[0], str) and row[0].count('.') == 2:
                root_idx = '.'.join(row[0].split('.')[:-1])
                if root_idx in multi_:
                    multi_[root_idx].append(idx)
                else:
                    multi_[root_idx] = [idx]
            if mode == 'test':
                lodict.append({'sentence': row[1]})
            elif mode == 'train':
                lodict.append({'sentence': row[1], 'cause': row[2], 'effect': row[3]})
        try:
            print('transformation example: ', lodict[2])
            # print('multi items: ', multi_)
        except:
            pass

        text, pos, tag = [], [], []
        for idx, d in enumerate(tqdm(lodict)):
            text_, pos_ = self.tokenize(d['sentence'])
            text.append(text_)
            pos.append(pos_)
            if mode == 'train':
                # Making BIO tags
                tag_ = self.tag_prep(d, text_)
                tag.append(tag_)
        for (key, idxs) in multi_.items():
            # print(key, idxs)
            for i, idx in enumerate(idxs):
                text[idx] = [str(i)] + text[idx]
                pos[idx] = ['CD'] + pos[idx]
                if mode == 'train':
                    tag[idx] = ['_'] + tag[idx]
        if mode == 'train':
            self.write_file('train.txt', (text, pos, tag))
        elif mode == 'test':
            self.write_file('test.txt', (text, pos))

clf = Preprocess('train', pd.read_csv('/content/train.csv', delimiter=';'))
clf.prepare()

clf2 = Preprocess('test', pd.read_csv('/content/test.csv', delimiter=';'))
clf2.prepare()