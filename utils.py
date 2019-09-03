# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/8/29
# __file__ = utils
# __desc__ =
from pathlib import Path


class WordCut(object):
    def __init__(self, wordfile):
        self.wordfile = wordfile
        self.worddict = self._worddict()
        self.reworddict = {v: k for k, v in self.worddict.items()}

    def _worddict(self):
        with open(self.wordfile, 'r', encoding='utf-8') as f:
            words = [word.strip('\n\t') for word in f]
        wordDict = {}
        for i, word in enumerate(words):
            wordDict[i] = word
        return wordDict

    def words2idxs(self, sentence):
        res = []
        for word in sentence:
            res.append(self.reworddict.get(word, 2))
        return res

    def idxs2words(self, idxs, end_token="<EOS>"):
        res = []
        for idx in idxs:
            word = self.worddict.get(idx)
            if word == end_token:
                return ' '.join(res)
            res.append(word)
        return ' '.join(res)

    def idx2word(self, idx):
        return self.worddict.get(idx, "<UNK>")

    def word2idx(self, word):
        return self.reworddict.get(word, 1)

    def __len__(self):
        return len(self.worddict)

    def add_word(self, word, idx=None):
        if idx == "append":
            end_key = max(self.worddict.keys())
            self.worddict[end_key + 1] = word

        elif idx is None:
            self.worddict = {k + 1: v for k, v in self.worddict.items()}
            self.worddict[0] = word
        else:
            tdict = {}
            for key, value in self.worddict.items():
                if idx - key == 1 or key >= idx:
                    tdict[key + 1] = value
                else:
                    tdict[key] = value
            tdict[idx - 1] = word
            self.worddict = tdict
            del tdict

        self.reworddict = {v: k for k, v in self.worddict.items()}
