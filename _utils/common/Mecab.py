import MeCab
import re

class Mecab():
    def parse(self,sentence):
        me = MeCab.Tagger()
        s = me.parse(sentence)
        return s

    def pos(self, s):
        s = self.parse(s)
        word_tag = []
        for r in s.split('\n'):
            p = r.split('\t')
            if len(p) > 1:
                w, o = p
                t = o.split(',')[0]
                word_tag.append((w, t))
        return word_tag

    def morphs(self,s):
        s = self.parse(s)
        morph_tag = list()
        for r in s.split('\n'):
            p = r.split('\t')
            if len(p)>1:
                w,o = p
                morph_tag.append(w)
        return morph_tag

    def nouns(self,s):
        s = self.parse(s)
        noun_tag = list()
        k = re.compile('^N')
        for r in s.split('\n'):
            p = r.split('\t')
            if len(p)>1:
                w,o = p
                t = o.split(',')[0]
                if k.match(t) is not None:
                    noun_tag.append(w)
        return noun_tag

