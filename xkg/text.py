from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from .base import *

__all__ = [
	'text_preproc',
]

def text_preproc(x):
	char_ascii = [ci(i) for i in range(128)]
	char_d = [ci(ic('0')+i) for i in range(10)]
	char_lu = [ci(ic('A')+i) for i in range(26)]
	char_ll = [ci(ic('a')+i) for i in range(26)]
	char_l = char_lu+char_ll
	char_luq = [ci(ic('Ａ')+i) for i in range(26)]
	char_llq = [ci(ic('ａ')+i) for i in range(26)]
	char_lq = char_luq+char_llq
	char_x = list('¡©«­¯°²´·¹º»×ˈ‎–—‘’“”„†•…™∈。・＊，？')
	char_q = dict(zip(char_lq, char_l))
	char_s = (set(char_ascii)-set(char_d+char_l))|set(char_x)
	word_k = {
		'NN': wordnet.NOUN,
		'VB': wordnet.VERB,
		'JJ': wordnet.ADJ,
		'RB': wordnet.ADV,
	}
	stop = stopwords.words('english')
	for s in x:
		t = ''.join(char_q.get(i, i) for i in s)
		t = word_tokenize(t)
		t = pos_tag(t)
		t = ((i, j[:2].upper()) for i, j in t)
		t = ((i, word_k[j]) for i, j in t if j in word_k)
		t = (wordnet.morphy(i, j) or i for i, j in t)
		t = ' '.join(t).lower()
		t = ((i if i not in char_s else ' ') for i in t)
		t = ''.join(t).split()
		t = (i for i in t if i not in stop)
		t = ' '.join(t)
		yield t
