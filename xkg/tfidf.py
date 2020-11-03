from .base import *
from .tens import *

__all__ = [
	'TFIDF',
]

class TFIDF(Obj):
	def __init__(self, x):
		n = len(x)
		x = [i.split() for i in x]
		df = []
		for i in x: df.extend(set(i))
		df = dict(sort(mulset(df)))
		m = len(df)
		w, df = list(df.keys()), list(df.values())
		wid = dict(zip(w, range(m)))
		idf = [log((n+1)/(i+1/m)) for i in df]
		tfidf = []
		for i in range(n):
			t = [wid[j] for j in x[i]]
			x[i] = t
			c = len(t)
			t = dict(sort(mulset(t)))
			t = {k: v/c*idf[k] for k, v in t.items()}
			tfidf.append(t)
		self.n = n
		self.m = m
		self.wid = wid
		self.tfidf = tfidf
	def __call__(self, t=NA, *args, dtr=NA, dev=NA, **kwargs):
		r = TensData(self.tfidf)
		if t:
			if na(dtr): dtr = DTR
			if na(dev): dev = DEV
			t = tens0((self.n, self.m), *args, dt=dtr, dev=dev, **kwargs)
			r = r(t)
		return r
