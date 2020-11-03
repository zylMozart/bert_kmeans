import random as rd
from .base import *
from .tens import *
from .eval import *

__all__ = [
	'KMeans',
	'KMeansEuc',
	'KMeansCos',

	'PCEFKMeans',
	'PCEFKMeansEuc',
	'PCEFKMeansCos',
]

class KMeans(Obj):
	def __init__(self, x, y, *, dds=NA, wcs=NA, ncs=NA,
			dtr=NA, dti=NA, dev=NA, eps=NA, p=NA):
		dds = nav(dds, 200)
		wcs = nav(wcs, 1e-4)
		ncs = nav(ncs, 10)
		eps = nav(eps, EPS)
		dtr = nav(dtr, DTR)
		dti = nav(dti, DTI)
		dev = nav(dev, DEV)

		self.eps = eps
		self.dtr = dtr
		self.dti = dti
		self.dev = dev
		self.p = p

		self.x = tens(x, dt=self.dtr, dev=self.dev)
		self.y = tens(y, dt=self.dti, dev=self.dev)

		self.n, self.m = self.x.shape
		self.k = self.y.max().tolist()+1

		self.dds = dds
		self.wcs = wcs*self.n
		self.ncs = ncs
		self.fmt = (f'{self.typename()}: '+', '.join([
			f'dd={{}}/{self.dds}',
			f'wc={{}}/{self.wcs}',
			f'nc={{}}/{self.ncs}',
		]))

		self.u = NA
		self.v = NA
		self.dd = NA
		self.wc = NA
		self.nc = NA

		self.tu = NA
		self.twc = NA

		self.scores = NA

	def __call__(self):
		self.init()
		while self.dd<self.dds and self.wc>self.wcs and self.nc<self.ncs:
			if self.p: print(self.fmt.format(self.dd, self.wc, self.nc))
			self.backup()
			self.update()
		if self.p:
			print(self.fmt.format(self.dd, self.wc, self.nc))
			print('='*64)
		return self

	def set(self, *args, **kwargs):
		return self

	def init(self):
		self.init_dd()
		self.init_wc()
		self.init_nc()
		self.init_v()
		self.init_u()
	def update(self):
		self.update_v()
		self.update_u()
		self.update_wc()
		self.update_nc()
		self.update_dd()
	def backup(self):
		self.tu = self.u
		self.twc = self.wc

	def init_u(self):
		self.u = NA
		self.update_u()
	def init_v(self):
		# r = []
		# p = tens1(self.n, dt=self.dtr, dev=self.dev)
		# a = list(range(self.n))
		# for k in range(self.k):
		# 	t = rd.choices(a, p.tolist())[0]
		# 	r.append(t)
		# 	p *= 1-self.sim[t].to(self.dev)
		r = rd.sample(range(self.n), self.k)
		self.v = self.x[r]
	def init_dd(self):
		self.dd = 0
	def init_wc(self):
		self.wc = self.n
	def init_nc(self):
		self.nc = 0

	def get_d(self, a, b):
		pass

	def update_u(self):
		d = self.get_d(self.x, self.v)
		d, u = d.min(dim=1)
		d, l = d.tolist(), u.tolist()
		if len(set(l))<self.k:
			s = mulset(l)
			t = sort(((i, d[i]) for i in range(self.n)), k=-1)
			for k in range(self.k):
				if k not in s:
					while TRUE:
						i, _ = t.pop()
						tk = l[i]
						if s[tk]>=2:
							u[i] = k
							s[tk] -= 1
							break
		self.u = u
	def update_v(self):
		v = tensn(self.v)
		for i in range(self.k):
			v[i] = self.x[self.u==i].mean(dim=0)
		self.v = v
	def update_dd(self):
		self.dd += 1
	def update_wc(self):
		self.wc = (self.u!=self.tu).sum().tolist()
	def update_nc(self):
		if equal(self.wc, self.twc, eps=0.1): self.nc += 1
		else: self.nc = 0

	def eval(self):
		self.scores = get_scores(self.u, self.y,
			self.n, self.k, self.dtr, self.dev)
		return self
class KMeansEuc(KMeans):
	def get_d(self, a, b):
		return tens_dist_ssq(a, b)
class KMeansCos(KMeans):
	def __init__(self, x, y, *args, **kwargs):
		KMeans.__init__(self, x, y, *args, **kwargs)
		self.x = tens_unit(self.x, dim=1)
	def get_d(self, a, b):
		return 1-a.matmul(b.t())
	def update_v(self):
		KMeans.update_v(self)
		self.v = tens_unit(self.v, dim=1)

class PCEFKMeans(KMeans):
	def __init__(self, x, y, *args, a=NA, b=NA, w=NA, **kwargs):
		KMeans.__init__(self, x, y, *args, **kwargs)
		a = nav(a, 0)
		b = nav(b, 1)
		self.a = NA
		self.b = NA
		self.w = NA
		self.set(a=a, b=b, w=w)

	def set(self, *args, a=NA, b=NA, w=NA, **kwargs):
		KMeans.set(self, *args, **kwargs)
		if nna(a):
			if equal(a): a = self.eps
			self.a = a
		if nna(b):
			self.b = b
		if nna(w):
			self.w = NA
			if not equal(w):
				n = iround(self.n*w)
				y = self.y[:n].tolist()
				pos, neg = pair_pos_neg(y)
				td = TensData(pos, 1, s=TRUE)+TensData(neg, -1, s=TRUE)
				w = tens0((self.n, self.n), dt=self.dtr, dev=self.dev)
				self.w = td(w)
		return self

	def update_u(self):
		d = self.get_d(self.x, self.v)
		t = -d
		if nna(self.u) and not equal(self.b) and nna(self.w):
			t += self.w.matmul(self.u)*self.b
		t = t*(self.k*log(self.k)/self.a)
		m, _ = t.max(dim=1, keepdim=TRUE)
		t = t-m
		t = t.exp()
		u = tens_rate(t, dim=1)

		cn = u.sum(dim=0)
		cnz = cn<self.eps
		if cnz.sum()>0:
			l = sort(zip(range(self.n), m.tolist()), k=-1)
			for k in range(self.k):
				if cnz[k]:
					while TRUE:
						i, _ = l.pop(0)
						tcn = cn-u[i]
						if (tcn[1-cnz]<self.eps).sum()<=0:
							cn = tcn
							cn[k] = 1
							cnz[k] = FALSE
							u[i] = 0
							u[i][k] = 1
							break

		self.u = u
	def update_v(self):
		t = self.u.t().matmul(self.x)
		self.v = tens_unit(t, dim=1)
	def update_wc(self):
		self.wc = ((self.u-self.tu).abs().sum()/2).tolist()

	def eval(self):
		self.scores = get_scores(self.u.argmax(dim=1), self.y,
			self.n, self.k, self.dtr, self.dev)
		return self
class PCEFKMeansEuc(PCEFKMeans):
	def get_d(self, a, b):
		return KMeansEuc.get_d(self, a, b)
	def update_v(self):
		t = self.u.t().matmul(self.x)
		self.v = t/self.u.t().sum(dim=1, keepdim=TRUE)
class PCEFKMeansCos(PCEFKMeans):
	def __init__(self, x, y, *args, a=NA, b=NA, w=NA, **kwargs):
		PCEFKMeans.__init__(self, x, y, *args, a=a, b=b, w=w, **kwargs)
		self.x = tens_unit(self.x, dim=1)
	def get_d(self, a, b):
		return KMeansCos.get_d(self, a, b)
	def update_v(self):
		t = self.u.t().matmul(self.x)
		self.v = tens_unit(t, dim=1)
