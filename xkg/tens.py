import random as _random
import numpy as _numpy
import torch as _torch

from .base import *

DTR16 = _torch.float16
DTR32 = _torch.float32
DTR64 = _torch.float64

DTI8 = _torch.int8
DTI16 = _torch.int16
DTI32 = _torch.int32
DTI64 = _torch.int64

DTN8 = _torch.uint8
DTB8 = _torch.bool

DTR = DTR32
DTI = DTI32
DTB = DTN8
DEV = 'cpu'

def isTens(*args, **kwargs):
	return _torch.is_tensor(*args, **kwargs)
def _tens(x, dt=NA, dev=NA):
	t = isTens(x)
	if not t:
		if na(dt):
			x = get_0(x)
			if isReal(x): dt = DTR
			elif isInt(x): dt = DTI
			elif isBool(x): dt = DTB
		if na(dev): dev = DEV
	else:
		if na(dt): dt = x.dtype
		if na(dev): dev = x.device
	return t, dt, dev
def tens(x, *args, dt=NA, dev=NA, copy=NA, **kwargs):
	t, dt, dev = _tens(x, dt, dev)
	if not t:
		r = _torch.tensor(x, *args, dtype=dt, device=dev, **kwargs)
	else:
		if not copy: r = x
		else: r = x.clone()
		r = r.to(dtype=dt, device=dev)
	return r
def _tenss(f, x, *args, dt=NA, dev=NA, **kwargs):
	t, dt, dev = _tens(x, dt, dev)
	f = f[boolv(t, 0, 1)]
	return f(x, *args, dtype=dt, device=dev, **kwargs)
def tensn(x, *args, dt=NA, dev=NA, **kwargs):
	f = (_torch.empty, _torch.empty_like)
	return _tenss(f, x, *args, dt=dt, dev=dev, **kwargs)
def tens0(x, *args, dt=NA, dev=NA, **kwargs):
	f = (_torch.zeros, _torch.zeros_like)
	return _tenss(f, x, *args, dt=dt, dev=dev, **kwargs)
def tens1(x, *args, dt=NA, dev=NA, **kwargs):
	f = (_torch.ones, _torch.ones_like)
	return _tenss(f, x, *args, dt=dt, dev=dev, **kwargs)
def tensf(x, v, *args, dt=NA, dev=NA, **kwargs):
	f = (_torch.full, _torch.full_like)
	return _tenss(f, x, v, *args, dt=dt, dev=dev, **kwargs)
def tensr(x, *args, dt=NA, dev=NA, **kwargs):
	f = (_torch.rand, _torch.rand_like)
	return _tenss(f, x, *args, dt=dt, dev=dev, **kwargs)
def tensrn(x, *args, dt=NA, dev=NA, **kwargs):
	f = (_torch.randn, _torch.randn_like)
	return _tenss(f, x, *args, dt=dt, dev=dev, **kwargs)
def tensri(x, *args, dt=NA, dev=NA, **kwargs):
	f = (_torch.randint, _torch.randint_like)
	return _tenss(f, x, *args, dt=dt, dev=dev, **kwargs)
def tensa(x, *args, dt=NA, dev=NA, **kwargs):
	t, dt, dev = _tens(x, dt, dev)
	return _torch.arange(x, *args, dtype=dt, device=dev, **kwargs)
def tense(x, *args, dt=NA, dev=NA, **kwargs):
	t, dt, dev = _tens(x, dt, dev)
	return _torch.eye(x, *args, dtype=dt, device=dev, **kwargs)
def tensl(x, y, n, *args, dt=NA, dev=NA, **kwargs):
	t, dt, dev = _tens(x, dt, dev)
	return _torch.linspace(x, y, n, *args, dtype=dt, device=dev, **kwargs)

def tens_floor(x, w=NA):
	f = getattr(x.__class__, 'floor')
	return frc(f, x, w)
def tens_ceil(x, w=NA):
	f = getattr(x.__class__, 'ceil')
	return frc(f, x, w)
def tens_round(x, w=NA):
	f = getattr(x.__class__, 'round')
	return frc(f, x, w)

def tens_sq(x):
	return x**2
def tens_sum(x, dim=NA, keepdim=NA):
	dim = nav(dim, ())
	keepdim = nav(keepdim, FALSE)
	return x.sum(dim=dim, keepdim=keepdim)
def tens_ssq(x, dim=NA, keepdim=NA):
	dim = nav(dim, ())
	keepdim = nav(keepdim, FALSE)
	return (x**2).sum(dim=dim, keepdim=keepdim)
def tens_norm(x, dim=NA, keepdim=NA):
	return tens_ssq(x, dim=dim, keepdim=keepdim).sqrt()
def tens_rate(x, dim=NA, eps=NA):
	t = tens_sum(x, dim=dim, keepdim=TRUE)
	eps = nav(eps, EPS)
	return x/(t+eps)
def tens_unit(x, dim=NA, eps=NA):
	t = tens_norm(x, dim=dim, keepdim=TRUE)
	eps = nav(eps, EPS)
	return x/(t+eps)
def tens_dist_ssq(x, y=NA, sn=NA):
	sn = nav(sn, 64)
	y = nav(y, x)
	n, m = x.shape
	k, _ = y.shape
	a = x.view(n, 1, m).expand(n, k, m)
	b = y.view(1, k, m).expand(n, k, m)
	r = tensn((n, k), dt=x.dtype, dev=x.device)
	sb = 0
	if n<=k:
		ss = iceil(n/sn)
		for i in range(ss):
			if i+1<ss:
				t = a[sb:sb+sn]-b[sb:sb+sn]
				r[sb:sb+sn] = (t**2).sum(dim=2)
				sb += sn
			else:
				t = a[sb:]-b[sb:]
				r[sb:] = (t**2).sum(dim=2)
	else:
		ss = iceil(k/sn)
		for i in range(ss):
			if i+1<ss:
				t = a[:, sb:sb+sn]-b[:, sb:sb+sn]
				r[:, sb:sb+sn] = (t**2).sum(dim=2)
				sb += sn
			else:
				t = a[:, sb:]-b[:, sb:]
				r[:, sb:] = (t**2).sum(dim=2)
	return r
def tens_dist_norm(x, y=NA):
	return tens_dist_ssq(x, y).sqrt()

def tens_sel(x, t, dim=NA):
	dim = nav(dim, 1)
	n, m = x.shape
	dev = x.device
	t = tens(t, dt=DTI64, dev=dev)
	if dim==0: r = t*m+tensa(m, dt=DTI64, dev=dev)
	else: r = tensa(n, dt=DTI64, dev=dev)*m+t
	return r
def tens_sel_get(x, t, dim=NA):
	t = tens_sel(x, t, dim=dim)
	return x.flatten()[t]
def tens_sel_set(x, t, v, dim=NA):
	t = tens_sel(x, t, dim=dim)
	v = tens(v, dt=x.dtype, dev=x.device)
	x.flatten()[t] = v
	return x

def init_rand(seed):
	_random.seed(seed)
	_numpy.random.seed(seed)
	_torch.manual_seed(seed)
	_torch.cuda.manual_seed_all(seed)
def pair_pos_neg(y):
	t = {}
	for i, k in enumerate(y):
		if k<0: continue
		if k not in t: t[k] = []
		t[k].append(i)
	for k in t: t[k] = sort(t[k])
	t = CDict(t).sort().val(list)
	pos, neg, s = {}, {}, []
	for tn, tt in enumerate(t):
		tl = len(tt)
		ts = list(s)
		for n, i in enumerate(tt):
			if n+1<tl: pos[i] = tt[n+1:]
			if tn>0: neg[i] = ts
		s.extend(tt)
	pos = CDict(pos).sort()
	neg = CDict(neg).sort()
	return pos, neg
def get_0(x):
	while isArr(x) and len(x)>0: x = x[0]
	return x

class TensData(Const, AAdd):
	def __init__(self, *args, **kwargs):
		self._init(*args, **kwargs)
	def _init(self, i, v=NA, s=NA):
		self._sym = bool(s)
		t = NA
		if isA(i, Dict): t = lambda: i.iter()
		elif isDict(i): t = lambda: i.items()
		elif isTuple(i):
			x, y = i
			x = mulset(x)
			b = isIter(v)
			if b: v = list(v)
			s = 0
			r = {}
			for tx, tn in x:
				ty = y[s:s+tn]
				if b: ty = CDict(ty, v[s:s+tn]).sort()
				r[tx] = ty
				s += tn
			self._ind = CDict(r).sort()
			self._val = NA if b else v
			return
		elif isIter(i): t = lambda: enumerate(i)
		else: assert FALSE
		tt = next(t())[-1]
		b = isA(tt, Dict) or isDict(tt)
		r = {}
		if b:
			for x, y in t():
				assert isA(y, Dict) or isDict(y)
				r[x] = CDict(y).sort()
			self._val = NA
		else:
			for x, y in t():
				assert not isA(y, Dict) and not isDict(y)
				r[x] = sort(list(y))
			self._val = v
		self._ind = CDict(r).sort()
	def _argtext_init(self):
		return Text(self._ind, v=self._val, s=self._sym)
	def add(self, *tds):
		return TensDatas(self, *tds)
	def radd(self, *tds):
		return TensDatas(*tds, self)
	def __call__(self, t):
		dt = t.dtype
		dev = t.device
		if not self._sym:
			if na(self._val):
				for x, y in self._ind:
					y, v = y.key(list), y.val(list)
					v = tens(v, dt=dt, dev=dev)
					t[x, y] = v
			else:
				for x, y in self._ind:
					t[x, y] = self._val
		else:
			if na(self._val):
				for x, y in self._ind:
					y, v = y.key(list), y.val(list)
					v = tens(v, dt=dt, dev=dev)
					t[x, y] = v
					t[y, x] = v
			else:
				for x, y in self._ind:
					t[x, y] = self._val
					t[y, x] = self._val
		return t
class TensDatas(TensData):
	def _init(self, *tds):
		t = []
		for i in tds:
			if isA(i, TensData): t.append(i)
			elif isA(i, TensDatas): t.extend(i._tds)
		self._tds = t
	def _argtext_init(self):
		return Text(*self._tds)
	def __call__(self, t):
		for i in self._tds: t = i(t)
		return t
