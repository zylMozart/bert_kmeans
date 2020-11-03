import random as rd
import torch as tc
from torch import nn, optim
from xkg import *
from xkg.tens import *

pr.params({'exts': '0'})
dev = 'cuda'

def gelu(x):
	return (tc.erf(x/sqr(2))+1)*x*0.5
class OMul(nn.Module):
	def omul_xx_0(self, x):
		n, m = x.shape
		r = tensn((n, m, m), dt=x.dtype, dev=x.device)
		for i in range(n):
			t = x[i].view(1, -1)
			r[i] = t.t().matmul(t)
		return r
	def omul_xx_1(self, x):
		n, m = x.shape
		r = tensn((n, m, m), dt=x.dtype, dev=x.device)
		for i in range(m):
			t = x[:, i].view(-1, 1)
			r[:, i] = t*x
		return r
	def omul_xy_0(self, x, y):
		n, m = x.shape
		_, k = y.shape
		r = tensn((n, m, k), dt=x.dtype, dev=x.device)
		for i in range(n):
			a = x[i].view(1, -1)
			b = y[i].view(1, -1)
			r[i] = a.t().matmul(b)
		return r
	def omul_xy_1(self, x, y):
		n, m = x.shape
		_, k = y.shape
		r = tensn((n, m, k), dt=x.dtype, dev=x.device)
		for i in range(m):
			t = x[:, i].view(-1, 1)
			r[:, i] = t*y
		return r
	def omul_xy_1r(self, x, y):
		n, m = x.shape
		_, k = y.shape
		r = tensn((n, m, k), dt=x.dtype, dev=x.device)
		for i in range(k):
			t = y[:, i].view(-1, 1)
			r[:, :, i] = x*t
		return r
	def omul(self, x, y=NA):
		assert isTens(x)
		assert len(x.shape)==2
		if na(y) or x is y:
			n, m = x.shape
			if n<=m: return self.omul_xx_0(x)
			return self.omul_xx_1(x)
		assert isTens(y)
		assert len(y.shape)==2
		assert x.shape[0]==y.shape[0]
		n, m = x.shape
		_, k = y.shape
		if n<=min(m, k): return self.omul_xy_0(x, y)
		if m<=k: return self.omul_xy_1(x, y)
		return self.omul_xy_1r(x, y)
	def flat(self, x):
		assert isTens(x)
		assert len(x.shape)==3
		n, m, k = x.shape
		return x.view(n, m*k)
	def forward(self, x, y=NA, flat=NA):
		r = self.omul(x, y)
		flat = nav(flat, TRUE)
		if flat: r = self.flat(r)
		return r
class Model(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.omul = OMul()
		self.il = nn.Linear(1, 4)
		self.hl = nn.Linear(4, 4)
		self.ol = nn.Linear(4, 1)
	def forward(self, x):
		t = self.il(x)
		t = gelu(t)+t
		# t = self.omul(x, t)
		t = self.hl(t)
		t = gelu(t)+t
		# t = self.omul(x, t)
		return self.ol(t)

lr = 0.01
epochs = 1000
model = Model()
x = tensn((1024, 1), dt=DTR, dev=dev)
def func(x):
	s = tens1(x)
	s[x<0] = -1
	r = s/(x.abs()+EPS)
	return r
def loss(y, Y):
	return -((1-(y/Y-1).abs()).relu()+EPS).log().mean()
prs = pr.copy().params({'end': ' '})
init_rand(8888)
while TRUE:
	epoch = 0
	nconv = 0
	model = model.to(dev)
	opt = optim.Adam(model.parameters(), lr=lr)
	while TRUE:
		tensr((1024, 1), dt=DTR, dev=dev, out=x)
		x.mul_(200).sub_(100)
		y = func(x)

		r = model(x)
		l = loss(r, y)
		tl = l.tolist()

		opt.zero_grad()
		l.backward()
		opt.step()

		prs((epoch, round(tl, 2)))
		epoch += 1
		if epoch%100==0: pr()
		if epoch>epochs: break
		if tl<0.01: nconv += 1
		else: nconv = 0
		if nconv>=10: break
	pl()
	model = model.cpu()
	with tc.no_grad():
		for k, v in model.named_parameters():
			pr(k, v.squeeze().tolist())
		pl()
		while TRUE:
			s = pris()
			try:
				tx = tens([[real(s)]], dt=DTR)
				ty = func(tx)
				r = model(tx)
				l = loss(r, ty).tolist()
				r = r.squeeze().tolist()
				ty = ty.squeeze().tolist()
				pr(y=r, Y=ty, loss=l)
			except Exception as e:
				pr(e)
				s = pris()
				lr, epochs = [real(i) for i in s.split()]
				break




exit()
class A(Const):
	def __init__(self, *args, **kwargs):
		self.x = (args, kwargs)
	def _argtext_init(self):
		args, kwargs = self.x
		return Text(*args, **kwargs)
def store_mem_files():
	n_mb = 16
	d = Path('hehe').dir().add()
	t = tens0(n_mb*1024**2, dt=DTI8)
	i = 0
	try:
		while TRUE:
			f = d.div(f'{i:08d}').file().clear()
			f.store(t)
			i += 1
	except Exception as e:
		pl(f'{i}: {i*n_mb}MB, {i*n_mb/1024:.4f}GB')
		pl(e)
def find_code_invalid(fn='base.py'):
	def find_tabs(s):
		tok = '\t'
		n = 0
		for c in s:
			if c==tok: n += 1
			else: break
		return n

	r = []
	with open(fn, 'r') as f:
		t = 0
		for i, l in enumerate(f):
			if len(l.strip())<=0: continue
			n = find_tabs(l)
			if 0<=n<=t+1:
				print(f'{i:-4d}|{n:-2d}|{l}', end='')
			else:
				print(f'{i:-4d}|{n:-2d}|{"#"*66}', end='\n')
				r.append(i)
			t = n
	print(r)
def info_dataset(dev=NA):
	dev = nav(dev, DEV)
	n, pos, neg = Path('so.pair').file().load()
	sim = Path('so.sim').file().load().to(dev)
	rs = []
	for pair in [pos, neg]:
		pair = TensData(pair, 1, s=TRUE)(tens0((n, n), dt=DTB, dev=dev))
		r = []
		for i in range(n):
			t = sim[i][pair[i]]
			a = t.min().tolist()
			m = t.mean().tolist()
			b = t.max().tolist()
			r.append([a, m, b])
			pr(i, a, m, b)
		t = tens(r, dt=DTR, dev=dev)
		ma = t[:, 0].min().tolist()
		a = t[:, 0].mean().tolist()
		m = t[:, 1].mean().tolist()
		b = t[:, 2].mean().tolist()
		mb = t[:, 2].max().tolist()
		t = [ma, a, m, b, mb]
		pl(*t)
		rs.append(t)
	pr.params({'exts':'10'})(rs)
@pnte
def test_func_run_eff(f, n=NA):
	n = nav(n, 1)
	for _ in range(n): f()
def EXIT(b=NA, s=NA):
	b = nav(b, FALSE)
	assert isBool(b)
	s = nav(s, '#EXIT')
	assert isStr(s)
	print(s, end='')
	if b: input()
	exit()
def OLD():
	def jf():
		n = 1
		r = 1

		def drf(dn):
			dr = pow(0.1, 1/min((dn+2), 5))
			if dr*dn>2: dr = 2/dn
			return dr

		dn = 3
		dr = drf(dn)
		for i in range(10):
			print(f'{i}: {r:.4f} * {n} = {r*n:.4f}')
			n *= dn
			r *= dr

	def khpd(s):
		t = []
		for c in s:
			if c=='0': break
			elif c in '([{': t.append(c)
			elif c in ')]}':
				if len(t)<=0: return False
				elif t[-1]=={
					')':'(',
					']':'[',
					'}':'{',
				}[c]: t.pop()
				else: return False
		return len(t)<=0

	def re():
		"""
			rep = lambda s: _re.compile(s, _re.S)
			def toJSON(x):
				try: r = _json.dumps(x, ensure_ascii=FALSE, separators=(',', ':'))
				except: r = NA
				return r
			def fromJSON(s):
				try: r = _json.loads(s)
				except: r = NA
				return r
			pq = _pyquery.PyQuery
		"""
		pass

	def torch_funcs():
		"""
		tc.logspace
		tc.eye
		tc.randperm
		tc.normal
		tc.cat
		tc.stack

		t.view==t.reshape
		t.squeeze
		t.unsqueeze
		t.transpose
		t.permute
		t.expand
		t.repeat
		t.take
		t.split
		t.chunk
		t.topk
		t.kthvalue
		t.where
		t.gather
		t.trunc
		t.frac
		t.median
		t.clamp
		"""
# model.load_state_dict({
# 	'enc.weight': tens([1, -1], dt=DTR, dev=dev).view(-1, 1),
# 	'output.weight': tens([-1, -1.001], dt=DTR, dev=dev).view(1, -1),
# })
