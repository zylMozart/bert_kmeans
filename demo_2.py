import random as rd
import numpy as np
import torch as tc
from torch import nn
from matplotlib import pyplot as plt
from xkg import *
from xkg.tens import *
from xkg.bert import *
from xkg.kmeans import *
from xkg.tfidf import *
from xkg.text import *

dev = File('data/dev.txt').read()
print(f'dev={dev}')
dns = ['so', 'gs', 'bm']
sns = ['new', 'tfidf', 'skipgram']
# ws = [0.5, 1, 5]

def init_pq(s, eps=NA):
	eps = nav(eps, EPS)
	n = s.shape[0]
	t = tc.eye(n, dtype=DTB, device=s.device)
	q = s.sum(dim=1).sub_(s[t])
	p = (s-q.view(-1, 1)/(n-1)).pow_(2)
	p[t] = 0
	p = p.div_(p.sum(dim=1, keepdim=TRUE)+eps)
	q = q.div_(q.sum()+eps)
	return p, q
def sample_pq(p, q, n=NA, m=NA, l=NA, w=NA, eps=NA):
	n = nav(n, 8)
	m = nav(m, 8)
	l = nav(l, 32)
	w = nav(w, 0.1)
	eps = nav(eps, EPS)
	for i in range(n):
		t = q.repeat(m, 1)
		r = []
		for i in range(l):
			k = t.sum(dim=1)<eps
			if k.sum()>0: t[k] = q
			c = t.multinomial(1)
			t = t.mul_(p[c.squeeze()])
			t = t.div_(t.sum(dim=1, keepdim=TRUE)+eps)
			r.append(c)
		r = tc.cat(r, dim=1)
		for i in range(m):
			t = r[i]
			q[t] *= w
			for j in range(l-1):
				p[t[j], t[j+1]] *= w
			yield t
def run_km(km, n=NA, p=NA):
	n = nav(n, 20)
	k = NA
	v = []
	for i in range(n):
		km()
		km.eval()
		scores = km.scores
		if na(k): k = list(scores.keys())
		v.append(list(scores.values()))
		if p: print(f'{i}: {scores}')
	if p: print('='*64)
	return k, v
def proc_km(k, v):
	v = tens(v, dt=DTR, dev=dev)
	ms = v.mean(dim=0).mul(100).tolist()
	cs = v.std(dim=0).mul(100).tolist()
	ms = [round(i, 4) for i in ms]
	cs = [round(i, 4) for i in cs]
	s = dict(zip(k, zip(ms, cs)))
	return s
def get_word_vec(x, k):
	import numpy as np
	def load_word_vec_skipgram():
		file = File('data/wv/GoogleNews-vectors-negative300.bin')
		dt = np.dtype('float32')
		with file.open('rb') as f:
			header = f.readline().decode()
			vocab_size, vec_size = map(int, header.split()) # 300_0000, 300
			yield vocab_size, vec_size
			vec_len = vec_size*dt.itemsize # 1200
			for line in range(vocab_size):
				word = []
				while TRUE:
					b = f.read(1)
					if b==b' ': break
					elif b!=b'\n': word.append(b)
				word = b''.join(word).decode()
				vec = np.frombuffer(f.read(vec_len), dtype=dt)
				yield word, vec
	def load_word_vec_glove():
		file = File('data/wv/glove.42B.300d.txt')
		dt = np.dtype('float32')
		b = FALSE
		with file.open('rb') as f:
			for line in f:
				line = line.decode()
				line = line.strip()
				word, vec = line.split(' ', 1)
				vec = np.fromstring(vec, dtype=dt, sep=' ')
				if not b:
					yield NAN, vec.shape[0]
					b = TRUE
				yield word, vec
	load_word_vec = (load_word_vec_skipgram, load_word_vec_glove)[k]()

	x = [i.split() for i in x]

	vocab = set()
	for i in x: vocab.update(i)

	_, vec_size = next(load_word_vec)
	word_vec = {}
	for word, vec in load_word_vec:
		if word in vocab: word_vec[word] = vec

	r = []
	for i in x:
		s, n = 0, 0
		for j in i:
			if j in word_vec:
				s += word_vec[j]
				n += 1
		if n>0: s = (s/n).tolist()
		else: s = [0]*vec_size
		r.append(s)

	return r
def get_lplc(x, eps=NA):
	eps = nav(eps, EPS)
	r = x.sum(dim=1).diag()-x
	t = 1/(r.diag().sqrt()+eps)
	a = t.view(-1, 1).expand(*r.shape)
	b = t.view(1, -1).expand(*r.shape)
	r = a*r*b
	return r
def get_wvs():
	import numpy as np
	def load_word_vec_skipgram(wid):
		file = File('data/wv/GoogleNews-vectors-negative300.bin')
		dt = np.dtype('float32')
		with file.open('rb') as f:
			header = f.readline().decode()
			vocab_size, vec_size = map(int, header.split()) # 300_0000, 300
			yield vocab_size, vec_size
			vec_len = vec_size*dt.itemsize # 1200
			for line in range(vocab_size):
				word = []
				while TRUE:
					b = f.read(1)
					if b==b' ': break
					elif b!=b'\n': word.append(b)
				word = b''.join(word).decode()
				vec = f.read(vec_len)
				if word in wid:
					yield wid[word], np.frombuffer(vec, dtype=dt)
	for dn in dns:
		print(f'dn: {dn}')
		xpp = File(f'data/{dn}-xpp').load()
		wid = TFIDF(xpp).wid
		print(f'size_vocab: {len(wid)}')
		load_word_vec = load_word_vec_skipgram(wid)
		_, vec_size = next(load_word_vec)
		wvs = {w: tens(v, dt=DTR, dev=dev).view(1, -1) for w, v in load_word_vec}
		print(f'size_vocab_wv: {len(wvs)}')
		wvs = [(wvs[i] if i in wvs else tens0((1, vec_size), dt=DTR, dev=dev)) for i in range(len(wid))]
		wvs = tc.cat(wvs, dim=0)
		File(f'{dn}-wvs').store(wvs)
def get_sim():
	for dn in dns:
		print(f'dn: {dn}')

		xpp = File(f'data/{dn}-xpp').load()
		tfidf = TFIDF(xpp)

		wvs = File(f'{dn}-wvs').load()
		wvs = wvs.to(dev)
		wvs = tens_unit(wvs, dim=1)
		wvs = wvs.matmul(wvs.t())
		for i in range(wvs.shape[0]):
			wvs[i, i] = 1

		tfidf = tfidf(TRUE, dev=dev)
		tfidf = tens_unit(tfidf, dim=1)

		t = tfidf.matmul(wvs)
		del wvs
		t = t.matmul(tfidf.t())
		del tfidf

		k = tc.eye(t.shape[0], dtype=DTB, device=dev)
		t[k] = 0
		m = t.max()
		print(m)
		t.div_(m)
		t[k] = 1
		File(f'data/{dn}-sim-new').store(t)
		del t
def T(*args, exts=NA, nstr=NA, rprob=NA, **kwargs):
	params = {}
	if nna(exts): params['exts'] = exts
	if nna(nstr): params['nstr'] = nstr
	params['rprob'] = nav(rprob, 2)
	return Text(*args, **kwargs).params(params)

def run_bkm_baseline(fn, ls):
	batch_size = 32
	iter_count = 10
	iters = 50

	def do_train_euc(self):
		ids = self.batch[0]
		r = self.model(*self.batch)

		s = self.sup()
		ids = ids.to(s.device)
		s = s[ids][:, ids]
		s = s.to(self.dev)

		if self.key()==0:
			r = tens_dist_ssq(r)
			loss = (r*s).mean()
		else:
			r = r-r.max(dim=1, keepdim=TRUE)[0]
			r = tens_unit(r.exp(), dim=1)
			r = r.matmul(r.t())

			r[r<=0] = self.eps
			rt = 1-r
			rt[rt<=0] = self.eps

			loss = -(r.log()*s+rt.log()*(1-s)).mean()
		loss = loss*self.weight()
		return loss
	def do_train_cos(self):
		ids = self.batch[0]
		r = self.model(*self.batch)
		r = tens_unit(r, dim=1)
		r = r.matmul(r.t())

		s = self.sup()
		ids = ids.to(s.device)
		s = s[ids][:, ids]
		s = s.to(self.dev)

		loss = ((r-s)**2).mean()
		loss = loss*self.weight()
		return loss

	if ls=='euc':
		do_train = do_train_euc
		get_kmeans = KMeansEuc
	elif ls=='cos':
		do_train = do_train_cos
		get_kmeans = KMeansCos
	else:
		do_train = NA
		get_kmeans = NA

	init_rand(88888888)
	rb = RunBert()
	rb.init_path()
	rb.init_dev(dev)
	file = File(f'res/{fn}')
	for dn in dns[-1:]:
		x, y = CDict(File(f'data/{dn}.dataset').load())[['x', 'y']].val(list)
		rb.init_data((x, y))
		for param in range(1, 3):
			if param==0:
				pn = 'l'
				def get_sampler(dataset):
					sp_l = SamplerPair(dataset, 1, pair_size)
					sp_l = SamplerPairn(dataset, sp_l, iters, iter_count)
					return sp_l
			else:
				if param==1:
					pn = 'lu(tfidf)'
					sim = get_sim(dn, 'tfidf')
				elif param==2:
					pn = 'lu(w2v)'
					sim = get_sim(dn, 'wv1')
				else:
					pn = NA
					sim = NA
				sim_size = sim.shape[0]
				rb.sim = sim
				del sim
				def get_sampler(dataset):
					sp_u = SamplerPair(dataset, 0, sim_size)
					sp_l = SamplerPair(dataset, 1, pair_size)
					sp_u = SamplerPairn(dataset, sp_u, iters, iter_count)
					sp_l = SamplerPairn(dataset, sp_l, iters, iter_count)
					if ls=='euc': sps = (sp_l, sp_u)
					elif ls=='cos': sps = (sp_u, sp_l)
					else: sps = ()
					sp_lu = SamplerPairs(dataset, *sps)
					return sp_lu

			for w in ws:
				pair = get_pair(y, w, dev)
				pair_size = pair.shape[0]
				rb.pair = pair
				rb.init_model()

				rb.init_sampler_eval()
				rb.init_sampler(get_sampler)
				rb.init_optimizer()
				for i in range(iters):
					rb.train(do_train)
					epoch_loss = rb.epoch_loss
					rb.clear_train()

					rb.init_repr()
					rb.eval()
					rb.clear_eval()

					km = rb.kmeans(get_kmeans)
					rb.clear_repr()

					t = run_km(km)
					del km
					_, ff = proc_km(t)
					t = {
						'nmi':ff['nmi'],
						'ari':ff['ari'],
						'loss':round(epoch_loss, 4),
					}
					t = f'{dn}-{pn}-{w}-{(i+1)*iter_count}: {t}'
					file.writea(t+'\n')
					print(TimeDate().valstr())
					print(t)

				rb.pair = NA
				rb.clear_optimizer()
				rb.clear_sampler()
				rb.clear_sampler_eval()
				rb.clear_model()
				print('='*64)

			rb.sim = NA
		rb.clear_data()
def run_bkm_proposed(fn):
	def do_train_cos(self):
		ids = self.batch[0]
		r = self.model(*self.batch)
		r = tens_unit(r, dim=1)
		r = r.matmul(r.t())

		s = self.sup()
		ids = ids.to(s.device)
		s = s[ids][:, ids]
		s = s.to(self.dev)

		loss = ((r-s)**2).mean()
		loss = loss*self.weight()
		return loss

	do_train = do_train_cos
	get_kmeans = KMeansCos

	init_rand(88888888)
	rb = RunBert()
	rb.init_path()
	rb.init_dev(dev)
	file = File(f'res/{fn}')
	for dn in dns:
		if dn=='so': continue

		y = File(f'data/{dn}-y').load()
		x = File(f'data/{dn}-x').load()
		rb.init_data((x, y))
		for sn in sns:
			if sn in ['tfidf', 'skipgram']:
				if dn=='gs' and sn=='tfidf': continue
				if dn=='bm' and sn=='skipgram': continue

			sim = File(f'data/{dn}-sim-{sn}').load()
			sim_size = sim.shape[0]
			rb.sim = sim
			del sim
			for pn in range(3):
				if pn in [0, 1]: continue

				if pn in [0, 1]:
					eig = File(f'data/{dn}-sim-{sn}-eig').load()
					eig = tens(eig, dt=DTR, dev=dev)
					eigk = (1, 0.1)[pn]
					ids = tensa(len(y), dt=DTI, dev=dev)
					ids = ids[eig>=eigk].tolist()
					print(f'len(ids)={len(ids)}, eigk={eigk}')
					del eig, eigk
				else: ids = NA
				model_name = f'model-{dn}-{sn}-{pn}'
				rb.init_model()

				iters = 100
				iter_count = 150
				def get_sampler(dataset):
					sp_u = SamplerPair(dataset, 0, sim_size, ids=ids, nks=10)
					sp_u = SamplerPairn(dataset, sp_u, iters, iter_count)
					return sp_u

				rb.init_sampler_eval()
				rb.init_sampler(get_sampler)
				rb.init_optimizer()
				for i in range(iters):
					rb.train(do_train)
					epoch_loss = rb.epoch_loss
					rb.clear_train()

					rb.init_repr()
					rb.eval()
					rb.clear_eval()

					km = rb.kmeans(get_kmeans)
					rb.clear_repr()

					k, v = run_km(km)
					del km
					s = proc_km(k, v)
					s['loss'] = round(epoch_loss, 4)

					t = f'{dn}-{sn}-{pn}-{i+1}: {s}'
					file.writea(t+'\n')
					print(f'{TimeDate().valstr()}: {t}')

				rb.clear_optimizer()
				rb.clear_sampler()
				rb.clear_sampler_eval()
				rb.clear_model()
				file.writea('\n')
				print('='*64)
			rb.sim = NA
		rb.clear_data()
def draw_bkm(fn):
	from matplotlib import pyplot as plt
	plt.rcParams['figure.figsize'] = (40, 15)
	plt.rcParams['figure.dpi'] = 100
	plt.rcParams['savefig.dpi'] = 100

	r = {}
	for i in get_lines(File(fn)):
		k, v = i.split(':', 1)
		dn, sn, pn, _ = k.split('-')
		sn = f'{sn}-{pn}'
		v = eval(v)
		if dn not in r: r[dn] = {}
		for c in v:
			if c not in r[dn]: r[dn][c] = {}
			if sn not in r[dn][c]: r[dn][c][sn] = []
			t = v[c]
			if isTuple(t): t = t[0]
			r[dn][c][sn].append(t)
	i = 0
	for dn in r:
		for c in r[dn]:
			plt.subplot(241+i)
			for sn in r[dn][c]:
				t = r[dn][c][sn]
				x = (tensa(len(t), dt=DTI)+1).numpy()
				y = tens(t, dt=DTR).numpy()
				plt.plot(x, y, label=sn)
			plt.title(f'{dn}-{c}')
			plt.legend()
			plt.grid()
			i += 1
	bj = 0.05
	jj = 0.2
	plt.subplots_adjust(left=bj, right=1-bj, bottom=bj, top=1-bj, wspace=jj, hspace=jj)
	plt.savefig(f'{fn}.jpg')
	plt.clf()
	plt.cla()
def main():
	# run_bkm_proposed('bert-kmeans-proposed-7.txt')
	# draw_bkm('bert-kmeans-proposed-7.txt')
	pass

def before_draw(w=NA, h=NA):
	w = nav(w, 10)
	h = nav(h, 5)
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.rcParams['axes.unicode_minus'] = FALSE
	plt.rcParams['figure.figsize'] = (w, h)
	plt.rcParams['figure.dpi'] = 100
	plt.rcParams['savefig.dpi'] = 100
def after_draw(b=NA, z=NA):
	b = nav(b, 0.1)
	z = nav(z, 0.2)
	plt.grid()
	plt.subplots_adjust(
		left=b, right=1-b,
		bottom=b, top=1-b,
		wspace=z, hspace=z,
	)
def draw_jm(k, n=NA):
	n = nav(n, 100)
	before_draw()
	x = tensl(-1, 1, n, dt=DTR)
	y = x
	t = x
	s = k
	while s>0:
		t = (t*10000+1).sin()
		if (s&1)>0:
			y = ((y*10000+1).sin()+t)*0.5
		s >>= 1
	x = x.numpy()
	y = y.numpy()
	plt.plot(x, y, label=f'{k}')
	after_draw()
	plt.show()
def draw_sigmoids(k, n=NA):
	n = nav(n, 1000)
	before_draw()
	x = tensl(-100, 100, n, dt=DTR)
	y = x
	for i in range(k):
		y = y.sigmoid()
		plt.plot(x.numpy(), y.numpy(), label=f'{i+1}')
	plt.legend()
	after_draw()
	plt.show()
	print(y)
def draw_sj(k=NA, t=NA, n=NA):
	k = nav(k, [0.1, 0.3, 0.5, 0.7, 0.9])
	t = nav(t, 100)
	n = nav(n, 1000)
	before_draw()
	l = len(k)
	k = tens(k, dt=DTR)
	a = k**(1/t)
	x = tensa(n, dt=DTR)
	y = a.view(-1, 1)**x
	for i in range(l):
		plt.plot(x.numpy(), y[i].numpy(),
			label=f'{k[i].tolist():.1f}:{a[i].tolist():.4f}')
	plt.legend()
	after_draw()
	plt.show()
def draw_qxl(x, y):
	before_draw()
	t = tens([x, y], dt=DTR).view(1, -1)
	r = (tense(t.numel(), dt=DTR)-t.t().matmul(t)).tolist()
	t = t.squeeze().tolist()
	plt.arrow(0, 0, *t,
		length_includes_head=TRUE,
		head_width=0.1, head_length=0.1,
		ec='b', fc='b')
	for i in r:
		plt.arrow(*t, *i,
			length_includes_head=TRUE,
			head_width=0.1, head_length=0.1,
			ec='r', fc='r')
	dom = 5
	plt.xlim(-dom, dom)
	plt.ylim(-dom/2, dom/2)
	after_draw()
	plt.show()
def draw_zyx(k=NA, d=NA, n=NA):
	k = nav(k, 1e-5)
	d = nav(d, 10)
	n = nav(n, 100)
	before_draw()
	x = tensl(-1, 0, n, dt=DTR)*d
	y = (1-(k/(10**x+EPS)).sqrt()).relu()
	x = x.numpy()
	y = y.numpy()
	plt.plot(x, y, label=f'下采样丢弃概率')
	plt.legend()
	after_draw()
	plt.show()

draw_sj()









if __name__=='__main__':
	pass
	# print(f'start: {TimeDate().valstr()}')
	# main()
	# print(f'finish: {TimeDate().valstr()}')
	"""
	exit()
	init_rand(8888)
	s = File(f'data/so.sim.tfidf').load()
	# t = tensri(s.shape[0], (1000,), dt=DTI64)
	# s = s[t][:, t]
	# s[tc.eye(s.shape[0], dtype=DTB, device=s.device)] = 0
	# File(f'temp-sim').store(s)
	# s = File(f'temp-sim').load()
	n = s.shape[0]
	sm = s.mean().tolist()
	print(f'{sm:.4f}')
	p, q = init_pq(s)
	print(f'{n/32:.4f}')
	g = sample_pq(p, q, 16, 16)
	d = {}
	r = []
	for n, t in enumerate(g):
		t = t.tolist()
		for i in t: d[i] = d.get(i, 0)+1
		tsm = s[t][:, t].mean().tolist()
		r.append([n, len(d), max(d.values()), round(tsm/sm, 4)])
	r = tens(r, dt=DTR)
	print(r)

	exit()

	def sample_l(p, q, eps=NA):
		n = q.shape[0]
		while TRUE:
			t = tc.randperm(n).tolist()
			for i in t: yield i
	class Model(nn.Module):
		def __init__(self, n, d):
			nn.Module.__init__(self)
			self.emb = nn.Embedding(n, d)
		def forward(self, x):
			return tens_unit(self.emb(x), dim=1)

	init_rand(233)
	y = File(f'data/so.dataset').load()['y']
	s = File(f'data/so.sim.tfidf').load()
	n = s.shape[0]

	s = s.to(dev)
	p, q = get_pq(s)
	g = sample_pq(p, q)

	model = Model(n, 32).to(dev)
	opt = Adam(model.parameters(), lr=1e-2)
	opt.zero_grad()
	losss = 0
	for n, l in enumerate(part(g, 128)):
		if n>0 and n%100==0:
			print(f'{n}: {losss:.04f}')
			losss = 0
			if n%1000==0:
				with tc.no_grad():
					emb = model.emb.weight.data
					km = KMeansCos(emb, y, dev=dev)
					r = run_km(km, p=TRUE)
					_, f2 = f_km(r)
					del km
					print(f'{f2}')
					input()
		r = model(tens(l, dt=DTI64, dev=dev))
		r = r.matmul(r.t())
		t = s[l][:, l].to(dev)
		loss = ((r-t)**2).mean()
		loss.backward()
		opt.step()
		opt.zero_grad()
		losss += loss.tolist()
	"""
	"""
	dns = ['so', 'gs', 'bm']
	ns = [20000, 12340, 20000]
	ms = [1250, 772, 1250]
	bs = 32
	nks = 2
	"""
	"""
	so.bert.tfidf.uull, w=0%, res=(0.65, 0.53, 0.74)
	so.bert.tfidf.uull, w=0.1%, res=(0.63, 0.4, 0.68)
	so.bert.tfidf.uull, w=0.5%, res=(0.68, 0.49, 0.73)
	so.bert.tfidf.uull, w=1%, res=(0.7, 0.52, 0.75)
	so.bert.tfidf.uull, w=5%, res=(0.73, 0.67, 0.81)

	so.bert.wv1.uull, w=0%, res=(0.46, 0.25, 0.55)
	so.bert.wv1.uull, w=0.1%, res=(0.41, 0.26, 0.5)
	so.bert.wv1.uull, w=0.5%, res=(0.62, 0.43, 0.69)
	so.bert.wv1.uull, w=1%, res=(0.63, 0.49, 0.7)
	so.bert.wv1.uull, w=5%, res=(0.71, 0.64, 0.79)
	----------------------------------------------------------------
	gs.bert.tfidf.uull, w=0%, res=(0.43, 0.37, 0.61)
	gs.bert.tfidf.uull, w=0.1%, res=(0.39, 0.25, 0.54)
	gs.bert.tfidf.uull, w=0.5%, res=(0.58, 0.54, 0.76)
	gs.bert.tfidf.uull, w=1%, res=(0.71, 0.72, 0.86)
	gs.bert.tfidf.uull, w=5%, res=(0.76, 0.77, 0.89)

	gs.bert.wv1.uull, w=0%, res=(0.59, 0.57, 0.77)
	gs.bert.wv1.uull, w=0.1%, res=(0.2, 0.12, 0.35)
	gs.bert.wv1.uull, w=0.5%, res=(0.65, 0.64, 0.81)
	gs.bert.wv1.uull, w=1%, res=(0.69, 0.68, 0.84)
	gs.bert.wv1.uull, w=5%, res=(0.74, 0.74, 0.87)
	----------------------------------------------------------------
	bm.bert.tfidf.uull, w=0%, res=(0.34, 0.25, 0.45)
	bm.bert.tfidf.uull, w=0.1%, res=(0.24, 0.12, 0.32)
	bm.bert.tfidf.uull, w=0.5%, res=(0.36, 0.23, 0.45)
	bm.bert.tfidf.uull, w=1%, res=(0.39, 0.27, 0.48)
	bm.bert.tfidf.uull, w=5%, res=(0.48, 0.38, 0.61)

	bm.bert.wv1.uull, w=0%, res=(0.28, 0.14, 0.33)
	bm.bert.wv1.uull, w=0.1%, res=(0.19, 0.09, 0.26)
	bm.bert.wv1.uull, w=0.5%, res=(0.26, 0.16, 0.34)
	bm.bert.wv1.uull, w=1%, res=(0.28, 0.18, 0.37)
	bm.bert.wv1.uull, w=5%, res=(0.45, 0.35, 0.58)
	"""
