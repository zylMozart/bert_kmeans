import torch as tc
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Sampler, SequentialSampler
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertAdam, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from .base import *
from .tens import *
from .kmeans import *

__all__ = [
	'RunBert',
	'BertForPair',
	'SamplerPair',
	'SamplerPairmz',
	'SamplerPairn',
	'SamplerPairs',
]

class RunBert:
	def __init__(self, **kwargs):
		self.Bert = BertForPair
		self.bert_model = 'bert-base-uncased'
		self.lower_case = TRUE
		self.warmup_prop = 0.1
		self.learning_rate = 5e-5
		self.base_path = Dir('temp')
		self.eps = EPS
		self.dtr = DTR
		self.dti = DTI
		self.dtb = DTB
		for k, v in kwargs.items(): setattr(self, k, v)
	def init_path(self, path_name=NA):
		path_name = nav(path_name, self.__class__.__name__)
		self.path = Dir(self.base_path/path_name).add()
		return self
	def init_dev(self, dev):
		self.dev = dev
		self.ncuda = tc.cuda.device_count() if dev=='cuda' else 0
		return self
	def init_data(self, dataset):
		self.dataset = dataset
		tokenizer = BertTokenizer.from_pretrained(
			self.bert_model, do_lower_case=self.lower_case
		)
		self.data, self.seq_length = self._get_data(dataset, tokenizer)
		return self
	def _get_data(self, dataset, tokenizer):
		x, y = dataset
		assert len(x)==len(y)
		n = len(x)
		ids = list(range(n))
		toks = [['[CLS]']+tokenizer.tokenize(i)+['[SEP]'] for i in x]
		seqs = [tokenizer.convert_tokens_to_ids(i) for i in toks]
		lengths = [len(i) for i in seqs]
		seq_length = int(exp2(ceil(log2(max(lengths)))))
		masks = [[1]*i for i in lengths]
		for i in range(n):
			t = [0]*(seq_length-lengths[i])
			seqs[i] += t
			masks[i] += t
		assert n==len(ids)==len(seqs)==len(masks)
		assert all(seq_length==len(seqs[i])==len(masks[i]) for i in range(n))
		data = (ids, seqs, masks)
		data = tuple(tens(i, dt=DTI64, dev=self.dev) for i in data)
		return data, seq_length
	def init_model(self, model=NA):
		if na(model): model = self.bert_model
		else: model = Dir(self.path/model).pathstr()
		self.model = self.Bert.from_pretrained(model).to(self.dev)
		if self.ncuda>1: self.model = nn.DataParallel(self.model)
		return self
	def init_sampler(self, get_sampler, batch_size=NA):
		self.batch_size = nav(batch_size, 32)
		dataset = TensorDataset(*self.data)
		self.sampler = get_sampler(dataset)
		self.dataloader = DataLoader(
			dataset, sampler=self.sampler, batch_size=self.batch_size
		)
		return self
	def init_optimizer(self):
		optimize_steps = iceil(len(self.sampler)/self.batch_size)
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_params = [
			{'params':[i for n, i in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
			{'params':[i for n, i in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay':0.0},
		]
		self.optimizer = BertAdam(
			optimizer_params, lr=self.learning_rate,
			warmup=self.warmup_prop, t_total=optimize_steps
		)
		return self
	def train(self, do_train):
		self.model.train()
		self.epoch_loss = 0
		for self.step, self.batch in enumerate(self.dataloader):
			loss = do_train(self)
			self.step_loss = loss.tolist()
			loss.backward()
			self.optimizer.step()
			self.optimizer.zero_grad()
			self.epoch_loss += self.step_loss
		return self
	def save_model(self, path_name):
		path = Dir(self.path/path_name).add()
		path_model = (path/WEIGHTS_NAME).pathstr()
		path_config = (path/CONFIG_NAME).pathstr()
		model = self.model.module if hasattr(self.model, 'module') else self.model
		tc.save(model.state_dict(), path_model)
		model.config.to_json_file(path_config)
		return self
	def clear_train(self):
		self.step = NA
		self.batch = NA
		self.step_loss = NA
		self.epoch_loss = NA
		return self
	def clear_optimizer(self):
		self.optimizer = NA
		return self
	def clear_sampler(self):
		self.batch_size = NA
		self.sampler = NA
		self.dataloader = NA
		return self
	def clear_model(self):
		self.model = NA
		return self
	def clear_data(self):
		self.dataset = NA
		self.data = NA
		self.seq_length = NA
		return self

	def init_sampler_eval(self, batch_size=NA):
		self.batch_size_eval = nav(batch_size, 128)
		dataset = TensorDataset(*self.data)
		self.sampler_eval = SequentialSampler(dataset)
		self.dataloader_eval = DataLoader(
			dataset, sampler=self.sampler_eval, batch_size=self.batch_size_eval
		)
		return self
	def init_repr(self):
		self.repr = []
		return self
	def eval(self):
		self.model.eval()
		for self.step, self.batch in enumerate(self.dataloader_eval):
			with tc.no_grad():
				repr = self.model(*self.batch)
				self.repr.extend(repr.tolist())
		return self
	def save_repr(self, path_name):
		File(self.path/path_name).store(self.repr)
		return self
	def clear_eval(self):
		self.step = NA
		self.batch = NA
		return self
	def clear_repr(self):
		self.repr = NA
		return self
	def clear_sampler_eval(self):
		self.batch_size_eval = NA
		self.sampler_eval = NA
		self.dataloader_eval = NA
		return self

	def kmeans(self, get_kmeans=NA):
		get_kmeans = nav(get_kmeans, KMeansCos)
		_, y = self.dataset
		repr = self.repr
		km = get_kmeans(repr, y, dtr=self.dtr, dti=self.dti, dev=self.dev)
		return km
	def key(self):
		return self.sampler.key()
	def weight(self):
		return 1
	def sup(self):
		return getattr(self, ['sim', 'pair'][self.key()])
class BertForPair(BertPreTrainedModel):
	def __init__(self, config):
		BertPreTrainedModel.__init__(self, config)
		self.bert = BertModel(config)
		self.apply(self.init_bert_weights)
	def forward(self, ids, seqs, masks=NA):
		t, _ = self.bert(seqs, attention_mask=masks,
			output_all_encoded_layers=FALSE)
		return t[:, 0]
class SamplerX(Sampler):
	def __init__(self, dataset, *args, **kwargs):
		Sampler.__init__(self, dataset)
		self.init(*args, **kwargs)
	def __iter__(self):
		return self.iter()
	def __len__(self):
		return self.size()
	def init(self, *args, **kwargs):
		pass
	def iter(self):
		pass
	def size(self):
		return self._size
	def key(self):
		return self._key
class SamplerPair(SamplerX):
	def init(self, key, n, ids=NA, nks=NA, bs=NA):
		nks = nav(nks, 1)
		bs = nav(bs, 32)
		if nna(ids): n = len(ids)

		m = max(1, n//bs)
		ks = [0, 1]+coprime(m, 100)
		ks = ks[:nks]

		self._info = (ids, n, ks, m, bs)
		self._size = len(ks)*m*bs
		self._key = key
	def iter(self):
		x, n, ks, m, bs = self._info
		t = tc.randperm(n).tolist()
		g = self.sample(t, ks, m, bs)
		if na(x):
			for i in g: yield i
		else:
			for i in g: yield x[i]
	def sample(self, x, ks, m, bs):
		n = len(x)
		l = m*bs
		for k in ks:
			for i in range(m):
				for j in range(bs):
					yield x[((i+j*k)%m*bs+j)%n]
			x = x[l:]+x[:l]
class SamplerPairmz(SamplerX):
	def init(self, key, p, q, n, m, bs=NA, w=NA, eps=NA):
		bs = nav(bs, 32)
		w = nav(w, 0.1)
		eps = nav(eps, EPS)
		self._info = (p, q, n, m, bs, w, eps)
		self._size = n*m*bs
		self._key = key
	def iter(self):
		return self.sample(*self._info)
	def sample(self, p, q, n, m, bs, w, eps):
		pdev = p.device
		qdev = q.device
		for _ in range(n):
			t = q.repeat(m, 1)
			r = []
			for _ in range(bs):
				k = t.sum(dim=1)<eps
				if k.sum()>0: t[k] = q
				c = t.multinomial(1)
				t = t.mul_(p[c.squeeze().to(pdev)].to(qdev))
				t = t.div_(t.sum(dim=1, keepdim=TRUE)+eps)
				r.append(c)
			del t

			r = tc.cat(r, dim=1)
			# for i in range(m): q[r[i]] *= w

			r = r.tolist()
			# for i in r:
			# 	for j in range(bs-1):
			# 		p[i[j], i[j+1]] *= w

			for i in r:
				for j in i:
					yield j
class SamplerPairn(SamplerX):
	def init(self, sp, n, m, bs=NA):
		bs = nav(bs, 32)
		self._sp = sp
		self._spn = 0
		self._n = n
		self._m = m
		self._bs = bs
		self._size = n*m*bs
	def iter(self):
		for _ in range(self._m*self._bs):
			if self._spn<=0:
				self._spi = iter(self._sp)
				self._spn = self._sp.size()
			yield next(self._spi)
			self._spn -= 1
	def key(self):
		return self._sp.key()
class SamplerPairs(SamplerX):
	def init(self, *sps):
		self._sps = sps
		self._size = sum(sp.size() for sp in sps)
	def iter(self):
		for sp in self._sps:
			self._sp = sp
			for i in sp: yield i
	def key(self):
		return self._sp.key()
