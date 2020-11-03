from .base import *
from .tens import *
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from scipy.optimize import linear_sum_assignment as ass

__all__ = [
	'get_scores',
]

def get_scores(x, y, n, k, dtr, dev):
	tx = tens0((n, k), dt=dtr, dev=dev)
	ty = tens0((n, k), dt=dtr, dev=dev)
	tx = tens_sel_set(tx, x, 1)
	ty = tens_sel_set(ty, y, 1)
	t = tx.t().matmul(ty)
	del tx, ty
	tt = t.max()-t
	tt = tt.cpu().numpy()
	row, col = ass(tt)
	del tt
	t = t.cpu().numpy()
	t = t[row, col].sum()
	t = t.tolist()/n
	x = x.cpu().numpy()
	y = y.cpu().numpy()
	s = {
		'nmi': nmi(x, y, average_method='geometric'),
		'ari': ari(x, y),
		'acc': t,
	}
	return s
def get_evs(x, y, eps=NA):
	eps = nav(eps, EPS)
	n = x.shape[0]

	cn = x.t().matmul(y)
	cnx = cn.sum(dim=1, keepdim=TRUE)
	cny = cn.sum(dim=0, keepdim=TRUE)

	cr = cn/n
	crx = cnx/n
	cry = cny/n
	crxy = crx.matmul(cry)

	mi = (cr*(cr/(crxy+eps)+eps).log()).sum()
	ex = -(crx*(crx+eps).log()).sum()
	ey = -(cry*(cry+eps).log()).sum()
	nmi = mi/((ex*ey).sqrt()+eps)
	nmi = nmi.tolist()

	cp = (cn*(cn-1)/2).sum()
	cpx = (cnx*(cnx-1)/2).sum()
	cpy = (cny*(cny-1)/2).sum()
	tp = n*(n-1)/2

	i = cp
	mi = (cpx+cpy)/2
	ei = cpx*cpy/tp
	ari = (i-ei)/(mi-ei+eps)
	ari = ari.tolist()

	p = cn/(cnx+eps)
	r = cn/(cny+eps)
	m = 2*p*r/(p+r+eps)
	fm = (cny*m.max(dim=0, keepdim=TRUE)[0]).sum()/n
	fm = fm.tolist()

	entr = -(x*(x+eps).log()).sum(dim=1).mean()
	entr = entr.tolist()

	return nmi, ari, fm, entr
