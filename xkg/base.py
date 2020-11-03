# ================================================================ #
Na = type(None)
NA = None
def na(x):
	return x is None
def nna(x):
	return x is not None
def nav(x, v, w=NA):
	return v if x is None else (x if w is None else w)
# ================================================================ #
FALSE = False
TRUE = True
def false(x):
	return x is False
def true(x):
	return x is True
def nfalse(x):
	return x is not False
def ntrue(x):
	return x is not True
def boolv(x, v, w):
	return v if not x else w
# ================================================================ #
real = float
import math as _math
EPS = 1e-8
TAU = _math.tau
PI = _math.pi
E = _math.e
INF = _math.inf
NAN = _math.nan
def isFIN(x):
	return _math.isfinite(x)
def isINF(x):
	return _math.isinf(x)
def isNAN(x):
	return _math.isnan(x)
def frc(f, x, w=NA):
	w = nav(w, 0)
	b = w!=0
	t = NA
	if b:
		t = exp10(w)
		x = x*t
	x = f(x)
	if b: x = x/t
	return x
def floor(x, w=NA):
	return frc(_math.floor, x, w)
def ceil(x, w=NA):
	return frc(_math.ceil, x, w)
def ifloor(x, w=NA):
	return int(frc(_math.floor, x, w))
def iceil(x, w=NA):
	return int(frc(_math.ceil, x, w))
def iround(x, w=NA):
	return int(frc(round, x, w))
def abs(x):
	return _math.fabs(x)
def sq(x):
	return x*x
def sqr(x):
	return _math.sqrt(x)
def pow(x, z):
	return _math.pow(x, z)
def exp(x, d=NA):
	if na(d): return _math.exp(x)
	return pow(d, x)
def log(x, d=NA):
	if na(d): return _math.log(x)
	return _math.log(x, d)
def exp2(x):
	return _math.pow(2, x)
def log2(x):
	return _math.log2(x)
def exp10(x):
	return _math.pow(10, x)
def log10(x):
	return _math.log10(x)
def deg(x):
	return _math.degrees(x)
def rad(x):
	return _math.radians(x)
def sin(x):
	return _math.sin(x)
def cos(x):
	return _math.cos(x)
def tan(x):
	return _math.tan(x)
def asin(x):
	return _math.asin(x)
def acos(x):
	return _math.acos(x)
def atan(x):
	return _math.atan(x)
def sinh(x):
	return _math.sinh(x)
def cosh(x):
	return _math.cosh(x)
def tanh(x):
	return _math.tanh(x)
def asinh(x):
	return _math.asinh(x)
def acosh(x):
	return _math.acosh(x)
def atanh(x):
	return _math.atanh(x)
# ================================================================ #
def _any_all(f, x, args, b, r):
	return boolv(b, any, all)(boolv(r, (f(x, i) for i in args), (f(i, x) for i in args)))
def hasA(x, *k, b=NA, r=NA):
	return _any_all(hasattr, x, k, b, r)
def isA(x, *t, b=NA, r=NA):
	return _any_all(isinstance, x, t, b, r)
def evoA(x, *t, b=NA, r=NA):
	return _any_all(issubclass, x, t, b, r)
# ================================================================ #
def isCall(x):
	return hasattr(x, '__call__')
def isMeta(x):
	return isinstance(x, type) and issubclass(x, type)
def isType(x):
	return isinstance(x, type)
def isFunc(x):
	return hasattr(x, '__call__') and not isinstance(x, type)
def isBool(x):
	return isinstance(x, bool)
def isInt(x):
	return isinstance(x, int)
def isReal(x):
	return isinstance(x, real)
def isNum(x):
	return isinstance(x, int) or isinstance(x, real)
def isChar(x):
	return isinstance(x, str) and len(x)==1
def isStr(x):
	return isinstance(x, str)
def isTuple(x):
	return isinstance(x, tuple)
def isList(x):
	return isinstance(x, list)
def isArr(x):
	return isinstance(x, tuple) or isinstance(x, list)
def isSet(x):
	return isinstance(x, set)
def isDict(x):
	return isinstance(x, dict)
def isIter(x):
	return hasattr(x, '__iter__') and not isinstance(x, str) and not isinstance(x, type)
# ================================================================ #
class AStr:
	def __str__(self):
		return self.str()
	def str(self, *args, **kwargs):
		return ''
class ACode:
	def __repr__(self):
		return self.code()
	def code(self, *args, **kwargs):
		return ''
class AHash:
	def __hash__(self):
		return self.hash()
	def hash(self, *args, **kwargs):
		return 0
class AEqual:
	def __eq__(self, x):
		return self.equal(x)
	def __ne__(self, x):
		return not self.equal(x)
	def equal(self, *args, **kwargs):
		return FALSE
class AComp(AEqual):
	def __lt__(self, x):
		return self.comp(x)<0
	def __gt__(self, x):
		return self.comp(x)>0
	def __le__(self, x):
		return self.comp(x)<=0
	def __ge__(self, x):
		return self.comp(x)>=0
	def equal(self, *args, **kwargs):
		return self.comp(*args, **kwargs)==0
	def comp(self, *args, **kwargs):
		return NAN
class ASize:
	def __len__(self):
		return self.size()
	def size(self, *args, **kwargs):
		return 0
class AHas:
	def __contains__(self, x):
		return self.has(x)
	def has(self, *args, **kwargs):
		return FALSE
class AIter:
	def __iter__(self):
		return self.iter()
	def iter(self, *args, **kwargs):
		pass
class ARIter:
	def __reversed__(self):
		return self.riter()
	def riter(self, *args, **kwargs):
		pass
class AGet:
	def __getitem__(self, k):
		return self.get(k)
	def get(self, *args, **kwargs):
		pass
class ASet:
	def __setitem__(self, k, v):
		return self.set(k, v)
	def set(self, *args, **kwargs):
		return self
class ADel:
	def __delitem__(self, k):
		return self.del_(k)
	def del_(self, *args, **kwargs):
		return self
class APop:
	def pop(self, *args, **kwargs):
		pass
class AAdd:
	def __add__(self, x):
		return self.add(x)
	def __radd__(self, x):
		return self.radd(x)
	def add(self, *args, **kwargs):
		pass
	def radd(self, *args, **kwargs):
		pass
class ASub:
	def __sub__(self, x):
		return self.sub(x)
	def __rsub__(self, x):
		return self.rsub(x)
	def sub(self, *args, **kwargs):
		pass
	def rsub(self, *args, **kwargs):
		pass
class AMul:
	def __mul__(self, x):
		return self.mul(x)
	def __rmul__(self, x):
		return self.rmul(x)
	def mul(self, *args, **kwargs):
		pass
	def rmul(self, *args, **kwargs):
		pass
class ADiv:
	def __truediv__(self, x):
		return self.div(x)
	def __rtruediv__(self, x):
		return self.rdiv(x)
	def div(self, *args, **kwargs):
		pass
	def rdiv(self, *args, **kwargs):
		pass
class AMulMul:
	def __pow__(self, x):
		return self.mulmul(x)
	def __rpow__(self, x):
		return self.rmulmul(x)
	def mulmul(self, *args, **kwargs):
		pass
	def rmulmul(self, *args, **kwargs):
		pass
class ADivDiv:
	def __floordiv__(self, x):
		return self.divdiv(x)
	def __rfloordiv__(self, x):
		return self.rdivdiv(x)
	def divdiv(self, *args, **kwargs):
		pass
	def rdivdiv(self, *args, **kwargs):
		pass
class AMod:
	def __mod__(self, x):
		return self.mod(x)
	def __rmod__(self, x):
		return self.rmod(x)
	def mod(self, *args, **kwargs):
		pass
	def rmod(self, *args, **kwargs):
		pass
class AAnd:
	def __and__(self, x):
		return self.and_(x)
	def __rand__(self, x):
		return self.rand(x)
	def and_(self, *args, **kwargs):
		pass
	def rand(self, *args, **kwargs):
		pass
class AOr:
	def __or__(self, x):
		return self.or_(x)
	def __ror__(self, x):
		return self.ror(x)
	def or_(self, *args, **kwargs):
		pass
	def ror(self, *args, **kwargs):
		pass
class AXor:
	def __xor__(self, x):
		return self.xor(x)
	def __rxor__(self, x):
		return self.rxor(x)
	def xor(self, *args, **kwargs):
		pass
	def rxor(self, *args, **kwargs):
		pass
class ALsh:
	def __lshift__(self, x):
		return self.lsh(x)
	def __rlshift__(self, x):
		return self.rlsh(x)
	def lsh(self, *args, **kwargs):
		pass
	def rlsh(self, *args, **kwargs):
		pass
class ARsh:
	def __rshift__(self, x):
		return self.rsh(x)
	def __rrshift__(self, x):
		return self.rrsh(x)
	def rsh(self, *args, **kwargs):
		pass
	def rrsh(self, *args, **kwargs):
		pass
class ANeg:
	def __neg__(self):
		return self.neg()
	def neg(self, *args, **kwargs):
		pass
class ANot:
	def __invert__(self):
		return self.not_()
	def not_(self, *args, **kwargs):
		pass
# ================================================================ #
class Obj(AStr, ACode):
	def meta(self):
		return self.__class__.__class__
	def metaname(self):
		return self.meta().__qualname__
	def type(self):
		return self.__class__
	def typename(self):
		return self.type().__qualname__
	def str(self):
		return f'{self.typename()}({self.argstr()})'
	def code(self):
		return f'{self.typename()}({self.argcode()})'
	def argstr(self):
		return f'<{id(self)}>'
	def argcode(self):
		return f'<{id(self)}>'

def wrap(x, r=NA):
	attrs = ['__name__', '__qualname__', '__module__', '__doc__', '__annotations__']
	attrs = [i for i in attrs if hasattr(x, i)]
	def w(r):
		for i in attrs: setattr(r, i, getattr(x, i))
		return r
	if nna(r): return w(r)
	return w
class Decor(Obj):
	def __call__(self, x):
		if isFunc(x):
			assert self._enab_func()
			assert self._check_func(x)
			return self._wrap_func(x)
		elif isType(x):
			assert self._enab_type()
			assert self._check_type(x)
			return self._wrap_type(x)
		assert FALSE

	def _enab_func(self):
		return TRUE
	def _check_func(self, f):
		return TRUE
	def _wrap_func(self, f):
		@wrap(f)
		def w(*args, **kwargs):
			return self._do_func(f, *args, **kwargs)
		return w
	def _do_func(self, f, *args, **kwargs):
		return f(*args, **kwargs)

	def _enab_type(self):
		return TRUE
	def _check_type(self, t):
		return TRUE
	def _wrap_type(self, t):
		f = t.__new__
		@wrap(f)
		def w(cls, *args, **kwargs):
			return self._do_type(f, cls, *args, **kwargs)
		t.__new__ = w
		return t
	def _do_type(self, f, cls, *args, **kwargs):
		return f(cls, *args, **kwargs)
class DecorFunc(Decor):
	def _enab_type(self):
		return FALSE
class DecorType(Decor):
	def _enab_func(self):
		return FALSE

class Attr(DecorFunc):
	KEY = 1
	VER = 2
	INIT = 3
	GET = 4
	SET = 5

	def _do_func(self, f, *args, **kwargs):
		if len(args)==1:
			assert self._enab_get()
			x, = args
			return self._do_get(f, x, **kwargs)
		elif len(args)==2:
			assert self._enab_set()
			x, v = args
			return self._do_set(f, x, v, **kwargs)
		assert FALSE

	def _enab_get(self):
		return FALSE
	def _do_get(self, f, x, **kwargs):
		k = self._key(f, x, **kwargs)
		v = NA
		if self._enab_ver():
			b, v = self._do_ver(f, x, k, **kwargs)
			if not b:
				t = f(x, **kwargs, attr=Attr.INIT)
				self._set(x, k, t)
		if na(v): v = self._get(x, k)
		return v

	def _enab_set(self):
		return FALSE
	def _do_set(self, f, x, v, **kwargs):
		k = self._key(f, x, **kwargs)
		self._set(x, k, v)
		return x

	def _key(self, f, x, **kwargs):
		return self.fmt().format(f.__name__)

	def _enab_ver(self):
		return FALSE
	def _do_ver(self, f, x, k, **kwargs):
		v = self._get(x, k)
		b = nna(v)
		return b, v
	def _reg_ver(self, t):
		if isBool(t): return t, NA
		elif isTuple(t):
			assert len(t)==2
			return t
		assert FALSE

	@classmethod
	def _get(cls, x, k, v=NA):
		return getattr(x, k, v)
	@classmethod
	def _set(cls, x, k, v):
		setattr(x, k, v)

	@classmethod
	def fmt(cls):
		return '$attr.{}'

	@classmethod
	def get(cls, x, k, v=NA):
		k = cls.fmt().format(k)
		return cls._get(x, k, v)
	@classmethod
	def set(cls, x, k, v):
		k = cls.fmt().format(k)
		cls._set(x, k, v)
		return cls
	@classmethod
	def del_(cls, x, k):
		k = cls.fmt().format(k)
		cls._set(x, k, NA)
		return cls
	@classmethod
	def pop(cls, x, k, v=NA):
		k = cls.fmt().format(k)
		r = cls._get(x, k, v)
		cls._set(x, k, NA)
		return r

	@classmethod
	def getn(cls, x, k, v=NA):
		return cls._get(x, k, v)
	@classmethod
	def setn(cls, x, k, v):
		cls._set(x, k, v)
		return cls
	@classmethod
	def deln(cls, x, k):
		cls._set(x, k, NA)
		return cls
	@classmethod
	def popn(cls, x, k, v=NA):
		r = cls._get(x, k, v)
		cls._set(x, k, NA)
		return r
class AttrKey(Attr):
	def _key(self, f, x, **kwargs):
		r = f(x, **kwargs, attr=Attr.KEY)
		assert isStr(r)
		return self.fmt().format(r)
class AttrVer(Attr):
	def _enab_ver(self):
		return TRUE
class AttrVerf(AttrVer):
	def _do_ver(self, f, x, k, **kwargs):
		b, v = AttrVer._do_ver(self, f, x, k, **kwargs)
		if b:
			t = f(x, k, **kwargs, attr=Attr.VER)
			if nna(t):
				tb, tv = self._reg_ver(t)
				b = nav(tb, b)
				if b: v = nav(tv, v)
				else: v = NA
		return b, v
class AttrVerx(AttrVer):
	def _do_ver(self, f, x, k, **kwargs):
		t = f(x, k, **kwargs, attr=Attr.VER)
		if nna(t): r = self._reg_ver(t)
		else: r = AttrVer._do_ver(self, f, x, k, **kwargs)
		return r
class AttrGet(Attr):
	def _enab_get(self):
		return TRUE
class AttrGetf(AttrGet):
	def _do_get(self, f, x, **kwargs):
		v = Attr._do_get(self, f, x, **kwargs)
		r = f(x, v, **kwargs, attr=Attr.GET)
		if na(r): r = v
		return r
class AttrSet(Attr):
	def _enab_set(self):
		return TRUE
class AttrSetf(AttrSet):
	def _do_set(self, f, x, v, **kwargs):
		k = self._key(f, x, **kwargs)
		r = f(x, v, **kwargs, attr=Attr.SET)
		if na(r): r = v
		self._set(x, k, r)
		return x

class AttrGetSetf(AttrGet, AttrSetf):
	pass
class AttrVerGet(AttrVer, AttrGet):
	pass
class AttrVerGetfSetf(AttrVer, AttrGetf, AttrSetf):
	pass
class AttrVerGetSetf(AttrVer, AttrGet, AttrSetf):
	pass
class AttrVerfGet(AttrVerf, AttrGet):
	pass
class AttrVerxGetf(AttrVerx, AttrGetf):
	pass
class AttrKeyVerGet(AttrKey, AttrVer, AttrGet):
	pass
class AttrKeyVerxGet(AttrKey, AttrVerx, AttrGet):
	pass

attrg = AttrGet()
attrgsf = AttrGetSetf()
attrvg = AttrVerGet()
attrvgfsf = AttrVerGetfSetf()
attrvgsf = AttrVerGetSetf()
attrvfg = AttrVerfGet()
attrvxgf = AttrVerxGetf()
attrkvg = AttrKeyVerGet()
attrkvxg = AttrKeyVerxGet()

class Single(DecorType):
	def _do_type(self, f, cls, *args, **kwargs):
		x = cls
		k = f'_{cls.__name__}'
		if na(getattr(x, k, NA)):
			t = f(cls, *args, **kwargs)
			setattr(x, k, t)
		return getattr(x, k)
single = Single()

class Meta(DecorType):
	def _check_type(self, t):
		return isMeta(t)
	def _do_type(self, f, mcs, name=NA, bases=NA, attrs=NA):
		assert nna(name)
		if na(bases) or na(attrs):
			cls = name
			name = cls.__name__
			bases = cls.__bases__
			attrs = dict(cls.__dict__)
			attrs['__qualname__'] = cls.__qualname__
			if '__doc__' in attrs and na(attrs['__doc__']): del attrs['__doc__']
			for i in ['__dict__', '__weakref__']:
				if i in attrs: del attrs[i]
		return type.__new__(mcs, name, bases, attrs)
meta = Meta()

class Struct(Obj):
	def str(self):
		return self.text().str()
	def code(self):
		return self.text().code()
	def argstr(self):
		return self.argtext().str()
	def argcode(self):
		return self.argtext().code()
	@attrvg
	def text(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return Text(self)
	@attrvg
	def argtext(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return self._argtext_init()
	def _argtext_init(self):
		pass
class Const(Struct, AHash):
	@attrvg
	def hash(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return hash(self.code())
class Var(Struct):
	@attrvg
	def tok(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return 0
	def _change(self):
		Attr.set(self, 'tok', self.tok()+1)
# ================================================================ #
class Text(Obj):
	def __init__(self, *args, **kwargs):
		if len(kwargs)<=0:
			if len(args)==1:
				x = args[0]
				t = TextVer
				if isA(x, Obj):
					if isA(x, Const): t = TextConst
					elif isA(x, Var): t = TextVar
				elif na(x): t = TextNa
				elif isBool(x): t = TextBool
				elif isNum(x):
					if isInt(x): t = TextInt
					elif isReal(x): t = TextReal
				elif isStr(x): t = TextStr
				elif isIter(x):
					if isTuple(x): t = TextTuple
					elif isList(x): t = TextList
					elif isSet(x): t = TextSet
					elif isDict(x): t = TextDict
				elif isType(x): t = TextType
				elif isFunc(x): t = TextFunc
			else:
				x = args
				t = TextArgs
		else:
			if len(args)<=0:
				x = kwargs
				t = TextKwargs
			else:
				x = (args, kwargs)
				t = TextArgsKwargs
		Attr.set(self, 'obj', x)
		self.__class__ = t

	def str(self):
		return self.text(text='str')
	def code(self):
		return self.text(text='code')

	@attrg
	def obj(self, kv=NA, attr=NA):
		pass

	@attrvg
	def info(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return self._info_init()
	def _info_init(self):
		return self.obj()

	@attrvgfsf
	def params(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return self._params_dft()
		elif attr==Attr.GET: return VDict(kv)
		elif attr==Attr.SET: return self._params_set(CDict(kv).dict())
	def _params_set(self, params):
		r = self.params().dict()
		t = {k: v for k, v in params.items() if k in r and v!=r[k]}
		if len(t)>0:
			r.update(t)
			self._text_del()
		return self._params_sets(r, t, params, FALSE)
	def _params_sets(self, r, t, params, copy):
		return r
	@classmethod
	def _params_dft(cls, d=NA):
		r = cls._PARAMS_DFT.copy()
		if nna(d): r.update(d)
		return r
	_PARAMS_DFT = {}

	@attrkvg
	def text(self, kv=NA, attr=NA, text=NA):
		if attr==Attr.KEY: return self._text_key(text)
		elif attr==Attr.INIT: return self._text_init(text)
	def _text_key(self, text):
		return f'text.{text}'
	def _text_init(self, text):
		x = self.obj()
		if text=='str': return str(x)
		return repr(x)
	def _text_del(self):
		Attr.del_(self, self._text_key('str'))
		Attr.del_(self, self._text_key('code'))
		return self

class TextType(Text):
	def _text_init(self, text):
		x = self.obj()
		if x is Na: return 'Na'
		elif x is real: return 'real'
		return x.__qualname__
class TextFunc(Text):
	pass
class TextNa(Text):
	def _text_init(self, text):
		if text=='str': return 'N'
		return 'NA'
class TextBool(Text):
	def _text_init(self, text):
		x = self.obj()
		if text=='str': return 'F' if false(x) else 'T'
		return 'FALSE' if false(x) else 'TRUE'
class TextNum(Text):
	def _info_int(self, x):
		if x>0: sign, vint = 1, x
		elif x<0: sign, vint = -1, -x
		else: sign, vint = 0, 0
		return sign, vint
	def _text_int(self, sign, vint, params, text):
		b = text=='str'
		if sign<0: ssign = '-'
		elif b and sign>0 and params['sign']: ssign = '+'
		else: ssign = ''
		if b and params['psign'] and ssign=='': ssign = ' '
		sint = str(vint)
		if b:
			nint = params['nint']
			if nint>len(sint):
				sint = params['pint']*(nint-len(sint))+sint
		return ssign+sint
	def _text_prob(self, vprob, params, text):
		if text=='str':
			sprob = str(round(vprob, params['rprob'])).split('.', 1)[-1]
			nprob = params['nprob']
			if nprob>len(sprob):
				sprob += params['pprob']*(nprob-len(sprob))
			return sprob
		return str(vprob).split('.', 1)[-1]
class TextInt(TextNum):
	_PARAMS_DFT = {
		'sign': FALSE,
		'psign': FALSE,
		'nint': 0,
		'pint': ' ',
	}
	def _info_init(self):
		return self._info_int(self.obj())
	def _text_init(self, text):
		sign, vint = self.info()
		params = self.params().dict()
		return self._text_int(sign, vint, params, text)
class TextReal(TextNum):
	_PARAMS_DFT = TextInt._params_dft({
		'rprob': 4,
		'nprob': 0,
		'pprob': ' ',
	})
	def _info_init(self):
		x = self.obj()
		if not isFIN(x): return x
		b = x<0
		if b: x = -x
		i = int(x)
		vprob = round(x-i, 15)
		if b: i = -i
		sign, vint = self._info_int(i)
		if sign==0: sign = boolv(b, 1, -1)
		return sign, vint, vprob
	def _text_init(self, text):
		info = self.info()
		if not isTuple(info):
			if isNAN(info): return 'NAN'
			return 'INF' if info>0 else '-INF'
		sign, vint, vprob = info
		params = self.params().dict()
		tint = self._text_int(sign, vint, params, text)
		tprob = self._text_prob(vprob, params, text)
		return tint+'.'+tprob
class TextStr(Text):
	_PARAMS_DFT = {
		'lev': 0,
	}
	def _text_init(self, text):
		x = self.obj()
		params = self.params().dict()
		if text=='str' and params['lev']<=0: return x
		return repr(x)

class TextVer(Text):
	@attrvg
	def _ver(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return self._ver_init()
	def _ver_init(self):
		return self.info()
	@attrkvxg
	def text(self, kv=NA, attr=NA, text=NA):
		if attr==Attr.KEY: return self._text_key(text)
		elif attr==Attr.VER: return self._text_ver(kv, text)
		elif attr==Attr.INIT: return self._text_init(text)
	def _text_ver(self, key, text):
		t = Attr.getn(self, key)
		if na(t): return FALSE
		return self._text_vers(key, text)
	def _text_vers(self, key, text):
		return FALSE

class Texts(TextVer):
	_PARAMS_DFT = {
		'lev': 0,
		'ntab': 0,
		'tab': '\t',
		'com': ',',
		'sep': ' ',
		'sepx': '\n',
		'ext': FALSE,
		'exts': '',
	}
	def _params_sets(self, r, t, params, copy):
		r, t, params, copy = self._params_setx(r, t, params, copy)
		self._params_info(self.info(), params)
		return r
	def _params_setx(self, r, t, params, copy):
		if not copy: params, copy = params.copy(), TRUE
		exts = r['exts']
		if len(exts)>0:
			ext = exts[0]!='0'
			if r['ext']!=ext:
				r['ext'] = ext
				if len(t)<=0: self._text_del()
			params['ext'] = ext
			params['exts'] = exts[1:]
		params['ntab'] = r['ntab']+(0 if not r['ext'] else self._params_ntab())
		params['lev'] = r['lev']+1
		return r, t, params, copy
	def _params_info(self, info, params):
		pass
	def _params_ntab(self):
		return 0
	def _text_vers(self, key, text):
		return all(i._text_ver(key, text) for i in self._ver())
	def _text_init(self, text):
		info = self.info()
		params = self.params().dict()
		return self._text_inits(info, params, text)
	def _text_inits(self, info, params, text):
		b = text=='str'
		bell = FALSE
		nell = 0
		if b and 'nstr' in params:
			nstr = params['nstr']
			if nna(nstr) and nstr<len(info):
				bell = TRUE
				nell = len(info)-nstr
				info = info[:nstr]
		t = [self._text_info(i, params, text) for i in info]
		if bell:
			ell = params['ell']
			if params['nell']:
				ell += f'<{nell}>'
			t.append(ell)
		ext = params['ext']
		if ext:
			ntab = params['ntab'] + self._params_ntab()
			if ntab>0:
				tabs = params['tab']*ntab
				t = [tabs+i for i in t]
		sep = params['sep'] if not ext else params['sepx']
		r = (params['com']+sep).join(t)
		return r
	def _text_info(self, i, params, text):
		return ''
class TextArgs(Texts):
	_PARAMS_DFT = Texts._params_dft({
		'nstr': NA,
		'ell': '...',
		'nell': TRUE,
	})
	def _info_init(self):
		info = tuple(Text(i) for i in self.obj())
		params = self.params().dict()
		_, _, params, _ = self._params_setx(params, {}, params, FALSE)
		self._params_info(info, params)
		return info
	def _ver_init(self):
		return tuple(i for i in self.info() if isA(i, TextVer))
	def _params_info(self, info, params):
		for i in info: i.params(params)
	def _text_info(self, i, params, text):
		return i.text(text=text)
class TextKwargs(Texts):
	_PARAMS_DFT = TextArgs._params_dft({
		'ass': '=',
	})
	def _info_init(self):
		info = tuple((k, Text(v)) for k, v in self.obj().items())
		params = self.params().dict()
		_, _, params, _ = self._params_setx(params, {}, params, FALSE)
		self._params_info(info, params)
		return info
	def _ver_init(self):
		return tuple(v for _, v in self.info() if isA(v, TextVer))
	def _params_info(self, info, params):
		for _, v in info: v.params(params)
	def _text_info(self, i, params, text):
		k, v = i
		return k+params['ass']+v.text(text=text)
class TextArgsKwargs(Texts):
	def _info_init(self):
		args, kwargs = self.obj()
		info = (Text(*args), Text(**kwargs))
		params = self.params().dict()
		_, _, params, _ = self._params_setx(params, {}, params, FALSE)
		self._params_info(info, params)
		return info
	def _params_setx(self, r, t, params, copy):
		r, t, params, copy = Texts._params_setx(self, r, t, params, copy)
		params['ntab'] = r['ntab']
		params['lev'] = r['lev']
		return r, t, params, copy
	def _params_info(self, info, params):
		args, kwargs = info
		args.params(params)
		kwargs.params(params)
	def _text_info(self, i, params, text):
		p = i.params().dict()
		t = i.text(text=text)
		if isA(i, Texts) and not isA(i, TextCont) and p['ext']:
			ntab = p['ntab']
			if ntab>0: t = t.split(p['tab']*ntab, 1)[-1]
		return t

class TextsVer(Texts):
	@attrvfg
	def _vobj(self, kv=NA, attr=NA):
		if attr==Attr.VER: return self._vobj_ver(kv)
		elif attr==Attr.INIT:
			self._vobj_del()
			self._info_del()
			self._ver_del()
			self._text_del()
			return self._vobj_init()
	def _vobj_ver(self, key):
		pass
	def _vobj_init(self):
		pass
	def _vobj_del(self):
		Attr.del_(self, '_vobj')

	@attrvfg
	def info(self, kv=NA, attr=NA):
		if attr==Attr.VER: return self._info_ver(kv)
		elif attr==Attr.INIT: return self._info_init()
	def _info_ver(self, key):
		self._vobj()
		return nna(Attr.getn(self, key))
	def _info_init(self):
		vx = self._vobj()
		vinfo = self._vinfo()
		if 'info' in vinfo: info = vinfo.pop('info')
		else: info = NA
		if nna(info): self._vinfo_update(vinfo, info)
		params = self.params().dict()
		_, _, params, _ = self._params_setx(params, {}, params, FALSE)
		r, rt = self._vinfo_find(vinfo, vx)
		self._params_infos(rt, params)
		return tuple(r)
	def _info_del(self):
		info = Attr.pop(self, 'info')
		vinfo = self._vinfo()
		vinfo['info'] = info

	@attrvg
	def _vinfo(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return {}
	def _vinfo_update(self, vinfo, info):
		pass
	def _vinfo_find(self, vinfo, vx):
		return NA, NA

	@attrvfg
	def _ver(self, kv=NA, attr=NA):
		if attr==Attr.VER: return self._ver_ver(kv)
		elif attr==Attr.INIT: return self._ver_init()
	def _ver_ver(self, key):
		self.info()
		return nna(Attr.getn(self, key))
	def _ver_del(self):
		Attr.del_(self, '_ver')

	def _params_infos(self, info, params):
		pass

	def _text_vers(self, key, text):
		ver = self._ver()
		if na(Attr.getn(self, key)): return FALSE
		return all(i._text_ver(key, text) for i in ver)
class TextArgsVer(TextArgs, TextsVer):
	_PARAMS_DFT = TextArgs._params_dft()
	def _vobj_ver(self, key):
		x = self.obj()
		vx = Attr.getn(self, key)
		if len(x)!=len(vx): return FALSE
		for n, i in enumerate(x):
			if i is not vx[n]: return FALSE
		return TRUE
	def _vobj_init(self):
		return tuple(self.obj())
	def _info_init(self):
		return TextsVer._info_init(self)
	def _vinfo_update(self, vinfo, info):
		for i in info: vinfo[id(i.obj())] = i
	def _vinfo_find(self, vinfo, vx):
		r, rt = [], []
		for i in vx:
			t = vinfo.get(id(i), NA)
			if na(t):
				t = Text(i)
				rt.append(t)
			r.append(t)
		return r, rt
	def _params_info(self, info, params):
		return TextArgs._params_info(self, info, params)
	def _params_infos(self, info, params):
		return TextArgs._params_info(self, info, params)
class TextKwargsVer(TextKwargs, TextsVer):
	_PARAMS_DFT = TextArgsVer._params_dft({
		'bind': ': ',
	})
	def _vobj_ver(self, key):
		x = self.obj()
		vx = Attr.getn(self, key)
		if len(x)!=len(vx): return FALSE
		for n, (k, v) in enumerate(x.items()):
			vk, vv = vx[n]
			if k is not vk or v is not vv: return FALSE
		return TRUE
	def _vobj_init(self):
		return tuple(self.obj().items())
	def _info_init(self):
		return TextsVer._info_init(self)
	def _vinfo_update(self, vinfo, info):
		for k, v in info:
			vinfo[id(k.obj())] = k
			vinfo[id(v.obj())] = v
	def _vinfo_find(self, vinfo, vx):
		r, rt = [], []
		for k, v in vx:
			tk = vinfo.get(id(k), NA)
			tv = vinfo.get(id(v), NA)
			if na(tk):
				tk = Text(k)
				rt.append(tk)
			if na(tv):
				tv = Text(v)
				rt.append(tv)
			r.append((tk, tv))
		return r, rt
	def _ver_init(self):
		info = self.info()
		r = []
		for k, v in info:
			if isA(k, TextVer): r.append(k)
			if isA(v, TextVer): r.append(v)
		return tuple(r)
	def _params_info(self, info, params):
		for k, v in info:
			k.params(params)
			v.params(params)
	def _params_infos(self, info, params):
		return TextArgs._params_info(self, info, params)
	def _text_info(self, i, params, text):
		k, v = i
		return k.text(text=text)+params['bind']+v.text(text=text)

class TextCont(Texts):
	def _params_ntab(self):
		return 1
	def _text_init(self, text):
		info = self.info()
		params = self.params().dict()
		a, b = self._text_brac(info, params, text)
		t = self._text_inits(info, params, text)
		if params['ext']:
			if isTuple(info): k, f = 1, lambda info: len(info)>0
			elif isA(info, Texts) and not isA(info, TextCont):
				k, f = 2, lambda info: len(info.info())>0
			else: k, f = 0, NA
			if k>0:
				sep = params['sepx']
				if f(info): t += params['com']+sep
				t = sep+t
				ntab = params['ntab']
				if ntab>0: t += params['tab']*ntab
		return a+t+b
	def _text_brac(self, info, params, text):
		return '', ''
class TextTuple(TextCont, TextArgs):
	def _text_brac(self, info, params, text):
		if text!='str' and len(info)==1 and not params['ext']:
			return '(', ',)'
		return '(', ')'
class TextList(TextCont, TextArgsVer):
	def _text_brac(self, info, params, text):
		return '[', ']'
class TextSet(TextCont, TextArgsVer):
	def _text_brac(self, info, params, text):
		if text!='str' and len(info)<=0:
			return 'set(', ')'
		return '{', '}'
class TextDict(TextCont, TextKwargsVer):
	def _text_brac(self, info, params, text):
		if text=='str' and len(info)<=0:
			return '{:', '}'
		return '{', '}'

class TextStruct(TextCont):
	def _info_init(self):
		info = self.obj().argtext()
		params = self.params().dict()
		_, _, params, _ = self._params_setx(params, {}, params, FALSE)
		self._params_info(info, params)
		return info
	def _ver_init(self):
		return not isA(self.info(), TextVer)
	def _params_setx(self, r, t, params, copy):
		r, t, params, copy = Texts._params_setx(self, r, t, params, copy)
		info = self.obj().argtext()
		params['exts'] = r['exts']
		params['ext'] = r['ext']
		params['ntab'] = r['ntab']+(0 if isA(info, TextCont) else 1)
		params['lev'] = r['lev']+1
		return r, t, params, copy
	def _params_info(self, info, params):
		info.params(params)
	def _text_vers(self, key, text):
		return self._ver()
	def _text_inits(self, info, params, text):
		return info.text(text=text)
	def _text_brac(self, info, params, text):
		return f'{self.obj().typename()}(', ')'
class TextConst(TextStruct):
	pass
class TextVar(TextStruct):
	pass
# ================================================================ #
class Dict(Struct, ASize, AHas, AIter, AGet):
	@classmethod
	def reg(cls, *args, f=NA):
		if len(args)<=0: return NA, NA
		elif len(args)==1:
			x, = args
			if na(x): return NA, NA
			elif isA(x, Dict): return x.dict(), NA
			elif isDict(x): return x, NA
			elif isTuple(x): return cls.reg(*x, f=f)
			elif isIter(x):
				r = []
				for i in x:
					if not isTuple(i) or len(i)!=2: i = (i, NA)
					r.append(i)
				return dict(r), NA
			return x, NA
		elif len(args)==2:
			k, v = args
			if na(k) or isIter(k):
				if isIter(v):
					def _f(f, v):
						if na(f): f = 'N'
						if isStr(f):
							t = len(v)
							f = {
								'N':lambda n, i:v[n] if n<t else NA,
								'O':lambda n, i:v[n%t],
								'L':lambda n, i:v[n if n<t else -1],
							}.get(f.upper(), NA) if t>0 else NA
						elif isCall(f): f = f(v)
						else: f = NA
						return f
					v = _f(f, list(v))
				if isIter(k):
					if isCall(v): k, v = {i:v(n, i) for n, i in enumerate(k)}, NA
					else: k, v = {i:v for i in k}, NA
			return k, v
		assert FALSE

	def _init(self, *args, f=NA):
		k, v = self.reg(*args, f=f)
		if na(k): r = {}
		elif isDict(k): r = k
		else: r = {k: v}
		Attr.set(self, 'dict', r)
	def _argtext_init(self):
		return Text(self.dict())

	@attrg
	def dict(self, kv=NA, attr=NA):
		pass

	def size(self, n=NA):
		d = self.dict()
		if not n: r = len(d)
		else: r = size(0 for i in d.values() if nna(i))
		return r
	def has(self, *args, n=NA):
		d = self.dict()
		k, v = self.reg(*args)
		if na(k):
			if not n: r = TRUE
			else: r = all(nna(i) for i in d.values())
		elif isDict(k):
			if not n: r = all(i in d for i in k)
			else: r = all(nna(d.get(i, NA)) for i in k)
		else:
			if not n: r = k in d
			else: r = nna(d.get(k, NA))
		return r
	def iter(self, t=NA):
		r = ((k, v) for k, v in self.dict().items())
		if nna(t):
			assert isCall(t)
			r = t(r)
		return r
	def key(self, t=NA):
		r = (i for i in self.dict().keys())
		if nna(t):
			assert isCall(t)
			r = t(r)
		return r
	def val(self, t=NA):
		r = (i for i in self.dict().values())
		if nna(t):
			assert isCall(t)
			r = t(r)
		return r
	def items(self):
		return self.iter()
	def keys(self):
		return self.key()
	def values(self):
		return self.val()

	def get(self, *args, f=NA, n=NA):
		d = self.dict()
		k, v = self.reg(*args, f=f)
		r = self._get(d, k, v, n=n)
		return r
	def _get(self, d, k, v, n=NA):
		if na(k):
			if not n: r = d.copy()
			else:
				if isCall(v):
					r = {ik: iv if nna(iv) else v(n, ik) for n, (ik, iv) in enumerate(d.items())}
				else: r = {ik: iv if nna(iv) else v for ik, iv in d.items()}
			r = self.type()(r)
		elif isDict(k):
			if not n: r = {ik: d.get(ik, iv) for ik, iv in k.items()}
			else:
				r = ((ik, d.get(ik, NA), iv) for ik, iv in k.items())
				r = {ik: ivt if nna(ivt) else iv for ik, ivt, iv in r}
			r = self.type()(r)
		else:
			if not n: r = d.get(k, v)
			else:
				r = d.get(k, NA)
				if na(r): r = v
		return r
	def copy(self, n=NA):
		d = self.dict()
		if not n: r = d.copy()
		else: r = {ik: iv for ik, iv in d.items() if nna(iv)}
		return self.type()(r)
	def sort(self, k=NA, r=NA):
		return self.type()(sort(self.dict(), k=k, r=r))
class CDict(Const, Dict):
	def __init__(self, *args, f=NA):
		self._init(*args, f=f)
class VDict(Var, Dict, ASet, ADel, APop):
	def __init__(self, *args, f=NA):
		self._init(*args, f=f)
	def set(self, *args, f=NA, n=NA):
		d = self.dict()
		k, v = self.reg(*args, f=f)
		self._set(d, k, v, n=n)
		return self
	def _set(self, d, k, v, n=NA):
		if na(k):
			if not n:
				if isCall(v):
					for n, i in enumerate(d): d[i] = v(n, i)
				else:
					for i in d: d[i] = v
			else:
				if isCall(v):
					for n, (ik, iv) in enumerate(d.items()):
						if na(iv): d[ik] = v(n, ik)
				else:
					for ik, iv in d.items():
						if na(iv): d[ik] = v
		elif isDict(k):
			if not n:
				for ik, iv in k.items(): d[ik] = iv
			else:
				for ik, iv in k.items():
					if na(d.get(ik, NA)): d[ik] = iv
		else:
			if not n: d[k] = v
			else:
				t = d.get(k, NA)
				if na(t): d[k] = v
		self._change()
	def del_(self, *args, n=NA):
		d = self.dict()
		k, v = self.reg(*args)
		self._del(d, k, n=n)
		return self
	def _del(self, d, k, n=NA):
		if na(k):
			if not n: d.clear()
			else:
				t = list(d.keys())
				for i in t:
					if na(d[i]): del d[i]
		elif isDict(k):
			if not n:
				for i in k:
					if i in d: del d[i]
			else:
				for i in k:
					if i in d and na(d[i]): del d[i]
		else:
			if not n:
				if k in d: del d[k]
			else:
				if k in d and na(d[k]): del d[k]
		self._change()
	def pop(self, *args, f=NA, n=NA):
		d = self.dict()
		k, v = self.reg(*args, f=f)
		r = self._get(d, k, v, n=n)
		self._del(d, k, n=n)
		return r
	def clear(self, n=NA):
		self._del(self.dict(), NA, n=n)
		return self
# ================================================================ #
import os as _os
import shutil as _shutil
import pickle as _pickle
class Path(Const, AComp, ADiv):
	class Sym(Const, AComp):
		def __init__(self, x):
			if isA(x, Path.Sym):
				x, t, r = x.sym(), x.tok(), x.type()
			else:
				assert isStr(x) and '/' not in x and '\\' not in x
				if x=='' or x[0]=='~': r, t = Path.Key, 0
				elif x[0]=='$':
					if len(x)<=1 or x[1] not in '~$':
						r, t = Path.Key, 1
					else: r, t = Path.Esc, 2
				elif x=='.': r, t = Path.Sym, NA
				elif x=='..': r, t = Path.Str, -1
				else: r, t = Path.Str, 3
			Attr.set(self, 'sym', x)
			Attr.set(self, 'tok', t)
			self.__class__ = r
		def _argtext_init(self):
			return Text(self.sym())
		def comp(self, x):
			if isA(x, Path.Sym):
				return comp(self.sym(), x.sym())
			return NAN
		@attrg
		def sym(self, kv=NA, attr=NA):
			pass
		@attrg
		def tok(self, kv=NA, attr=NA):
			pass
		@attrvg
		def map(self, kv=NA, attr=NA):
			if attr==Attr.INIT: return self._map_init()
		def _map_init(self):
			pass
	class Key(Sym):
		@attrvxgf
		def map(self, kv=NA, attr=NA):
			if attr==Attr.VER:
				return self._pathtok()==Path.path().tok() and self.sym() in self._path()
			elif attr==Attr.INIT:
				x = self.sym()
				path = Path.path()
				if x not in path: t = ()
				else: t = Path.reg(path[x], map=TRUE)
				assert all(isA(i, Path.Esc, Path.Str) for i in t)
				self._path()[x] = t
				Attr.set(self, '_pathtok', path.tok())
			elif attr==Attr.GET:
				return self._path()[self.sym()]
		@attrvg
		def _pathtok(self, kv=NA, attr=NA):
			if attr==Attr.INIT: return Path.path().tok()
		@classmethod
		@attrvg
		def _path(cls, kv=NA, attr=NA):
			if attr==Attr.INIT: return {}
	class Esc(Sym):
		def _map_init(self):
			return self.sym()[1:]
	class Str(Sym):
		def _map_init(self):
			return self.sym()

	@classmethod
	@attrvgsf
	def path(cls, kv=NA, attr=NA):
		if attr==Attr.INIT: return VDict()
		elif attr==Attr.SET:
			return cls.path().set(kv)
	@classmethod
	def reg(cls, *args, map=NA):
		if len(args)<=0: return ()
		elif not map and len(args)==1 and isA(args[0], Path):
			return args[0]._path()
		def _f(x):
			if isA(x, Path):
				for i in reversed(x._path()):
					for j in _f(i): yield j
			elif isA(x, Path.Key):
				if not map: yield x
				else:
					for i in reversed(x.map()):
						for j in _f(i): yield j
			elif isA(x, Path.Sym): yield x
			elif isArr(x):
				for i in reversed(x):
					for j in _f(i): yield j
			elif isStr(x):
				x = x.replace('\\', '/').split('/')
				b = TRUE
				for i in reversed(x):
					i = i.strip()
					if b:
						b = FALSE
						if i=='': continue
					yield Path.Sym(i)
			else: assert FALSE
		r = []
		for i in _f(args):
			t = i.tok()
			if na(t): continue
			elif t>=2 and len(r)>0 and r[-1].tok()==-1: r.pop()
			else: r.append(i)
			if t==0: break
		return tuple(reversed(r))

	def __init__(self, *args):
		Attr.set(self, '_path', self.reg(*args))

	def _argtext_init(self):
		return Text('/'.join(i.sym() for i in self._path()) or '.')

	@attrg
	def _path(self, kv=NA, attr=NA):
		pass
	@attrvg
	def _pathtok(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return self.path().tok()
	@attrvfg
	def pathstr(self, kv=NA, attr=NA):
		if attr==Attr.VER:
			return self._pathtok()==self.path().tok()
		elif attr==Attr.INIT:
			r = self.reg(*self._path(), map=TRUE)
			assert all(isA(i, self.Esc, self.Str) for i in r)
			r = '/'.join(i.map() for i in r) or '.'
			Attr.set(self, '_pathtok', self.path().tok())
			return r

	def comp(self, *args):
		try:
			x = Path(*args)
			r = comp(self._path(), x._path())
		except: r = NAN
		return r
	def div(self, *args):
		return Path(self, *args)
	def rdiv(self, *args):
		return Path(*args, self)

	def isDir(self):
		return _os.path.isdir(self.pathstr())
	def isFile(self):
		return _os.path.isfile(self.pathstr())

	def dir(self):
		return Dir(self)
	def file(self):
		return File(self)
class _Dir_File(Const, AComp, ADiv, ADel):
	def _init(self, *args):
		Attr.set(self, 'path', Path(*args))
	def _argtext_init(self):
		return self.path().argtext()
	def comp(self, *args):
		return self.path().comp(*args)
	def div(self, *args):
		return self.path().div(*args)
	def rdiv(self, *args):
		return self.path().rdiv(*args)
	@attrg
	def path(self, kv=NA, attr=NA):
		pass
	def pathstr(self):
		return self.path().pathstr()
	@attrvg
	def name(self, kv=NA, attr=NA):
		if attr==Attr.INIT:
			return self.pathstr().rsplit('/', 1)[-1]
class Dir(_Dir_File, ASize, AHas, AIter):
	def __init__(self, *args):
		self._init(*args)
		assert not self.path().isFile()
	def size(self):
		return len(_os.listdir(self.pathstr()))
	def has(self, *args):
		x = Path(*args)
		for i in self.iter():
			if x==i: return TRUE
		return FALSE
	def iter(self):
		for i in _os.listdir(self.pathstr()):
			yield self.path().div(i)
	def add(self, mode=NA):
		if na(mode): mode = 0o777
		_os.makedirs(self.pathstr(), mode=mode, exist_ok=TRUE)
		return self
	def del_(self, mm):
		if mm=='MM123456':
			path = self.pathstr()
			_shutil.rmtree(path, ignore_errors=TRUE)
		return self
class File(_Dir_File):
	def __init__(self, *args):
		self._init(*args)
		assert not self.path().isDir()
	@attrvg
	def names(self, kv=NA, attr=NA):
		if attr==Attr.INIT:
			return self.name().rsplit('.', 1)[0]
	@attrvg
	def namex(self, kv=NA, attr=NA):
		if attr==Attr.INIT:
			return self.name().rsplit('.', 1)[-1]
	def open(self, mode):
		return open(self.pathstr(), mode=mode)
	def load(self):
		with self.open(mode='rb') as f:
			return _pickle.load(f)
	def store(self, data):
		with self.open(mode='wb') as f:
			_pickle.dump(data, f)
		return self
	def read(self, str=NA, encoding=NA):
		r = NA
		try:
			with self.open(mode='rb') as f: r = f.read()
		except Exception as e: print(e)
		if na(str): str = TRUE
		if str:
			if na(encoding): encoding = 'UTF-8'
			r = r.decode(encoding=encoding)
		return r
	def write(self, data, encoding=NA, mode=NA):
		if isStr(data):
			if na(encoding): encoding = 'UTF-8'
			data = data.encode(encoding=encoding)
		if na(mode): mode = 'wb'
		try:
			with self.open(mode=mode) as f: f.write(data)
		except Exception as e: print(e)
		return self
	def writea(self, data, encoding=NA):
		return self.write(data, encoding=encoding, mode='ab')
	def clear(self):
		try:
			with self.open(mode='wb') as f: f.write(b'')
		except Exception as e: print(e)
		return self
	def del_(self, mm):
		if mm=='MM123456':
			path = self.pathstr()
			if _os.path.exists(path): _os.remove(path)
		return self
# ================================================================ #
import time as _time
def delay(n):
	return _time.sleep(n)
class _Time_Elap(Const, AComp, AAdd, ASub):
	def _init(self, val=NA):
		if na(val): val = self._val_init()
		assert isNum(val)
		Attr.set(self, 'val', val)
	def _argtext_init(self):
		return Text(self.val())
	@attrg
	def val(self, kv=NA, attr=NA):
		pass
	@attrvg
	def valstr(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return self._valstr_init()
	@attrvgsf
	def fmt(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return self._fmt_dft()
		elif attr==Attr.SET:
			kv = nav(kv, self._fmt_dft())
			assert isStr(kv)
			if Attr.get(self, 'fmt')!=kv:
				Attr.del_(self, 'valstr')
			return kv
	def _val_init(self):
		pass
	def _valstr_init(self):
		pass
	def _fmt_dft(self):
		pass
class Time(_Time_Elap):
	def comp(self, x):
		assert isA(x, self.type())
		return comp(self.val(), x.val())
	def add(self, x):
		assert isA(x, Elap)
		return self.type()(self.val()+x.val())
	def radd(self, x):
		return self.add(x)
	def sub(self, x):
		if isA(x, self.type()): return Elap(self.val()-x.val())
		elif isA(x, Elap): return self.type()(self.val()-x.val())
		assert FALSE
	def rsub(self, x):
		assert isA(x, self.type())
		return Elap(x.val()-self.val())
class TimeDate(Time):
	def __init__(self, val=NA):
		self._init(val=val)
	@attrvg
	def info(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return _time.localtime(self.val())
	def _val_init(self):
		return _time.time()
	def _valstr_init(self):
		return _time.strftime(self.fmt(), self.info())
	def _fmt_dft(self):
		return '%Y-%m-%d %H:%M:%S'
class TimeCount(Time):
	def __init__(self, val=NA):
		self._init(val=val)
	def _val_init(self):
		return _time.perf_counter()
	def _valstr_init(self):
		return self.fmt().format(self.val())
	def _fmt_dft(self):
		return '{:.4f}'
class Elap(_Time_Elap):
	def __init__(self, val=NA):
		self._init(val=val)
	def comp(self, x):
		assert isA(x, self.type())
		return comp(self.val(), x.val())
	def add(self, x):
		assert isA(x, _Time_Elap)
		return x.type()(self.val()+x.val())
	def radd(self, x):
		return self.add(x)
	def sub(self, x):
		assert isA(x, self.type())
		return self.type()(self.val()-x.val())
	def rsub(self, x):
		assert isA(x, _Time_Elap)
		return x.type()(x.val()-self.val())
	@attrvg
	def info(self, kv=NA, attr=NA):
		if attr==Attr.INIT:
			t = self.val()
			if t<60: s = 's'
			else:
				t /= 60
				if t<60: s = 'm'
				else:
					t /= 60
					if t<24: s = 'h'
					else:
						t /= 24
						s = 'd'
			return t, s
	def _val_init(self):
		return 0
	def _valstr_init(self):
		return self.fmt().format(*self.info())
	def _fmt_dft(self):
		return '{:.4f}{}'
class Clock(Obj):
	def __init__(self, time=NA):
		time = nav(time, TimeDate)
		assert evoA(time, Time)
		Attr.set(self, 'time', time)
		self.init()
	def init(self):
		Attr.set(self, 'stat', 0)
		return self
	@attrg
	def time(self, kv=NA, attr=NA):
		pass
	@attrg
	def stat(self, kv=NA, attr=NA):
		pass
	@attrvxgf
	def beg(self, kv=NA, attr=NA):
		if attr==Attr.VER:
			return self.stat()>=1
		elif attr==Attr.INIT:
			Attr.set(self, 'stat', 1)
			return self.time()()
		elif attr==Attr.GET:
			assert self.stat()>=1
	@attrvxgf
	def end(self, kv=NA, attr=NA):
		if attr==Attr.VER:
			return self.stat()>=2
		elif attr==Attr.INIT:
			Attr.set(self, 'stat', 2)
			return self.time()()
		elif attr==Attr.GET:
			assert self.stat()>=2
	@attrvxgf
	def elap(self, kv=NA, attr=NA):
		if attr==Attr.VER:
			return self.stat()>=3
		elif attr==Attr.INIT:
			Attr.set(self, 'stat', 3)
			return self.end()-self.beg()
		elif attr==Attr.GET:
			assert self.stat()>=3
# ================================================================ #
class Print(Obj):
	def __call__(self, *args, **kwargs):
		r = NA
		p = self.print()
		f = self.file()
		if p or nna(f):
			params = self.params().dict()
			t = Text(*args, **kwargs)
			t = t.params(params)
			t = t.text(text=self.text())
			t += params['end']
			if not self.input():
				if p: print(t, end='')
				if nna(f): f.writea(t)
			else:
				t += params['in']
				if p: print(t, end='')
				if nna(f): f.writea(t)
				t = input()
				if nna(f): f.writea(t+'\n')
				if self.inputr(): r = t
		return r
	def do(self):
		return self.print() or nna(self.file())
	@attrvgfsf
	def params(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return self._params_dft()
		elif attr==Attr.GET: return VDict(kv)
		elif attr==Attr.SET:
			return self.params().set(kv).dict()
	def _params_dft(cls):
		return {
			'end': '\n',
			'in': '>>> ',
		}
	@attrvgsf
	def text(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return 'str'
		elif attr==Attr.SET:
			kv = nav(kv, 'str')
			assert kv in {'str', 'code'}
			return kv
	@attrvgsf
	def print(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return TRUE
		elif attr==Attr.SET:
			kv = nav(kv, TRUE)
			assert isBool(kv)
			return kv
	@attrgsf
	def file(self, kv=NA, attr=NA):
		if attr==Attr.SET:
			if nna(kv): assert isA(kv, File)
			return kv
	@attrvgsf
	def input(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return FALSE
		elif attr==Attr.SET:
			kv = nav(kv, FALSE)
			assert isBool(kv)
			return kv
	@attrvgsf
	def inputr(self, kv=NA, attr=NA):
		if attr==Attr.INIT: return FALSE
		elif attr==Attr.SET:
			kv = nav(kv, FALSE)
			assert isBool(kv)
			return kv
	def copy(self):
		r = self.type()()
		Attr.set(r, 'params', self.params().dict().copy())
		Attr.set(r, 'text', self.text())
		Attr.set(r, 'print', self.print())
		Attr.set(r, 'file', self.file())
		Attr.set(r, 'input', self.input())
		Attr.set(r, 'inputr', self.inputr())
		return r
# ================================================================ #
class PrintNameTimeElap(DecorFunc):
	def __init__(self, print=NA):
		if na(print): print = Print()
		assert isA(print, Print)
		Attr.set(self, 'print', print)
	def _do_func(self, f, *args, **kwargs):
		name = f.__qualname__
		fmt_beg, fmt_end = NA, NA
		clock = NA
		pr = self.print()
		if pr.do():
			fmt_beg, fmt_end = self.fmt()
			clock = Clock()
			s = fmt_beg.format(
				name=name,
				time=clock.beg().valstr(),
			)
			pr(s)
		r = f(*args, **kwargs)
		if pr.do():
			s = fmt_end.format(
				name=name,
				time=clock.end().valstr(),
				elap=clock.elap().valstr(),
			)
			pr(s)
		return r
	@attrg
	def print(self, kv=NA, attr=NA):
		pass
	@attrvg
	def fmt(self, kv=NA, attr=NA):
		if attr==Attr.INIT:
			fmt_l = '#'*16
			fmt_p = ' '
			fmt_s = ' | '
			fmt_name = '{name}'
			fmt_time = '{time}'
			fmt_elap = '{elap}'
			fmt_beg = fmt_l+fmt_p+fmt_s.join([fmt_time, fmt_name])+fmt_p+fmt_l
			fmt_end = fmt_l+fmt_p+fmt_s.join([fmt_time, fmt_name, fmt_elap])+fmt_p+fmt_l
			return fmt_beg, fmt_end
	def copy(self):
		r = self.type()(self.print().copy())
		Attr.set(r, 'fmt', self.fmt())
		return r
# ================================================================ #
def ic(c):
	return ord(c)
def ci(i):
	return chr(i)
def code(x):
	return repr(x)
def comp(x, y):
	if isA(x, AComp): r = x.comp(y)
	elif isA(y, AComp): r = -y.comp(x)
	elif x<y: r = -1
	elif x==y: r = 0
	elif x>y: r = 1
	else: r = NAN
	return r
def size(x):
	if hasattr(x, '__len__'): return len(x)
	r = 0
	for _ in x: r += 1
	return r
def riter(x):
	return reversed(x)
def sort(x, k=NA, r=NA):
	if not isList(x):
		if isDict(x): x = list(x.items())
		else: x = list(x)
	if isInt(k) or isStr(k):
		t = k
		k = lambda x: x[t]
	r = nav(r, FALSE)
	return sorted(x, key=k, reverse=r)
def equal(x, y=NA, eps=NA):
	y = nav(y, 0)
	eps = nav(eps, EPS)
	return abs(x-y)<eps
def mulset(x):
	r = {}
	for i in x: r[i] = r.get(i, 0)+1
	return r
def nexts(x, n=NA):
	n = nav(n, 1)
	for i in range(n): yield next(x)
def part(l, pf=NA, rf=NA):
	if na(pf): pf = lambda i, v, t: 'a'
	elif isInt(pf) and pf>0:
		n = pf
		pf = lambda i, v, t: 'yca' if len(t)>=n else 'a'
	elif isChar(pf):
		c = pf
		pf = lambda i, v, t: 'yc' if v==c else 'a'
	elif isIter(pf):
		cs = set(pf)
		pf = lambda i, v, t: 'yc' if v in cs else 'a'

	if na(rf): rf = lambda t: list(t)
	elif isStr(rf):
		s = rf
		rf = lambda t: s.join(t)

	t = []
	for i, v in enumerate(l):
		for k in pf(i, v, t):
			if k=='a': t.append(v)
			elif k=='y': yield rf(t)
			elif k=='c': t.clear()
			elif k=='Y':
				if len(t)>0: yield rf(t)
	if len(t)>0: yield rf(t)
def prime(n):
	r = []
	k = [TRUE]*n
	k[0] = FALSE
	k[1] = FALSE
	for i in range(n):
		if not k[i]: continue
		r.append(i)
		for t in range(1, iceil(n/i)):
			k[i*t] = FALSE
	return r
def prime_factor(n):
	p = prime(iceil(sqr(n)))
	r = []
	for k, i in enumerate(p):
		if n<=1: break
		t = 0
		while n%i==0:
			n //= i
			t += 1
		if t>0: r.append((i, t))
	return r
def coprime(n, m=NA):
	p = prime(nav(m, n))
	r = []
	for k, i in enumerate(p):
		if i**2>n:
			r += p[k:]
			break
		if n%i!=0: r.append(i)
	return r
# ================================================================ #
timestr_dir = TimeDate().fmt('%Y-%m-%d').valstr()
timestr_file = TimeDate().fmt('%H-%M-%S').valstr()
dir_print = Dir(f'print/{timestr_dir}')
file_print = dir_print.div(f'{timestr_file}.txt').file()
pr = Print()
pri = pr.copy().input(TRUE)
pris = pri.copy().inputr(TRUE)
pl = pr.copy().params({'end': '\n'+'='*64+'\n'})
pli = pl.copy().input(TRUE)
plis = pli.copy().inputr(TRUE)
pnte = PrintNameTimeElap(pr.copy())
