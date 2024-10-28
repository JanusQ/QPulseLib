# Control envelopes in time domain and frequency domain
#
# For a pair of functions g(t) <-> h(f) we use the following
# convention for the Fourier Transform:
#
#          / +inf
#         |
# h(f) =  | g(t) * exp(-2j*pi*f*t) dt
#         |
#        / -inf
#
#          / +inf
#         |
# g(t) =  | h(f) * exp(2j*pi*f*t) df
#         |
#        / -inf
#
# Note that we are working with frequency in GHz, rather than
# angular frequency.  Also note that the sign convention is opposite
# to what is normally taken in physics.  But this is the convention
# used here and in the DAC deconvolution code, so you should use it.

# Also, this convention is better :)
#
# To get Mathematica to use this convention, add the option FourierParameters -> {0, -2*Pi}
#
# The physics convention comes from a very unfortunate choice of sign in
# Heisenberg/Schrodinger's equation, or Schrodinger picked the bad sign
# because of unfortunate pre-existing physics conventions, not sure. - DTS

# have to do this so we get math std library
from __future__ import absolute_import
import labrad
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from labrad.units import Unit
from scipy.special import erf
# import pyle.util.cache
# from others import convertUnits
import functools
import inspect

V, mV, us, ns, GHz, MHz, dBm, rad, au = [
    Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad', 'au')
]

# enable_caching = True
# cache_hits = 0
# cache_misses = 0

# #exp = pyle.util.cache.LRU_Cache(np.exp, pyle.util.cache.keyfunc_ndarray, 1024)
exp = np.exp  # To disable caching for exponentials
_zero = lambda x: 0 * x

# def memoize_envelope(cls):
#     '''
#     Decorator for memoizing envelope classes.
#     If all of the parameters to __init__ are the same, and the function
#     is evaulated at the same frequency points, return the same data.

#     The array is set to be read-only.  This prevents accidental
#     corruption of the cache.  We could also just return a copy
#     if that turns out to be a problem.
#     '''
#     if not enable_caching:
#         return cls
#     old_init = cls.__init__
#     old_freqFunc = cls.freqFunc

#     def __init__(self, *args, **kwargs):
#         # NOTE: Caching now defaults to True
#         cache = kwargs.pop('cache', True)
#         if cache:
#             key = (args, tuple(kwargs.items()))
#             self._instance_key = key
#         old_init(self, *args, **kwargs)

#     def freqFunc(self, f):
#         # Allow global disabling of caching
#         if not enable_caching or not hasattr(self, '_instance_key'):
#             return old_freqFunc(self, f)
#         data_key = (f[0], f[-1], len(f))
#         key = (self._instance_key, data_key)
#         try:
#             x = self._cache[key]
#             global cache_hits
#             cache_hits += 1
#             return x
#         except KeyError:
#             global cache_misses
#             cache_misses += 1
#             x = old_freqFunc(self, f)
#             x.setflags(write=False)
#             self._cache[key] = x
#             return x
#         except TypeError:
#             for k in key:
#                 print('===========')
#                 print(k, type(key))
#                 print(hash(key))
#             print(
#                 f"Warning: unable to hash parameters {key} for envelope of type {cls.__name__}"
#             )
#             return old_freqFunc(self, f)

#     __init__.__doc__ = old_init.__doc__
#     freqFunc.__doc__ = (old_freqFunc.__doc__ or "Envelope Frequency Function"
#                         ) + '\n\nCaching added by decorator memoize_envelope'
#     cls._cache = {}
#     cls.freqFunc = freqFunc
#     if not hasattr(cls, '_memoize_manual_key'):
#         cls.__init__ = __init__
#     return cls

# def envelope_factory_t0(cls):
# '''
# Use only on envelopes that take a t0 as their first argument.
# This will construct envelopes at time zero and shift them, allowing
# for better caching.
# '''
# # I would like to rewrite this in a way that it modifies the class rather than
# # wrapping it in a factory, but I didn't see a clean way to do that -- ERJ
# @convertUnits(t0='ns')
# def factory(t0, *args, **kwargs):
#     if not enable_caching or t0 == 0.0:  # If caching is disabled, this doesn't help, so keep it simple
#         return cls(
#             t0, *args,
#             **kwargs)  # Also, avoid extra shift if t0 is already zero.
#     x = cls(0.0, *args, **kwargs)
#     y = shift(x, t0)
#     return y

# factory.__doc__ = 'Envelope factory for type %s.  Access original class via __klass__ attribute' % (
#     cls.__name__, )
# factory.__klass__ = cls
# return factory

def convertUnits(**unitdict):
    """
    Decorator to create functions that automatically
    convert arguments into specified units.  If a unit
    is specified for an argument and the user passes
    an argument with incompatible units, an Exception
    will be raised.  Inside the decorated function, the
    arguments no longer have any units, they are just
    plain floats.  Not all arguments to the function need
    to be specified in the decorator.  Those that are not
    specified will be passed through unmodified.

    Usage:

    @convertUnits(t0='ns', amp=None)
    def func(t0, amp):
        <do stuff>

    This is essentially equivalent to:

    def func(t0, amp):
        t0 = convert(t0, 'ns')
        amp = convert(amp, None)
        <do stuff>

    The convert function is defined internally, and will
    convert any quantities with units into the specified
    units, or strip off any units if unit is None.
    """

    def convert(v, u):
        if isinstance(v, labrad.units.Value):  # prefer over subclass check: isinstance(v, Value)
            if u is None:
                if hasattr(v, 'value'):
                    return v.value
                elif hasattr(v, '_value'):
                    return v._value
                else:
                    raise TypeError(f'Value object <{v}> has no attribute `value` or `_value`')
            else:
                return v[u]
        else:
            return v

    def wrap(f):
        args = inspect.getfullargspec(f)[0]  # list of argument names
        for arg in unitdict:
            if arg not in args:
                raise Exception('function %s does not take arg "%s"' % (f, arg))
        # unitdict maps argument names to units
        # posdict maps argument positions to units
        posdict = dict((i, unitdict[arg]) for i, arg in enumerate(args) if arg in unitdict)

        @functools.wraps(f)
        def wrapped(*a, **kw):
            # convert positional arguments if they have a unit
            a = [convert(val, posdict.get(i, None)) for i, val in enumerate(a)]
            # convert named arguments if they have a unit
            for arg, val in kw.items():
                if arg in unitdict:
                    kw[arg] = convert(val, unitdict[arg])
            # call the function with converted arguments
            return f(*a, **kw)

        return wrapped

    return wrap





# -------- Envelops ---------------
class Envelope(object):
    """Represents a control envelope as a function of time or frequency.        时间或频率的一个control命令包？

    Envelopes can be added to each other or multiplied by constant values.
    Multiplication of two envelopes and addition of a constant value (other
    than zero) are not equivalent in time and fourier domains, so these
    operations are not supported.                                               得去看一下傅里叶变换，和两个包相乘的关系

    Envelopes keep track of their start and end time, and when added            是不是可以认为是一个时间戳的集合
    together the new envelope will use the earliest start and latest end,
    to cover the entire range of its constituent parts.                         很好理解，是时间上的包含关系

    Envelopes can be evaluated as functions of time or frequency using the
    fourier flag.  By default, they are evaluated as a function of time.        认为是时域或者频域上的分立的包，可以组合到一起
    """
    @convertUnits(start='ns', end='ns')
    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end

    def timeFunc(self, t):
        raise NotImplementedError('Envelope timeFunc must be overridden')

    def freqFunc(self, f):
        raise NotImplementedError('Envelope freqFunc must be overridden')

    def duration(self):
        return self.end - self.start

    def __call__(self, x, fourier=False):
        if fourier:
            return self.freqFunc(x)
        else:
            return self.timeFunc(x)

    def __add__(self, other):
        if isinstance(other, Envelope):
            return EnvSum(self, other)
        else:
            # if we try to add envelopes with the built in sum() function,
            # the first envelope is added to 0 before adding the rest.  To support
            # this, we add a special case here since adding 0 in time or fourier
            # is equivalent
            if other == 0:
                return self
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Envelope):
            return EnvSum(self, -other)
        else:
            # if we try to add envelopes with the built in sum() function,
            # the first envelope is added to 0 before adding the rest.  To support
            # this, we add a special case here since adding 0 in time or fourier
            # is equivalent
            if other == 0:
                return self
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Envelope):
            return EnvSum(-self, other)
        else:
            # if we try to add envelopes with the built in sum() function,
            # the first envelope is added to 0 before adding the rest.  To support
            # this, we add a special case here since adding 0 in time or fourier
            # is equivalent
            if other == 0:
                return -self
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Envelope):
            return NotImplemented
        elif isinstance(self, EnvProd):
            self.const *= other
            return self
        elif isinstance(self, EnvSum):
            for item in self.items:
                item *= other
            return self
        else:
            return EnvProd(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Envelope):
            return NotImplemented
        else:
            return EnvProd(self, 1.0 / other)

    def __rtruediv__(self, other):
        if isinstance(other, Envelope):
            return NotImplemented
        else:
            return EnvProd(EnvInv(self), other)

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self


class EnvSum(Envelope):                    # envelope之间做加减     与envolope格式的区别？怎么理解
    '''
    Helper class to support __add__ and __sub__
    '''
    def __init__(self, *envelopes):        # 多envolope的情况检查类型

        self.items = []
        for env in envelopes:
            if isinstance(env, EnvSum):
                self.items += env.items        # 怎么理解"+="
            elif isinstance(env, Envelope):
                self.items.append(env)
            else:
                raise Exception(f'invalid envelope type {type(env)}')

        start, end = timeRange(envelopes)
        Envelope.__init__(self, start, end)

    def timeFunc(self, t):
        return sum([item.timeFunc(t) for item in self.items])    # ——递归降低加和的深度
        # self.a.timeFunc(t) + self.b.timeFunc(t)

    def freqFunc(self, f):
        # print(self, self.a, self.b)
        return sum([item.freqFunc(f) for item in self.items])
        # self.a.freqFunc(f) + self.b.freqFunc(f)


# class EnvSum(Envelope):
#     '''
#     Helper class to support __add__ and __sub__
#     '''
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#         start, end = timeRange((self.a, self.b))
#         Envelope.__init__(self, start, end)

#     def timeFunc(self, t):
#         return self.a.timeFunc(t) + self.b.timeFunc(t)

#     def freqFunc(self, f):
#         # print(self, self.a, self.b)
#         return self.a.freqFunc(f) + self.b.freqFunc(f)


class EnvProd(Envelope):
    '''
    Helper class to support __mul__, __div__, and __neg__.  Represents multiplication by a scalar
    '''
    def __init__(self, envIn, const):
        self.envIn = envIn
        self.const = const
        Envelope.__init__(self, envIn.start, envIn.end)

    def timeFunc(self, t):
        return self.envIn.timeFunc(t) * self.const

    def freqFunc(self, f):
        return self.envIn.freqFunc(f) * self.const


class EnvInv(Envelope):
    '''
    Helper class to support division of a scalar by an
    envelope. (__rdiv__).  I don't know why this is helpful -- ERJ
    '''
    def __init__(self, envIn):
        self.envIn = envIn
        Envelope.__init__(self, envIn.start, envIn.end)

    def timeFunc(self, t):
        return 1.0 / self.envIn.timeFunc(t)

    def freqFunc(self, f):
        return 1.0 / self.envIn.freqFunc(f)


class EnvConvFOnly(Envelope):
    '''
    Helper class that convolves an envelope with a smoothing function
    **in the frequency domain only**.  Convolution is expensive in the
    time domain, and hard to get right.  so we just pass that data
    through directly -- it is only used for plotting anyway.
    '''
    def __init__(self, env_in, env_filter):
        self.env_in = env_in
        self.env_filter = env_filter

    def timeFunc(self, t):
        return self.env_in.timeFunc(t)

    def freqFunc(self, f):
        return self.env_in.freqFunc(f) * self.env_filter.freqFunc(f)


class EnvZero(Envelope):
    '''
    Zero envelope.  Duh.
    '''
    def timeFunc(self, t):
        return 0j * t

    def freqFunc(self, f):
        return 0j * f


# empty envelope
NOTHING = EnvZero(start=None, end=None)
ZERO = EnvZero(start=0, end=0)

# ------- modified Envelopes --------


class mix(Envelope):
    """Apply sideband mixing at difference frequency df."""
    @convertUnits(df='GHz')
    def __init__(self, env, df=0.0, phase=0.0):
        if df is None:
            raise Exception
        self.df = df
        self.phase = phase
        self.env = env
        Envelope.__init__(self, env.start, env.end)

    def timeFunc(self, t):
        return self.env(t) * np.exp(-2j * np.pi * self.df * t -
                                    1.0j * self.phase)

    def freqFunc(self, f):
        return self.env(f + self.df, fourier=True) * np.exp(-1.0j * self.phase)


class deriv(Envelope):
    @convertUnits(dt='ns')
    def __init__(self, env, dt=0.1):
        """Get the time derivative of a given envelope."""
        self.env = env
        self.dt = dt
        Envelope.__init__(self, env.start, env.end)

    def timeFunc(self, t):
        return (self.env(t + self.dt) - self.env(t - self.dt)) / (2 * self.dt)

    def freqFunc(self, f):
        return 2j * np.pi * f * self.env(f, fourier=True)


class dragify(Envelope):
    @convertUnits(dt='ns')
    def __init__(self, env, alpha, dt):
        self.env = env
        self.alpha = alpha
        self.dt = dt
        Envelope.__init__(self, env.start, env.end)

    def freqFunc(self, f):
        return (1 + 2j * np.pi * f * self.alpha) * self.env(f, fourier=True)

    def timeFunc(self, t):
        return (self.env(t) + self.alpha *
                (self.env(t + self.dt) - self.env(t - self.dt)) /
                (2 * self.dt))


class shift(Envelope):
    @convertUnits(dt='ns')
    def __init__(self, env, dt):
        self.dt = dt
        self.env = env
        Envelope.__init__(self,
                          env.start + dt if env.start is not None else None,
                          env.end + dt if env.end is not None else None)

    def timeFunc(self, t):
        return self.env(t - self.dt)

    def freqFunc(self, f):
        return self.env(f, fourier=True) * exp(-2j * np.pi * f * self.dt)


# ------- instance of specific Envelopes -----
# @envelope_factory_t0
# @memoize_envelope
class gaussian(Envelope):
    @convertUnits(t0='ns', w='ns', amp=None, phase=None, df='GHz')
    def __init__(self, t0, w, amp=1.0, phase=0.0, df=0.0):
        """A gaussian pulse with specified center and full-width at half max."""
        # convert fwhm to std. deviation
        self.sigma = w / np.sqrt(8 * np.log(2))
        self.t0 = t0
        self.amp = amp
        self.phase = phase
        self.df = df
        # Envelope.__init__(self, start=t0 - 2 * w, end=t0 + 2 * w) # UCSB-pyle2014
        Envelope.__init__(self, start=t0 - w, end=t0 + w)

    def timeFunc(self, t):
        return self.amp * np.exp(-(t - self.t0)**2 /
                                 (2 * self.sigma**2) - 2j * np.pi * self.df *
                                 (t - self.t0) + 1j * self.phase)

    def freqFunc(self, f):
        sigmaf = 1 / (2 * np.pi * self.sigma)  # width in frequency space
        ampf = self.amp * np.sqrt(
            2 * np.pi * self.sigma**2)  # amp in frequency space
        return ampf * np.exp(-(f + self.df)**2 / (2 * sigmaf**2) -
                             2j * np.pi * f * self.t0 + 1j * self.phase)


# @envelope_factory_t0
# @memoize_envelope
class cosine(Envelope):
    @convertUnits(t0='ns', w='ns', amp=None, phase=None)
    def __init__(self, t0, w, amp=1.0, phase=0.0):
        """A cosine function centered at t0 with FULL WIDTH w"""
        self.t0 = t0
        self.w = w
        self.amp = amp
        self.phase = phase
        Envelope.__init__(self, t0 - w / 2.0, t0 + w / 2.0)

    def timeFunc(self, t):
        return self.amp * 0.5 * (1 + np.cos(
            2 * np.pi *
            (t - self.t0) / self.w)) * (((t - self.t0) + self.w / 2.) > 0) * (
                (-(t - self.t0) + self.w / 2.) > 0) * np.exp(1j * self.phase)

    def freqFunc(self, f):
        wf = self.w * f
        a = 1. - wf
        b = 1. + wf
        return self.amp * np.exp(-2j * np.pi * f * self.t0 + 1j * self.phase
                                 ) * self.w / 4. * (2. * np.sinc(wf) +
                                                    np.sinc(a) + np.sinc(b))


# @envelope_factory_t0
# @memoize_envelope
class triangle(Envelope):
    @convertUnits(t0='ns', tlen='ns', amp=None)
    def __init__(self, t0, tlen, amp, fall=True):
        """A triangular pulse, either rising or falling."""
        self.t0 = t0
        self.tlen = tlen
        self.amp = amp
        if not fall:
            self.t0 = t0 + tlen
            self.tlen = -tlen
            self.amp = amp

        tmin = min(t0, t0 + tlen)
        tmax = max(t0, t0 + tlen)
        Envelope.__init__(self, tmin, tmax)

    def timeFunc(self, t):
        return self.amp * (t >= self.start) * (t < self.end) * (
            1 - (t - self.t0) / self.tlen)

    def freqFunc(self, f):
        if self.tlen == 0 or self.amp == 0:
            return 0.0 * f
        # this is tricky because the fourier transform has a 1/f term, which blows up for f=0
        # the z array allows us to separate the zero-frequency part from the rest
        z = f == 0
        f = 2j * np.pi * (f + z)
        return self.amp * ((1 - z) * np.exp(-f * self.t0) *
                           (1.0 / f - (1 - np.exp(-f * self.tlen)) /
                            (f**2 * self.tlen)) + z * self.tlen / 2.0)


# @envelope_factory_t0
# @memoize_envelope
class rect_origin(Envelope):
    @convertUnits(t0='ns', tlen='ns', amp=None, overshoot=None)
    def __init__(self, t0, tlen, amp, overshoot=0.0, overshoot_w=1.0):
        """A rectangular pulse with sharp turn on and turn off.

        Note that the overshoot_w parameter, which defines the FWHM of the gaussian overshoot peaks
        is only used when evaluating the envelope in the time domain.  In the fourier domain, as is
        used in the dataking code which uploads sequences to the boards, the overshoots are delta
        functions.
        """
        self.t0 = t0
        self.amp = amp
        self.overshoot = overshoot * np.sign(
            amp)  # overshoot will be zero if amp is zero
        tmin = min(t0, t0 + tlen)
        tmax = max(t0, t0 + tlen)
        self.tmid = (tmin + tmax) / 2.0
        self.tlen = tlen

        # to add overshoots in time, we create an envelope with two gaussians
        if overshoot:
            o_w = overshoot_w
            o_amp = 2 * np.sqrt(np.log(2) / np.pi) / o_w  # total area == 1
            self.o_env = gaussian(tmin, o_w, o_amp) + gaussian(
                tmax, o_w, o_amp)
        else:
            self.o_env = EnvZero(tmin, tmax)

        Envelope.__init__(self, tmin, tmax)

    def timeFunc(self, t):
        return (self.amp * (t >= self.start) * (t < self.end) +
                self.overshoot * self.o_env(t))

    # to add overshoots in frequency, use delta funcs (smoothed by filters)
    def freqFunc(self, f):
        return (self.amp * abs(self.tlen) * np.sinc(self.tlen * f) *
                np.exp(-2j * np.pi * f * self.tmid) + self.overshoot *
                (np.exp(-2j * np.pi * f * self.start) +
                 np.exp(-2j * np.pi * f * self.end)))


class rect(Envelope):
    @convertUnits(t0='ns', tlen='ns', amp=None, overshoot=None)
    def __init__(self, t0, tlen, amp, overshoot=0.0, overshoot_w=1.0):
        """A rectangular pulse with sharp turn on and turn off.

        Note that the overshoot_w parameter, which defines the FWHM of the gaussian overshoot peaks
        is only used when evaluating the envelope in the time domain.  In the fourier domain, as is
        used in the dataking code which uploads sequences to the boards, the overshoots are delta
        functions.
        """
        self.t0 = t0
        self.amp = amp
        self.overshoot = overshoot * np.sign(
            amp)  # overshoot will be zero if amp is zero
        tmin = min(t0, t0 + tlen)
        tmax = max(t0, t0 + tlen)
        self.tmid = (tmin + tmax) / 2.0
        self.tlen = tlen

        # to add overshoots in time, we create an envelope with two gaussians
        if overshoot:
            o_w = overshoot_w
            o_amp = 2 * np.sqrt(np.log(2) / np.pi) / o_w  # total area == 1
            self.o_env = gaussian(tmin, o_w, o_amp) + gaussian(
                tmax, o_w, o_amp)
        else:
            self.o_env = EnvZero(tmin, tmax)

        Envelope.__init__(self, tmin, tmax)

    def timeFunc(self, t):
        return (self.amp * (t >= self.start) * (t < self.end) +
                self.overshoot * self.o_env(t))

    # to add overshoots in frequency, use delta funcs (smoothed by filters)
    def freqFunc(self, f):
        if self.overshoot:
            return (self.amp * abs(self.tlen) * np.sinc(self.tlen * f) *
                    np.exp(-2j * np.pi * f * self.tmid) + self.overshoot *
                    (np.exp(-2j * np.pi * f * self.start) +
                     np.exp(-2j * np.pi * f * self.end)))
            # return (self.amp * abs(self.tlen) * np.sinc(self.tlen * f) *
            #         np.exp(-2j * np.pi * f * self.tmid) + self.overshoot *
            #         (np.exp(-2j * np.pi * f * (self.start + self.end))))  # 这种 self.start 和 self.end 合并的写法相较分开写结果会不一样
        else:
            return self.amp * abs(self.tlen) * np.sinc(self.tlen * f) * np.exp(
                -2j * np.pi * f * self.tmid)


# @envelope_factory_t0
# @memoize_envelope
class flattop_new(Envelope):
    # @convertUnits(t0='ns', tlen='ns', w='ns', amp=None)
    # def __new__(cls, t0, tlen, w, amp=1.0):
    #     '''
    #     __new__ optimizes the case where amp=0 by constructing an EnvZero instance
    #     instead of a flattop.  This seems to happen unnecessarily often, so this saves memory,
    #     and maybe a bit of performance
    #     '''
    #     if amp == 0:
    #         return EnvZero(t0, t0 + tlen)
    #     else:
    #         # __init__ will be called!
    #         return Envelope.__new__(cls)

    @convertUnits(t0='ns', tlen='ns', w='ns', amp=None)
    def __init__(self, t0, tlen, w, amp=1.0):
        self.t0 = t0
        self.tlen = tlen
        self.amp = amp
        self.w = w
        Envelope.__init__(self, t0, t0 + tlen)

    def timeFunc(self, t):
        t0 = self.t0
        length = self.tlen
        w = self.w
        amp = self.amp
        if w > 0:
            return amp / 2. * (
                1 + np.sin(np.pi * (t - t0) / w - np.pi / 2.)) * (t > t0) * (
                    t < (t0 + w)) + amp / 2. * (
                        1 + np.sin(np.pi * (t - t0 - length) / w - np.pi / 2.)
                    ) * (t > (t0 + length - w)) * (t < (t0 + length)) + amp * (
                        t >= (t0 + w)) * (t <= (t0 + length - w))
        else:
            return rect(t0, length, amp=amp)(t, fourier=False)

    def freqFunc(self, f):
        rect_env = rect(self.t0 + self.w / 2., self.tlen - self.w, 1.0)
        if self.w > 0:
            kernel = cosine(0, self.w, amp=2 / self.w)  # area=1
            return self.amp * rect_env(f, fourier=True) * kernel(
                f, fourier=True)  # convolve with cosine kernel
        else:
            return self.amp * rect_env(f, fourier=True)


class flattop_origin(Envelope):
    # @convertUnits(t0='ns', tlen='ns', w='ns', amp=None)
    # def __new__(cls, t0, tlen, w, amp=1.0):
    #     '''
    #     __new__ optimizes the case where amp=0 by constructing an EnvZero instance
    #     instead of a flattop.  This seems to happen unnecessarily often, so this saves memory,
    #     and maybe a bit of performance
    #     '''
    #     if amp == 0:
    #         return EnvZero(t0, t0 + tlen)
    #     else:
    #         # __init__ will be called!
    #         return Envelope.__new__(cls)

    @convertUnits(t0='ns', tlen='ns', w='ns', amp=None)
    def __init__(self, t0, tlen, w, amp=1.0, overshoot=0.0, overshoot_w=1.0):
        self.t0 = t0
        self.tlen = tlen
        self.amp = amp
        self.w = w
        self.overshoot = overshoot * np.sign(
            amp)  # overshoot will be zero if amp is zero
        self.overshoot_w = overshoot_w
        Envelope.__init__(self, t0, t0 + tlen)

    def timeFunc(self, t):
        t0 = self.t0
        length = self.tlen
        w = self.w
        amp = self.amp
        overshoot = self.overshoot
        overshoot_w = self.overshoot_w

        tmin = min(t0, t0 + length)
        tmax = max(t0, t0 + length)

        # to add overshoots in time, we create an envelope with two gaussians
        a = 2 * np.sqrt(np.log(2)) / w
        if overshoot:
            o_w = overshoot_w
            o_amp = 2 * np.sqrt(np.log(2) / np.pi) / o_w  # total area == 1
            o_env = gaussian(tmin, o_w, o_amp) + gaussian(tmax, o_w, o_amp)
        else:
            o_env = NOTHING

        return (amp * (erf(a * (tmax - t)) - erf(a * (tmin - t))) / 2.0 +
                overshoot * o_env(t))

    def freqFunc(self, f):
        t0 = self.t0
        length = self.tlen
        w = self.w
        amp = self.amp
        overshoot = self.overshoot

        tmin = min(t0, t0 + length)
        tmax = max(t0, t0 + length)

        # to add overshoots in frequency, use delta funcs (smoothed by filters)
        rect_env = rect(t0, length, 1.0)
        kernel = gaussian(0, w, 2 * np.sqrt(np.log(2) / np.pi) / w)  # area = 1
        # rect_env = rect(self.t0 + self.w / 2., self.tlen - self.w, 1.0)

        return (
            amp * rect_env(f, fourier=True) * kernel(f, fourier=True)
            +  # convolve with gaussian kernel
            overshoot *
            (np.exp(-2j * np.pi * f * tmin) + np.exp(-2j * np.pi * f * tmax)))


class flattop(Envelope):
    # @convertUnits(t0='ns', tlen='ns', w='ns', amp=None)
    # def __new__(cls, t0, tlen, w, amp=1.0):
    #     '''
    #     __new__ optimizes the case where amp=0 by constructing an EnvZero instance
    #     instead of a flattop.  This seems to happen unnecessarily often, so this saves memory,
    #     and maybe a bit of performance
    #     '''
    #     if amp == 0:
    #         return EnvZero(t0, t0 + tlen)
    #     else:
    #         # __init__ will be called!
    #         return Envelope.__new__(cls)

    @convertUnits(t0='ns', tlen='ns', w='ns', amp=None)
    def __init__(self, t0, tlen, w, amp=1.0, overshoot=0.0, overshoot_w=1.0):
        self.t0 = t0
        self.tlen = tlen
        self.amp = amp
        self.w = w
        self.overshoot = overshoot * np.sign(
            amp)  # overshoot will be zero if amp is zero
        self.overshoot_w = overshoot_w
        Envelope.__init__(self, t0, t0 + tlen)

    def timeFunc(self, t):
        t0 = self.t0
        length = self.tlen
        w = self.w
        amp = self.amp
        overshoot = self.overshoot
        overshoot_w = self.overshoot_w

        tmin = min(t0, t0 + length)
        tmax = max(t0, t0 + length)

        # to add overshoots in time, we create an envelope with two gaussians
        a = 2 * np.sqrt(np.log(2)) / w
        if overshoot:
            o_w = overshoot_w
            o_amp = 2 * np.sqrt(np.log(2) / np.pi) / o_w  # total area == 1
            o_env = gaussian(tmin, o_w, o_amp) + gaussian(tmax, o_w, o_amp)
        else:
            o_env = NOTHING

        return (amp * (erf(a * (tmax - t)) - erf(a * (tmin - t))) / 2.0 +
                overshoot * o_env(t))

    def freqFunc(self, f):
        t0 = self.t0
        length = self.tlen
        w = self.w
        amp = self.amp
        overshoot = self.overshoot

        tmin = min(t0, t0 + length)
        tmax = max(t0, t0 + length)

        # to add overshoots in frequency, use delta funcs (smoothed by filters)
        rect_env = rect(t0, length, 1.0)
        kernel = gaussian(0, w, 2 * np.sqrt(np.log(2) / np.pi) / w)  # area = 1
        # rect_env = rect(self.t0 + self.w / 2., self.tlen - self.w, 1.0)

        if overshoot:
            return (amp * rect_env(f, fourier=True) * kernel(f, fourier=True)
                    +  # convolve with gaussian kernel
                    overshoot * (np.exp(-2j * np.pi * f * tmin) +
                                 np.exp(-2j * np.pi * f * tmax)))
        else:
            return amp * rect_env(f, fourier=True) * kernel(
                f, fourier=True)  # convolve with gaussian kernel


# @envelope_factory_t0
# @memoize_envelope
class ripple_rect(Envelope):
    @convertUnits(t0='ns', tlen='ns', amp=None, ripples=None)
    def __init__(self, t0, tlen, amp, ripples):
        self.t0 = t0
        self.tlen = tlen
        ripples = amp * np.array(ripples)
        # for future coarse tuning DCZ gate in 1120 swap and phase procidure
        self.amp = amp - ripples[1] - ripples[3]
        self.ripples = ripples

        tmin = min(t0, t0 + tlen)
        tmax = max(t0, t0 + tlen)
        tmid = (tmin + tmax) / 2.0
        self.tmin = tmin
        self.tmax = tmax
        self.tmid = tmid

        Envelope.__init__(self, t0, t0 + tlen)

    def timeFunc(self, t):
        amps = self.amp
        tmin = self.tmin
        tmax = self.tmax
        tmid = self.tmid
        tlen = self.tlen
        ripples = self.ripples
        amp_r = 0
        if self.tlen > 0:
            for idx, r in enumerate(ripples):
                idx_r = 2**(idx // 2)
                if np.mod(idx, 2) == 0:
                    amp_r += r * np.sin(idx_r * np.pi * (t - tmid) / tlen)
                if np.mod(idx, 2) == 1:
                    amp_r += r * np.cos(idx_r * np.pi * (t - tmid) / tlen)
        amps = (amps + amp_r) * (t >= tmin) * (t < tmax)
        return amps

    def freqFunc(self, f):
        tmid = self.tmid
        amp = self.amp
        ripples = self.ripples
        rect_env = rect(self.t0, self.tlen, 1.0)
        tlen = self.tlen
        ampf = 0
        for idx, r in enumerate(ripples):
            idx_r = 2**(idx // 2)
            if np.mod(idx, 2) == 0:
                ampf += r * tlen * (np.sinc(tlen * f - idx_r / 2) -
                                    np.sinc(tlen * f + idx_r / 2)) / 2j
            if np.mod(idx, 2) == 1:
                ampf += r * tlen * (np.sinc(tlen * f - idx_r / 2) +
                                    np.sinc(tlen * f + idx_r / 2)) / 2
        return (amp * rect_env(f, fourier=True) +
                ampf * np.exp(-2j * np.pi * f * tmid))


class diabaticCZ(Envelope):
    @convertUnits(t0='ns', tlen='ns', amp=None, w='ns', ripples=None)
    def __init__(self, t0, tlen, amp, w, ripples):
        self.t0 = t0
        self.tlen = tlen
        self.w = w
        ripples = amp * np.array(ripples)
        # for future coarse tuning DCZ gate in 1120 swap and phase procidure
        self.amp = amp - ripples[1] - ripples[3]
        self.ripples = ripples

        tmin = min(t0, t0 + tlen)
        tmax = max(t0, t0 + tlen)
        tmid = (tmin + tmax) / 2.0
        kernel = gaussian(0.0, w, 2 * np.sqrt(np.log(2) / np.pi) / w)
        tstep = 0.05
        window = kernel(np.arange(-3 * w, 3 * w + tstep, tstep))
        self.window = window / window.sum()
        self.tmin = tmin
        self.tmax = tmax
        self.tmid = tmid
        self.tstep = tstep
        self.kernel = kernel

        Envelope.__init__(self, t0, t0 + tlen)

    def timeFuncRect(self, t):
        amps = self.amp
        w = self.w
        tmin = self.tmin
        tmax = self.tmax
        tmid = self.tmid
        tlen = self.tlen
        ripples = self.ripples
        a = 2 * np.sqrt(np.log(2)) / w
        amp_r = 0
        if self.tlen > 0:
            for idx, r in enumerate(ripples):
                idx_r = 2**(idx // 2)
                if np.mod(idx, 2) == 0:
                    amp_r += r * np.sin(idx_r * np.pi * (t - tmid) / tlen)
                if np.mod(idx, 2) == 1:
                    amp_r += r * np.cos(idx_r * np.pi * (t - tmid) / tlen)
        amps = (amps + amp_r) * (t >= tmin) * (t < tmax)
        return amps

    def timeFunc(self, t):
        tstep = self.tstep
        window = self.window
        w = self.w
        if w > 0:
            if np.shape(t):
                return np.array([self.timeFunc(ti) for ti in t])
            else:
                tlist = np.arange(t - 3 * w, t + 3 * w + tstep, tstep)
                signal = self.timeFuncRect(tlist)
                data = np.convolve(window, signal, mode='valid')
            return data[0]
        else:
            return self.timeFuncRect(t)

    def freqFunc(self, f):
        tmid = self.tmid
        amp = self.amp
        ripples = self.ripples
        rect_env = rect(self.t0, self.tlen, 1.0)
        tlen = self.tlen
        ampf = 0
        for idx, r in enumerate(ripples):
            idx_r = 2**(idx // 2)
            if np.mod(idx, 2) == 0:
                ampf += r * tlen * (np.sinc(tlen * f - idx_r / 2) -
                                    np.sinc(tlen * f + idx_r / 2)) / 2j
            if np.mod(idx, 2) == 1:
                ampf += r * tlen * (np.sinc(tlen * f - idx_r / 2) +
                                    np.sinc(tlen * f + idx_r / 2)) / 2
        return (amp * rect_env(f, fourier=True) + ampf *
                np.exp(-2j * np.pi * f * tmid)) * self.kernel(f, fourier=True)


class cache(Envelope):
    @convertUnits(start='ns', end='ns')
    def __init__(self, time_func, freq_func, start, end):
        self.time_func = time_func
        self.freq_func = freq_func
        Envelope.__init__(self, start, end)

    def timeFunc(self, t):
        return self.time_func(t)

    def freqFunc(self, f):
        return self.freq_func(f)


class class_NOTHING_0_5ns(Envelope):
    def __init__(self, ):
        Envelope.__init__(self, 0.0 * ns, 5.0 * ns)

    def timeFunc(self, t):
        return 0 * t

    def freqFunc(self, f):
        return 0 * f


NOTHING_0_5ns = class_NOTHING_0_5ns()


# utility functions
def timeRange(envelopes):
    """Calculate the earliest start and latest end of a list of envelopes.        # 总时间戳，没啥好说的
    
    Returns a tuple (start, end) giving the time range.  Note that one or
    both of start and end may be None if the envelopes do not specify limits.
    """
    starts = [env.start for env in envelopes if env.start is not None]
    start = min(starts) if len(starts) else None
    ends = [env.end for env in envelopes if env.end is not None]
    end = max(ends) if len(ends) else None
    return start, end


@convertUnits(time='ns', dt='ns')
def fftFreqs(time=1024, dt=1):
    """Get a list of frequencies for evaluating fourier envelopes.

    The time is rounded up to the nearest power of two, since powers
    of two are best for the fast fourier transform.  Returns a tuple
    of frequencies to be used for complex and for real signals.
    """
    nfft = 2**int(math.ceil(math.log(time / dt, 2)))
    f_complex = np.fft.fftfreq(nfft, d=dt)
    f_real = f_complex[:int(nfft // 2) + 1]
    # TODO: 检查f_real该如何定义
    return f_complex, f_real


# def fftFreqs(time=1024, dt=1.0):
#     """Get a list of frequencies for evaluating fourier envelopes.

#     The time is rounded up to the nearest power of two, since powers
#     of two are best for the fast fourier transform.  Returns a tuple
#     of frequencies to be used for complex and for real signals.
#     """
#     # print(f"=================================time is:{time}")
#     # time = time*ns
#     try:
#         nfft = 2**int(math.ceil(math.log(time, 2)))
#     except:
#         nfft = 2**int(math.ceil(math.log(time['ns'], 2)))
#     f_complex = np.fft.fftfreq(nfft,d=dt)
#     f_real = f_complex[:int(nfft/2)+1]
#     f_real = np.copy(f_real)
#     f_real[-1] = f_real[-2]+f_real[-2]-f_real[-3]
#     return f_complex, f_real


def ifft(envelope, t0=-200, n=1000):
    f = np.fft.fftfreq(n)
    return np.fft.ifft(envelope(f, fourier=True) * np.exp(2j * np.pi * t0 * f))


def fft(envelope, t0=-200, n=1000):
    t = t0 + np.arange(n)
    return np.fft.fft(envelope(t))


def plotFT(envelope, t0=-200, n=1000):
    t = t0 + np.arange(n)
    y = ifft(envelope, t0, n)
    plt.plot(t, np.real(y))
    plt.plot(t, np.imag(y))


def plotTD(envelope, t0=-200, n=1000):
    t = t0 + np.arange(n)
    y = envelope(t)
    plt.plot(t, np.real(y))
    plt.plot(t, np.imag(y))


def plot_env(env, padding=100.0, N=1024):
    tstart = env.start - padding
    tend = env.end + padding
    t = np.linspace(tstart, tend, N, endpoint=False)
    x = env(t)
    f = np.fft.fftfreq(N, (tend - tstart) * 1.0 / N)
    x2 = np.fft.ifft(env(f, fourier=True) *
                     np.exp(2j * np.pi * f * tstart)) * N / (tend - tstart)
    plt.figure()
    plt.plot(t, x, label='Time domain data')
    plt.plot(t, x2.real, label='Re(IFFT) of Frequeny envelope')
    plt.plot(t, x2.imag, label='Im(IFFT) of Frequeny envelope')
    plt.legend()
    plt.figure()
    plt.plot(f, np.abs(env(f, fourier=True)))
    plt.plot(f, np.abs(np.fft.fft(env(t)) / (2 * np.sqrt(2 * np.pi))))


if __name__ == '__main__':
    # t_list = np.arange(60)
    # # g = gaussian(30, w=20, amp=0.5)
    # g = triangle(30, tlen=20, amp=0.5)
    # res = g.timeFunc(t_list)
    # plt.axis('off')
    # plt.plot(t_list, res.real, linewidth=7, color='black')
    # plt.savefig('triangle.svg', transparent=True)
    # plt.show()

    plot_env(cosine(0, 2))
    plt.show()