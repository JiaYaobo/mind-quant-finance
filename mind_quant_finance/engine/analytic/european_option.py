import mindspore.numpy as np
from mindspore import dtype as mstype
import mindspore.nn.probability.distribution as msd
from mind_quant_finance.utils.rate_factor import get_discount_rate_factor


def black_scholes_merton(expiries, strikes, spots, volatilities,
                         discount_factors=None,
                         discount_rates=None,
                         dividend_rates=None,
                         is_call_options=None,
                         is_normal_volatility: bool = False,
                         dtype=mstype.float32
                         ):
    """Args:
        expiries: The expiry of the options. ms.Tensor
        strikes: The strikes of the options. the ms.Tensor of any shape
        spots: The spots price of the underlying object. ms.Tensor
        volatilities: the volatilities of the options. ms.Tensor
        discount_factors: A ms.Tensor with the same shape of der_price/ spot.
                         the discount factor of the der_price/spot. (e^{-rT})
        discount_rates: A ms.Tensor with the same shape of der_price/spot.
                       The discount rate of the der_price/spote. (r)
                       discount_factor = exp(-expiries*discount_rate).
                       at most one of the factor/rate can be supplied.
                       if both are None, it means discount_rate = 1.0 (Default/No discount)
        dividend_rates: A ms.Tensor with the same shape of der_price/ obj_prices
                        The dividend_rates q of the object, Default q=0
        is_call_options: A boolean ms.Tensor of a shape compatible with `prices`.
                         Indicates whether the option is a call (True) or a put (False).
                         If not supplied, call options are assumed.
        is_normal_volatility: An optional Python boolean specifying whether the volatilities
                              correspond to lognormal Black volatility (if False) or normal
                              Black volatility (if True).
        dtype: the data type of the solver, default: ms.dtype.float32
    """
    """Return:
        The discounted price of derivatives calculated by the Black-Scholes-Merton Model 
    """
    discounted_flag = (dividend_rates is not None) | \
                      (discount_rates is not None) | \
                      (discount_factors is not None)
    discount_factors, discount_rates = \
        get_discount_rate_factor(discount_factors, discount_rates, spots.shape,
                                 expiries, dtype)
    if dividend_rates is None:
        dividend_rates = np.zeros_like(spots, dtype=dtype)
    if discounted_flag:
        # make obj price the un-discounted price
        obj_prices = spots * np.exp((discount_rates - dividend_rates) * expiries)
    else:
        obj_prices = spots
    if is_call_options is None:
        is_call_options = np.ones_like(obj_prices, dtype=mstype.bool_)
    if not is_normal_volatility:
        return _bsm_lognormal_option_price_solve(expiries, strikes, obj_prices, volatilities,
                                                 discount_factors, is_call_options, dtype)
    else:
        return _bsm_normal_option_price_solve(expiries, strikes, obj_prices, volatilities,
                                              discount_factors, is_call_options, dtype)


# [ycb] The current MindSpore Graph Mode dose not support msd.Normal.cdf()
# Once this function is supported by MindSpore Community,
# this implied function _bsm_option_price_solve() can be declare as @ms_function
# @ms_function
def _bsm_lognormal_option_price_solve(expiries, strikes, obj_prices,
                                      volatilities, discount_factors, is_call_options, dtype):
    lnf = np.log(obj_prices) - np.log(strikes)
    sqrt_t = np.sqrt(expiries)
    n_g = msd.Normal(0.0, 1.0, dtype=dtype, name='StandNormal')
    vol_t = volatilities * sqrt_t
    d1 = (lnf / vol_t + vol_t / 2)
    d2 = d1 - vol_t
    der_call = np.where(vol_t > 0,
                        obj_prices * n_g.cdf(d1) - strikes * n_g.cdf(d2),
                        np.maximum(obj_prices - strikes, np.zeros(1, dtype=dtype))
                        )
    der_put = der_call - obj_prices + strikes
    der_price_undiscounted = np.where(is_call_options, der_call, der_put)
    return discount_factors * der_price_undiscounted


# [ycb] The current MindSpore Graph Mode dose not support msd.Normal.cdf()
# Once this function is supported by MindSpore Community,
# this implied function _bsm_option_price_solve() can be declare as @ms_function
# @ms_function
def _bsm_normal_option_price_solve(expiries, strikes, obj_prices,
                                   volatilities, discount_factors, is_call_options, dtype):
    sqrt_t = np.sqrt(expiries)
    n_g = msd.Normal(0.0, 1.0, dtype=dtype, name='StandNormal')
    vol_t = volatilities * sqrt_t
    d1 = np.divide((obj_prices - strikes), vol_t)
    der_call = np.where(vol_t > 0,
                        (obj_prices - strikes) * n_g.cdf(d1) + vol_t * n_g.prob(d1),
                        np.maximum(obj_prices - strikes, np.zeros(1, dtype=dtype))
                        )
    der_put = der_call - obj_prices + strikes
    der_price_undiscounted = np.where(is_call_options, der_call, der_put)
    return discount_factors * der_price_undiscounted
