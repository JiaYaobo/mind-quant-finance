import mindspore.numpy as np
from mindspore import dtype as mstype
from mindspore import nn, ms_function
import mindspore as ms
from mind_quant_finance.math.distribution import standnormal_pdf, standnormal_cdf


@ms_function
def _bsm_lognormal_option_price_solve(expiries, strikes, obj_prices, volatilities, is_call_options, dtype):
    lnf = np.log(obj_prices) - np.log(strikes)
    sqrt_t = np.sqrt(expiries)
    vol_t = volatilities * sqrt_t
    d1 = (lnf / vol_t + vol_t / 2)
    d2 = d1 - vol_t
    der_call = np.where(vol_t > 0,
                        obj_prices * standnormal_cdf(d1) - strikes * standnormal_cdf(d2),
                        np.maximum(obj_prices - strikes, np.zeros(1, dtype=dtype))
                        )
    der_put = der_call - obj_prices + strikes
    der_price_undiscounted = np.where(is_call_options, der_call, der_put)
    return der_price_undiscounted


@ms_function
def _bsm_normal_option_price_solve(expiries, strikes, obj_prices, volatilities, is_call_options, dtype):
    sqrt_t = np.sqrt(expiries)
    vol_t = volatilities * sqrt_t
    d1 = np.divide((obj_prices - strikes), vol_t)
    der_call = np.where(vol_t > 0,
                        (obj_prices - strikes) * standnormal_cdf(d1) + vol_t * standnormal_pdf(d1),
                        np.maximum(obj_prices - strikes, np.zeros(1, dtype=dtype))
                        )
    der_put = der_call - obj_prices + strikes
    der_price_undiscounted = np.where(is_call_options, der_call, der_put)
    return der_price_undiscounted


class AnalyticBlackScholesMerton(nn.Cell):
    def __init__(self, is_normal_volatility: bool = False, dtype=mstype.float32):
        """Args:
            is_normal_volatility: An optional Python boolean specifying whether the volatilities
                                  correspond to lognormal Black volatility (if False) or normal
                                  Black volatility (if True).
            dtype: the data type of the solver, default: ms.dtype.float32
        """
        super(AnalyticBlackScholesMerton, self).__init__()
        if is_normal_volatility:
            self.calc = _bsm_normal_option_price_solve
        else:
            self.calc = _bsm_lognormal_option_price_solve
        self.dtype = dtype

    def construct(self, expiries, strikes, spots, volatilities,
                  discount_rates=0, dividend_rates=0, is_call_options=True):
        """Args:
            expiries: The expiry of the options. ms.Tensor
            strikes: The strikes of the options. the ms.Tensor of any shape
            spots: The spots price of the underlying object. ms.Tensor
            volatilities: the volatilities of the options. ms.Tensor
            discount_rates: ms.Tensor or float
                            The discount rate of the der_price/spote. (r)
                            discount_factor = exp(-expiries*discount_rate).
                            if None, it means discount_rate = 1.0 (Default/No discount)
            dividend_rates: ms.Tensor or float
                            The dividend_rates q of the object, Default q=0
            is_call_options: A boolean ms.Tensor of a shape compatible with `prices`.
                             Indicates whether the option is a call (True) or a put (False).
                             If not supplied, call options are assumed.
        """
        obj_prices = spots * np.exp((discount_rates - dividend_rates) * expiries)
        forwards = self.calc(expiries, strikes, obj_prices, volatilities, is_call_options, self.dtype)
        discount_factors = np.exp(-discount_rates * expiries)
        return forwards * discount_factors
