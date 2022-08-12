import mindspore as ms
import mindspore.numpy as np
from mindspore import dtype as mstype
from typing import Optional
from mind_quant_finance.math.distribution import standnormal_pdf, standnormal_cdf
from mindspore import ms_function
import enum
from mind_quant_finance.math.root_search import newton
from mind_quant_finance.utils.rate_factor import get_discount_rate_factor


@enum.unique
class ImpliedVolUnderlyingDistribution(enum.Enum):
    """Underlying distribution.
  * `BSM`: Lognormal distribution for the standrad Black-Scholes models.
  * `Bachelier`: Normal distribution used in Bachelier model.
  """
    BSM = 0
    Bachelier = 1


def implied_vol_solver(expiries, strikes, der_prices, spots,
                       discount_factors=None,
                       discount_rates=None,
                       is_call_options=None,
                       underlying_distribution=ImpliedVolUnderlyingDistribution.BSM,
                       initial_volatilities=None,
                       tolerance: Optional[float] = 1e-6,
                       max_iterations: Optional[int] = 20,
                       dtype=mstype.float32
                       ):
    """Args:
        expiries: The expiry of the options.
        strikes: The strikes of the options.
        der_prices: The current price of the options.
        spots: The spot of the underlying product.
        discount_factors: A ms.Tensor with the same shape of der_price/ spot.
                         the discount factor of the der_price/spot. (e^{-rT})
        discount_rates: A ms.Tensor with the same shape of der_price/spot.
                       The discount rate of the der_price/spote. (r)
                       discount_factor = exp(-expiries*discount_rate).
                       at most one of the factor/rate can be supplied.
                       if both are None, it means discount_rate = 1.0 (Default/No discount)
        is_call_options: A boolean ms.Tensor of a shape compatible with `prices`.
                         Indicates whether the option is a call (True) or a put (False).
                         If not supplied, call options are assumed.
        initial_volatilities: the initial point of volatilities for Implied-Vol Solver
        underlying_distribution:  Enum value of ImpliedVolUnderlyingDistribution.
                                  Select the distribution of the underlying.
                                  (BSM / Bachelier Model)
        tolerance: The root finder will stop where this tolerance is crossed.
        max_iterations: The maximum number of iterations of Implied-Vol Solver.
        dtype: the data type of the solver, default: ms.dtype.float32
    """
    """Return: 
        Tuple(vols, converged, failed)
        vols: the implied volatilities given by Implied Vol Solver.
        converged: indicating whether the corresponding root results in an objective
                   function value less than the tolerance.
        failed: indicating whether the corresponding 'root' is not finite.
        root not converged & not failed: limit by the max_iterations, 
                    not fail but also not converged, need to increase the max_iterations
                    to get the needy tolerance / accuracy
    """
    discount_factors, discount_rates = \
        get_discount_rate_factor(discount_factors, discount_rates, der_prices.shape,
                                 expiries, dtype)
    der_prices = der_prices / discount_factors
    obj_prices = spots / discount_factors
    if initial_volatilities is None:
        initial_volatilities = der_prices * np.sqrt(2.0 * np.pi)
    if is_call_options is None:
        is_call_options = np.ones_like(der_prices, dtype=mstype.bool_)
    strikes_abs = np.abs(strikes)
    obj_price_abs = np.abs(obj_prices)
    normalized_mask = ms.Tensor(strikes_abs > obj_price_abs, mstype.bool_)
    normalization = np.where(normalized_mask, strikes_abs, obj_price_abs)
    normalization = np.where(np.equal(normalization, np.zeros(1)),
                             np.ones_like(normalization), normalization)
    der_prices = der_prices / normalization
    obj_prices = obj_prices / normalization
    strikes = strikes / normalization
    if underlying_distribution is ImpliedVolUnderlyingDistribution.BSM:
        func_cell = bsm_lognormal_vega_func(strikes, der_prices, obj_prices,
                                            expiries, discount_factors, is_call_options, dtype)
    elif underlying_distribution is ImpliedVolUnderlyingDistribution.Bachelier:
        func_cell = bachelier_normal_vega_func(strikes, der_prices, obj_prices,
                                               expiries, discount_factors, is_call_options,
                                               normalization, dtype)
    else:
        raise AttributeError("The Underlying Distribution of ImpliedVol is not Supported.")
    result = newton.newton_root_finder(func_cell, initial_volatilities,
                                       max_iterations=max_iterations, tolerance=tolerance
                                       )
    return result


def bsm_lognormal_vega_func(strikes, der_prices, obj_prices,
                            expiries, discount_factors, is_call_options, dtype):
    lnf = np.log(obj_prices) - np.log(strikes)
    sqrt_t = np.sqrt(expiries)

    @ms_function
    def val_vega_func(volatilities):
        vol_t = volatilities * sqrt_t
        d1 = (lnf / vol_t + vol_t / 2)
        d2 = d1 - vol_t
        implied_call = obj_prices * standnormal_cdf(d1) - strikes * standnormal_cdf(d2)
        implied_put = implied_call - obj_prices + strikes
        implied_prices = np.where(is_call_options, implied_call, implied_put)
        vega = obj_prices * standnormal_pdf(d1) * sqrt_t / discount_factors
        return implied_prices - der_prices, vega

    return val_vega_func


# Ref: https://optionsformulas.com/pages/bachelier-with-drift-delta-gamma-and-vega-derivation.html
def bachelier_normal_vega_func(strikes, der_prices, obj_prices, expiries,
                               discount_factors, is_call_options, normalization, dtype):
    sqrt_t = np.sqrt(expiries)

    @ms_function
    def val_vega_func(volatilities):
        vol_t = volatilities * sqrt_t / normalization
        d1 = (obj_prices - strikes) / vol_t
        implied_call = (obj_prices - strikes) * standnormal_cdf(d1) + vol_t * standnormal_pdf(d1)
        implied_put = implied_call - obj_prices + strikes
        implied_price = np.where(is_call_options, implied_call, implied_put)
        vega = sqrt_t * standnormal_pdf(d1) / discount_factors / normalization
        return implied_price - der_prices, vega

    return val_vega_func
