import mindspore.numpy as np
from mindspore import dtype as mstype


def get_discounted_rate_factor(discounted_factors, discounted_rates, shape,
                             expiries, dtype=mstype.float32):
    """Args:
        discounted_factors: A ms.Tensor with the same shape of der_price/ spot.
                         the discount factor of the der_price/spot. (e^{-rT})
        discounted_rates: A ms.Tensor with the same shape of der_price/spot.
                       The discount rate of the der_price/spote. (r)
                       at most one of the factor/rate can be supplied.
                       if both are None, it means discount_rate = 1.0 (Default/No discount)
        shape: A ms.shape which has define the shape of discount_factor/discount_rate
        expiries: The expiry of the product.
        dtype: default ms.dtype.float32
    """
    """Return:
        the corresponding discount_factor / discount_rate 
        discount_factor = exp(- expiries * discount_rate).
    """
    if (discounted_rates is not None) and (discounted_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may '
                         'be supplied')
    elif (discounted_factors is None) and (discounted_rates is not None):
        discounted_factors = np.exp(-discounted_rates * expiries)
    elif (discounted_rates is None) and (discounted_factors is not None):
        discounted_rates = -np.log(discounted_factors) / expiries
    else:
        discounted_rates = np.zeros(shape, dtype=dtype)
        discounted_factors = np.ones(shape, dtype=dtype)
    return discounted_factors, discounted_rates
