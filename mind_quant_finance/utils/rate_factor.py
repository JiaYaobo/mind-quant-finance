import mindspore.numpy as np
from mindspore import dtype as mstype


def get_discount_rate_factor(discount_factors, discount_rates, shape,
                             expiries, dtype=mstype.float32):
    """Args:
        discount_factors: A ms.Tensor with the same shape of der_price/ spot.
                         the discount factor of the der_price/spot. (e^{-rT})
        discount_rates: A ms.Tensor with the same shape of der_price/spot.
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
    if (discount_rates is not None) and (discount_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may '
                         'be supplied')
    elif (discount_factors is None) and (discount_rates is not None):
        discount_factors = np.exp(-discount_rates * expiries)
    elif (discount_rates is None) and (discount_factors is not None):
        discount_rates = -np.log(discount_factors) / expiries
    else:
        discount_rates = np.zeros(shape, dtype=dtype)
        discount_factors = np.ones(shape, dtype=dtype)
    return discount_factors, discount_rates
