import mindspore.ops as ops
import mindspore.nn as nn

class FirstOrderGrad(nn.Cell):
    """compute first-order derivative"""
    def __init__(self, model, argnums=0):
        """ Args:
            model (Cell): a function or network that takes Tensor inputs.
            argnum (int): specifies which input the output takes the first derivative of. Default: 0.
        """
        super(FirstOrderGrad, self).__init__()
        self.model = model
        self.argnums = argnums
        self.grad = ops.GradOperation(get_all=True)

    def construct(self, *x):
        """ Inputs
            *x: The input is variable-length argument.
        """
        gradient_function = self.grad(self.model)
        gradient = gradient_function(*x)
        output = gradient[self.argnums]
        return output


class SecondOrderGrad(nn.Cell):
    """compute second-order derivative"""
    def __init__(self, model, argnums=0):
        """ Args:
            model (Cell): a function or network that takes Tensor inputs.
            argnum (int): specifies which input the output takes the first derivative of. Default: 0.
        """
        super(SecondOrderGrad, self).__init__()
        self.grad1 = FirstOrderGrad(model, argnums=argnums)
        self.grad2 = FirstOrderGrad(self.grad1, argnums=argnums)

    def construct(self, *x):
        """ Inputs
            *x: The input is variable-length argument.
        """
        output = self.grad2(*x)
        return output
