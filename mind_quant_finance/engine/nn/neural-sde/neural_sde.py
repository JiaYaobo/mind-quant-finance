import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as P
from mind_quant_finance.engine.nn.composed.mlp import MLP

class DriftField(nn.Cell):
    net: MLP
    _hidden_size: int
    _width_size: int
    _depth: int

    def __init__(self, hidden_size, width_size, depth):
        super(DriftField, self).__init__()
        self._hidden_size = hidden_size
        self._width_size = width_size
        self._depth = depth
        self.net = MLP(in_size=hidden_size + 1, out_size=hidden_size, width_size=width_size, depth=depth, final_activation=nn.Tanh())

    def construct(self, t, y):
        return self.net(mnp.concatenate([t, y]))


class SigmaField(nn.Cell):
    _hidden_size: int
    _width_size: int
    _noise_size: int
    
    def __init__(self, noise_size, hidden_size, width_size, depth):
        super(SigmaField, self).__init__()
        self._hidden_size = hidden_size
        self._width_size = width_size
        self._noise_size = noise_size
        self.net = MLP(in_size=hidden_size + 1, out_size=hidden_size, width_size=width_size * noise_size, depth=depth, final_activation=nn.Tanh())

    def construct(self, t, y):
        return self.net(mnp.concatenate([t, y])).reshape(
            self.hidden_size, self.noise_size
        )


class SDEStep(nn.Cell):
    drift_field: DriftField
    sigma_field: SigmaField
    noise_size: int

    def __init__(self,
        noise_size,
        hidden_size,
        drift_width_size,
        sigma_width_size,
        drift_depth,
        sigma_depth,
        **kwargs):
        super().__init__(**kwargs)
        self.drift_field = DriftField(hidden_size, drift_width_size, drift_depth)
        self.sigma_field = SigmaField(
            noise_size, hidden_size, sigma_width_size, sigma_depth
        )

        self.noise_size = noise_size
    

    def construct(self, inputs):
        (i, t0, dt, y0) = inputs
        t = mnp.full((1, ), t0 + dt)
        bm = mnp.randn((self.noise_size, )) * mnp.sqrt(dt)
        drift_term = self.drift_field(t=t, y=y0) * dt
        diffusion_term = mnp.dot(self.sigma_field(t=t, y=y0), bm)
        y1 = y0 + drift_term + diffusion_term
        outputs = (i+1, t0, dt, y1)

        return outputs, y1

class NeuralSDE(nn.Cell):
    step: SDEStep
    noise_size: int
    hidden_size: int
    drift_depth: int
    sigma_depth: int
    drift_width_size: int
    sigma_width_size: int

    
    def __init__(
        self,
        noise_size,
        hidden_size,
        drift_width_size,
        sigma_width_size,
        drift_depth,
        sigma_depth,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.step = SDEStep(
            noise_size=noise_size,
            hidden_size=hidden_size,
            mu_width_size=drift_width_size,
            sigma_width_size=sigma_width_size,
            drift_depth=drift_depth,
            sigma_depth=sigma_depth
            )

        self.noise_size = noise_size
        self.hidden_size = hidden_size
        self.drift_width_size = drift_width_size
        self.sigma_width_size = sigma_width_size
        self.drift_depth = drift_depth
        self.sigma_depth = sigma_depth
    

    def construct(self, y0, ts):
        t0 = ts[0]
        num_timesteps = len(ts)        
        i = 1
        ys = mnp.zeros_like((num_timesteps, self.hidden_size))
        ys[0] = y0
        while i < num_timesteps:
            inputs = (i, ts[i - 1], ts[i] - ts[i - 1], ys[i - 1])
            _, y_i = self.step(inputs)
            ys[i] = y_i

        return ys
