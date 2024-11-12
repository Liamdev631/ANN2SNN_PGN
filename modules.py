import torch
import torch.nn as nn

class IF(nn.Module):
    def __init__(self, threshold: float | torch.Tensor, noise: float = 0.0, dt: float = 1.0):
        super().__init__()
        with torch.no_grad():
            if isinstance(threshold, float):
                self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)
            else:
                self.threshold = nn.Parameter(threshold, requires_grad=False)
            self.noise = noise
            self.dt = dt
        self.is_initialized: bool = False

    def forward(self, x):
        if not self.is_initialized:
            self.is_initialized = True
            self.forward_init(x)
        
        noise_sig = torch.randn_like(x) * self.noise
        p = self.v + (x + noise_sig) * self.dt
        self.spikes = p > self.threshold
        psp = self.spikes * self.threshold
        self.v = p - psp
        return psp

    def forward_init(self, x):
        self.v = torch.zeros_like(x)
        self.reset()

    def reset(self):
        if not self.is_initialized:
            return
        self.v.zero_()
        self.spikes = torch.zeros_like(self.v, dtype=torch.bool)

# Group neuron
class GN(IF):
    def __init__(self, threshold: float, tau: int = 4, noise: float = 0.0, dt: float = 1.0):
        super().__init__(threshold / tau, noise, dt)
        with torch.no_grad():
            self.tau = nn.Parameter(data=torch.tensor(tau), requires_grad=False)
            self.subthreshold = nn.Parameter(data=torch.arange(1, tau+1).float() * threshold / tau, requires_grad=False)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1).expand(*([-1] * len(x.shape)), self.tau)
        if not self.is_initialized:
            self.is_initialized = True
            self.forward_init(x)

        noise_sig = torch.randn_like(x) * self.noise
        p = self.v + (x + noise_sig) * self.dt
        self.spikes = p > self.subthreshold
        self.v = p - self.threshold * self.spikes.sum(dim=-1, dtype=torch.float32, keepdim=True)
        return self.spikes.sum(dim=-1, dtype=torch.float32, keepdim=False) * self.threshold

# Reduced Threshold
class RT(IF):
    def __init__(self, threshold: float, tau: int = 4, noise: float = 0.0, dt: float = 1.0):
        super().__init__(threshold / tau, noise, dt)

# Phased Group neuron
class PGN(IF):
    def __init__(self, threshold: float, tau: int = 4, noise: float = 0.0, dt: float = 1.0):
        super().__init__(threshold, noise, dt)
        with torch.no_grad():
            self.tau = nn.Parameter(data=torch.tensor(tau), requires_grad=False)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1).expand(*([-1] * len(x.shape)), self.tau)
        super().forward(x) # Use IF logic but scale down the output spikes
        return self.spikes.mean(dim=-1, dtype=torch.float32) * self.threshold

    def reset(self):
        if not self.is_initialized:
            return
        init_shape = self.v.shape
        v_init = torch.arange(0, self.tau + 1, device=self.v.device)[:-1] * self.threshold / self.tau
        v_init = v_init.view(*([1] * (len(self.v.shape)-1)), *v_init.shape)
        self.v = v_init.repeat(*self.v.shape[:-1], 1)
        assert init_shape == self.v.shape
        self.spikes = torch.zeros_like(self.v, dtype=torch.bool)

# QCFS is the reproduction of the paper "QCFS"
class QCFS(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.v_threshold = nn.Parameter(torch.tensor(float(T)))
        self.p0 = torch.tensor(0.5)

    @staticmethod
    def floor_ste(x):
        return (x.floor() - x).detach() + x

    def extra_repr(self):
        return f"T={self.T}, p0={self.p0.item()}, v_threshold={self.v_threshold.item()}"

    def forward(self, x):
        y = self.floor_ste(x * self.T / self.v_threshold + self.p0)
        y = torch.clamp(y, 0, self.T) 
        return y* self.v_threshold / self.T

def reset_net(model: nn.Module):
	for module in model.modules():
		if issubclass(module.__class__, IF):
			module.is_initialized = False
			module.reset()