import torch
from torch import nn
from torch.nn import Parameter
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/TFHub/biggan_v1.py

def l2normalize(v, eps=1e-4):
	return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
	def __init__(self, module, name='weight', power_iterations=1):
		super(SpectralNorm, self).__init__()
		self.module = module
		self.name = name
		self.power_iterations = power_iterations
		if not self._made_params():
			self._make_params()

	def _update_u_v(self):
		u = getattr(self.module, self.name + "_u")
		v = getattr(self.module, self.name + "_v")
		w = getattr(self.module, self.name + "_bar")

		height = w.data.shape[0]
		_w = w.view(height, -1)
		for _ in range(self.power_iterations):
			v = l2normalize(torch.matmul(_w.t(), u))
			u = l2normalize(torch.matmul(_w, v))

		sigma = u.dot((_w).mv(v))
		setattr(self.module, self.name, w / sigma.expand_as(w))

	def _made_params(self):
		try:
			getattr(self.module, self.name + "_u")
			getattr(self.module, self.name + "_v")
			getattr(self.module, self.name + "_bar")
			return True
		except AttributeError:
			return False

	def _make_params(self):
		w = getattr(self.module, self.name)

		height = w.data.shape[0]
		width = w.view(height, -1).data.shape[1]

		u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		u.data = l2normalize(u.data)
		v.data = l2normalize(v.data)
		w_bar = Parameter(w.data)

		del self.module._parameters[self.name]
		self.module.register_parameter(self.name + "_u", u)
		self.module.register_parameter(self.name + "_v", v)
		self.module.register_parameter(self.name + "_bar", w_bar)

	def forward(self, *args):
		self._update_u_v()
		return self.module.forward(*args)

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, eps=1e-4, momentum=0.1):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
    self.gamma_embed = nn.Linear(num_classes, num_features, bias=False)
    self.beta_embed = nn.Linear(num_classes, num_features, bias=False)

  def forward(self, x, y):
    out = self.bn(x)
    gamma = self.gamma_embed(y) + 1
    beta = self.beta_embed(y)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out

        