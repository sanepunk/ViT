from torch import nn
import torch

class MLPBlock(nn.Module):
	"""
	A Multi-Layer Perceptron (MLP) block commonly used in transformer architectures, such as the Vision Transformer (ViT).

	This block consists of:
	- Layer Normalization applied to the input.
	- A two-layer MLP with a GELU activation function and dropout for regularization.

	Parameters:
	-----------
	embedding_dim : int, optional (default=768)
		The dimensionality of the input embeddings.
	mlp_size : int, optional (default=3072)
		The number of units in the hidden layer of the MLP.
	dropout : float, optional (default=0.1)
		The dropout probability applied after the activation function and the second linear layer.

	Attributes:
	-----------
	layer_norm : nn.LayerNorm
		Layer normalization module to normalize the input.
	mlp : nn.Sequential
		nn.Sequential container comprising:
		- Linear layer transforming input of size `embedding_dim` to `mlp_size`.
		- GELU activation function.
		- Dropout layer with probability `dropout`.
		- Linear layer transforming input of size `mlp_size` back to `embedding_dim`.
		- Dropout layer with probability `dropout`.

	Methods:
	--------
	forward(x)
		Applies layer normalization followed by the MLP to the input tensor `x`.

	Example:
	--------
	>>> mlp_block = MLPBlock(embedding_dim=768, mlp_size=3072, dropout=0.1)
	>>> input_tensor = torch.randn(64, 128, 768)  # (batch_size, sequence_length, embedding_dim)
	>>> output = mlp_block(input_tensor)
	>>> print(output.shape)
	torch.Size([64, 128, 768])
	"""
	def __init__(self,
				 embedding_dim: int = 768,
				 mlp_size: int = 3072,
				 dropout: float = 0.1):
		super().__init__()
		self.layer = nn.LayerNorm(embedding_dim)

		self.mlp = nn.Sequential(
			nn.Linear(
				embedding_dim,
				mlp_size
			),
			nn.GELU(),
			nn.Dropout(p = dropout),
			nn.Linear(
				mlp_size,
				embedding_dim
			),
			nn.Dropout(p = dropout)
		)

	def forward(self, x):
		x = self.layer(x)
		x = self.mlp(x)
		return x