from torch import nn
from .MultiHeadAttention import MultiHeadSelfAttention
from .MLPBlock import MLPBlock
import torch


class TransformerEncoderBlock(nn.Module):
	"""
	A PyTorch module representing a single encoder block in a transformer architecture.

	This block consists of:
	- A multi-head self-attention mechanism with a residual connection.
	- A multi-layer perceptron (MLP) with a residual connection.

	Parameters:
	-----------
	embedding_dim : int, optional (default=768)
		The dimensionality of the input embeddings.
	mlp_size : int, optional (default=3072)
		The size of the hidden layer in the MLP.
	mlp_dropout : float, optional (default=0.1)
		The dropout probability applied after each MLP layer.
	num_heads : int, optional (default=12)
		The number of attention heads in the multi-head self-attention mechanism.
	attention_dropout : float, optional (default=0.0)
		The dropout probability applied to the attention weights.

	Attributes:
	-----------
	MSABlock : MultiHeadSelfAttention
		The multi-head self-attention module.
	MLPBlock : MLPBlock
		The multi-layer perceptron module.

	Methods:
	--------
	forward(x)
		Applies the transformer encoder block to the input tensor `x`.

	Example:
	--------
	>>> encoder_block = TransformerEncoderBlock(
	...     embedding_dim=768,
	...     mlp_size=3072,
	...     mlp_dropout=0.1,
	...     num_heads=12,
	...     attention_dropout=0.0
	... )
	>>> input_tensor = torch.randn(64, 128, 768)  # (batch_size, sequence_length, embedding_dim)
	>>> output = encoder_block(input_tensor)
	>>> print(output.shape)
	torch.Size([64, 128, 768])
	"""
	def __init__(self,
				 embedding_dim: int = 768,
				 mlp_size: int = 3072,
				 mlp_dropout: float = 0.1,
				 num_head: int = 12,
				 attention_dropout: float = 0.,
				 # prevResidual: torch.Tensor = None
				 ):
		super().__init__()
		self.MSABlock = MultiHeadSelfAttention(
			embedding_dim=embedding_dim,
			num_head=num_head,
			attention_dropout=attention_dropout)
		self.MLPBlock = MLPBlock(
			embedding_dim=embedding_dim,
			mlp_size=mlp_size,
			dropout=mlp_dropout
		)

	def forward(self, x):
		x = self.MSABlock(x) + x # for residual
		x = self.MLPBlock(x) + x
		return x