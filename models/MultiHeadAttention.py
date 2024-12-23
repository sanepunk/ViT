from torch import nn
import torch


class MultiHeadSelfAttention(nn.Module):
	"""
	A PyTorch module that applies multi-head self-attention to input sequences, commonly used in transformer architectures.

	This module consists of:
	- Layer normalization applied to the input.
	- Multi-head self-attention mechanism.

	Parameters:
	-----------
	embedding_dim : int, optional (default=768)
		The dimensionality of the input embeddings.
	num_heads : int, optional (default=12)
		The number of attention heads.
	attention_dropout : float, optional (default=0.0)
		The dropout probability applied to the attention weights.

	Attributes:
	-----------
	layer_norm : nn.LayerNorm
		Layer normalization module to normalize the input.
	multi_head_attention : nn.MultiheadAttention
		Multi-head attention module to perform self-attention on the input sequence.

	Methods:
	--------
	forward(x)
		Applies layer normalization followed by multi-head self-attention to the input
		tensor `x`.

	Example:
	--------
	>>> self_attention = MultiHeadSelfAttention(embedding_dim=768, num_heads=12, attention_dropout=0.1)
	>>> input_tensor = torch.randn(64, 128, 768)  # (batch_size, sequence_length, embedding_dim)
	>>> output = self_attention(input_tensor)
	>>> print(output.shape)
	torch.Size([64, 128, 768])
	"""
	def __init__(self,
				 embedding_dim: int = 768,
				 num_head: int = 12,
				 attention_dropout: float = 0.):
		# create layer norm
		super().__init__()
		self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

		# create Multi Head Self Attention
		self.multi_head_attention = nn.MultiheadAttention(
			embed_dim=embedding_dim,
			num_heads=num_head,
			dropout=attention_dropout,
			batch_first=True)

	def forward(self, x):
		x = self.layer_norm(x)
		attention_output, _ = self.multi_head_attention(
			query = x,
			key = x,
			value = x
		)
		return attention_output