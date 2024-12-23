from torch import nn
import torch

class PatchEmbedding(nn.Module):
	"""
	A PyTorch module that converts input images into a sequence of patch embeddings, as used in Vision Transformers (ViTs).

	This module performs the following steps:
	1. Divides the input image into non-overlapping patches of a specified size.
	2. Projects each patch into a vector of a given embedding dimension using a convolutional layer.
	3. Flattens and rearranges the projected patches to form a sequence suitable for transformer models.

	Parameters:
	-----------
	input_shape : int, optional (default=3)
		The number of input channels in the image (e.g., 3 for RGB images).
	patch_size : int, optional (default=16)
		The height and width of each patch. The input image dimensions must be divisible by this value.
	embedding_dim : int, optional (default=768)
		The dimensionality of the embedding vector for each patch.

	Attributes:
	-----------
	patch_size : int
		Stores the size of each patch.
	patcher : nn.Conv2d
		A convolutional layer that extracts and projects patches from the input image.
	flatten : nn.Flatten
		A layer that flattens the spatial dimensions of the patches.

	Methods:
	--------
	forward(x)
		Processes the input tensor `x` through the patch embedding pipeline and returns the sequence of patch embeddings.

	Example:
	--------
	>>> patch_embed = PatchEmbedding(input_shape=3, patch_size=16, embedding_dim=768)
	>>> input_tensor = torch.randn(32, 3, 224, 224)  # (batch_size, channels, height, width)
	>>> output = patch_embed(input_tensor)
	>>> print(output.shape)
	torch.Size([32, 196, 768])
	"""
	def __init__(self,
				 input_shape: int = 3,
				 patch_size: int = 16,
				 embedding_dim: int = 768):
		super().__init__()
		self.patch_size = patch_size
		self.patcher = nn.Conv2d(
			in_channels=input_shape,
			out_channels=embedding_dim,
			kernel_size=patch_size,
			stride=patch_size,
			padding=0
		)
		self.flatten = nn.Flatten(start_dim=2, end_dim=3)

	def forward(self, x):
		image_resolution = x.shape[-1]
		assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch_size: {self.patch_size}"

		x = self.patcher(x)
		x = self.flatten(x)
		x = x.permute(0, 2, 1)
		return x