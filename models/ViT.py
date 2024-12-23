import torch
from torch import nn
from PatchEmbedding import PatchEmbedding
from Transformer import TransformerEncoderBlock

class ViT(nn.Module):
	"""
	Vision Transformer (ViT) implementation in PyTorch.

	The Vision Transformer applies the transformer architecture, originally designed for natural language processing, to image data.
	It divides an image into fixed-size patches, embeds them, and processes the sequence of patch embeddings using transformer layers.

	Parameters:
	-----------
	image_size : int, optional (default=224)
		The size of the input image (assumes square images).
	in_channels : int, optional (default=3)
		The number of input channels in the image (e.g., 3 for RGB images).
	patch_size : int, optional (default=16)
		The height and width of each patch. The input image dimensions must be divisible by this value.
	num_transformer_layers : int, optional (default=12)
		The number of transformer encoder layers.
	embedding_dim : int, optional (default=768)
		The dimensionality of the embedding vector for each patch.
	mlp_size : int, optional (default=3072)
		The size of the hidden layer in the MLP.
	num_heads : int, optional (default=12)
		The number of attention heads in the multi-head self-attention mechanism.
	attention_dropout : float, optional (default=0.0)
		The dropout probability applied to the attention weights.
	mlp_dropout : float, optional (default=0.1)
		The dropout probability applied after each MLP layer.
	embedding_dropout : float, optional (default=0.1)
		The dropout probability applied to the patch and positional embeddings.
	num_classes : int, optional (default=1000)
		The number of output classes for classification.

	Attributes:
	-----------
	num_patches : int
		The total number of patches extracted from the input image.
	class_embedding : nn.Parameter
		A learnable embedding that represents the classification token.
	positional_embedding : nn.Parameter
		A learnable positional embedding added to the patch embeddings.
	embedding_dropout : nn.Dropout
		Dropout layer applied to the embeddings.
	patch_embedding : PatchEmbedding
		Module for extracting and embedding image patches.
	transformerEncoderBlock : nn.Sequential
		A sequence of transformer encoder blocks.
	classifier : nn.Sequential
		The classification head consisting of a layer normalization and a linear layer.

	Methods:
	--------
	forward(x)
		Processes the input tensor `x` through the Vision Transformer pipeline and returns the classification logits.

	Example:
	--------
	>>> vit = ViT(image_size=224, patch_size=16, num_classes=1000)
	>>> input_tensor = torch.randn(32, 3, 224, 224)  # (batch_size, channels, height, width)
	>>> output = vit(input_tensor)
	>>> print(output.shape)
	torch.Size([32, 1000])
	"""
	def __init__(
			self,
			image_size: int = 224,
			in_channels: int = 3,
			patch_size: int = 16,
			num_transformer_layers: int = 12, # According to paper ViT(Base)
			embedding_dim: int = 768, # hidden size from paper
			mlp_size: int = 3072,
			num_heads: int = 12,
			attention_dropout: float = 0.,
			mlp_dropout: float = 0.1,
			embedding_dropout: float = 0.1,
			num_classes: int = 1000
				 ):
		super().__init__()

		# make an assertion that image size is compatible with patch size
		assert image_size % patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_size}, patch_size: {patch_size}"
		# calculate the number of patches for positional embedding
		# (height * width) / patch ^ 2
		self.num_patches = (image_size * image_size) // patch_size ** 2

		# Create learnable class embedding (for front of patch embedding)
		# Learnable classification token; this token is prepended to the sequence of patch embeddings
		# and is used to aggregate information for classification tasks.
		self.class_embedding = nn.Parameter(
			data=torch.randn(1, 1, embedding_dim),
			requires_grad=True
		)

		# Create Position Embedding
		# Learnable positional embeddings; these provide information about the position of each patch
		# in the sequence, enabling the model to capture spatial relationships.
		self.positional_embedding = nn.Parameter(
			data = torch.randn(1, self.num_patches + 1, embedding_dim),
			requires_grad=True
		)

		# Create embedding dropout
		self.embedding_dropout = nn.Dropout(embedding_dropout)

		# Create patch embedding layer
		# Patch embedding layer; this layer divides the image into patches and projects each patch
		# into a vector of size `embedding_dim`.
		self.patch_embedding = PatchEmbedding(
			input_shape=in_channels,
			patch_size=patch_size,
			embedding_dim=embedding_dim
		)

		# Create the Transformer Encoder Block
		# Transformer encoder layers; a sequence of transformer blocks that process the patch embeddings.
		# Each block consists of multi-head self-attention and MLP layers with residual connections.
		self.transformerEncoderBlock = nn.Sequential(
			*[
				TransformerEncoderBlock(
					embedding_dim=embedding_dim,
					num_head=num_heads,
					mlp_size=mlp_size,
					mlp_dropout=mlp_dropout,
					attention_dropout=attention_dropout
				)
				 for _ in range(num_transformer_layers)
			]
		)

		# Create classifier head
		self.classifier = nn.Sequential(
			nn.LayerNorm(normalized_shape=embedding_dim),
			nn.Linear(
				in_features=embedding_dim,
				out_features=num_classes
			)
		)

	def forward(self, x):
		"""
        Forward pass of the Vision Transformer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, image_size, image_size).

        Returns:
        --------
        torch.Tensor
            Classification logits of shape (batch_size, num_classes).
        """

		# Get batch size
		batch_size = x.shape[0]

		# Create and expand the class token embedding to match the batch size.
		# This token is concatenated with the patch embeddings and will capture
		# the aggregated information for classification.
		class_token = self.class_embedding.expand(batch_size, -1, -1)
		# -1 means to infer the dimensions
		# since we are going to merge it with the batches of images and its (class_token) defined only for one image so we need to do this
		# (1, 1, 768) -> (batch_size, 1, 768)

		# Create the patch embedding
		# Extract patch embeddings from the input image.
		# The image is divided into patches, each of which is flattened and projected
		# into the embedding space.
		x = self.patch_embedding(x)

		# Concatenate the class token with the patch embeddings along the sequence dimension.
		# The class token is placed at the beginning of the sequence.
		x = torch.cat([class_token, x], dim = 1)

		# Add positional embeddings to the patch embeddings to retain spatial information.
		# This step encodes the position of each patch in the original image.
		x = self.positional_embedding + x

		# Apply dropout to patch embedding
		x = self.embedding_dropout(x)

		# Pass position and patch embedding to transformer
		x = self.transformerEncoderBlock(x)

		# Take the zeroth index of last layer and pass it through the classifier
		x = x[:, 0]
		x = self.classifier(x)
		return x