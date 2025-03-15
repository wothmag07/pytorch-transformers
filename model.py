import torch.nn as nn
import math
import torch

class InputEmbedding(nn.Module):
    """
    This class creates an embedding layer for input tokens.
    It maps each token in the vocabulary to a d_model-dimensional vector.
    """

    def __init__(self, d_model, vocab_size):
        """
        Initializes the embedding layer.

        Args:
            d_model (int): The dimension of each embedding vector.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # The embedding layer maps each token index to a d_model-dimensional vector
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass of the embedding layer.

        Args:
            x (Tensor): Input tensor containing token indices.

        Returns:
            Tensor: Scaled embedding tensor.
        """
        # Scale embeddings by sqrt(d_model) to maintain variance stability
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    This class adds positional encodings to token embeddings.
    It helps the model understand word order since transformers lack recurrence.
    """

    def __init__(self, d_model, seq_length, dropout=0.1):
        """
        Initializes the positional encoding module.

        Args:
            d_model (int): The embedding dimension.
            seq_length (int): The maximum sequence length.
            dropout (float, optional): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

        # Create a matrix of zeros (shape: [seq_length, d_model])
        pe = torch.zeros(seq_length, d_model)

        # Create a tensor for positions (shape: [seq_length, 1])
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)

        # Compute the sinusoidal frequency factors (shape: [d_model/2])
        divTerm = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices and cos to odd indices in each position vector
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)

        # Add a batch dimension (shape: [1, seq_length, d_model])
        pe = pe.unsqueeze(0)

        # Register pe as a buffer (won't be updated during training)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass to add positional encodings.

        Args:
            x (Tensor): Input embeddings (batch_size, seq_length, d_model).

        Returns:
            Tensor: Positional encoding added to embeddings.
        """
        # Add positional encoding (ensure it's not trainable)
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)

        return self.dropout(x)


class LayerNorm(nn.Module):
    """
    Custom implementation of Layer Normalization.
    It normalizes inputs across the last dimension (features) 
    and applies learnable scale (alpha) and shift (bias) parameters.
    """

    def __init__(self, d_model, epsilon= 1e-7):
        """
        Initializes the LayerNorm module.

        Args:
            d_model (int): Number of features in the input tensor.
            epsilon (float, optional): Small value to prevent division by zero.
        """
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(d_model))  # Scale factor
        self.bias = nn.Parameter(torch.zeros(d_model))  # Shift factor

    def forward(self, x):
        """
        Applies layer normalization to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            Tensor: Layer-normalized output of the same shape as input.
        """
        # Compute mean and standard deviation along the last dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)  # Used population std

        # Normalize and apply learnable scale/shift parameters
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Implements a Position-wise FeedForward Network used in Transformers.
    It consists of two linear layers with a non-linearity (ReLU or GELU) in between.
    """

    def __init__(self, d_model, d_ffn, dropout= 0.1, use_gelu= False):
        """
        Initializes the FeedForwardNetwork.

        Args:
            d_model (int): Dimensionality of the model (input and output size).
            d_ffn (int): Dimensionality of the hidden layer (usually 4 * d_model).
            dropout (float, optional): Dropout probability.
            use_gelu (bool, optional): Whether to use GELU instead of ReLU. Default is False.
        """
        super().__init__()

        # First linear layer expands to a higher-dimensional space
        self.linear1 = nn.Linear(d_model, d_ffn)

        # Second linear layer projects back to d_model dimensions
        self.linear2 = nn.Linear(d_ffn, d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # GELU is often preferred in modern architectures
        self.activation = nn.GELU() if use_gelu else nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the feedforward network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            Tensor: Transformed output tensor of the same shape as input.
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# ============================ Dimension Flow in Multi-Head Attention ============================

#  step 1  -> Input Q, K, V from linear layers  | (B, L, d_model)
#  step 2  -> Reshape: Split into multiple heads | (B, L, H, d_k) where d_k = d_model / H
#  step 3  -> Transpose: Move heads to second dim | (B, H, L, d_k)
#  step 4  -> Compute attention scores (QK^T / sqrt(d_k)) | (B, H, L, L)
#  step 5  -> Apply softmax to get attention weights | (B, H, L, L)
#  step 6  -> Multiply attention weights with V (Weighted sum) | (B, H, L, d_k)
#  step 7  -> Transpose & Reshape: Concatenate heads | (B, L, d_model)
#  step 8  -> Apply output projection (W_O) | (B, L, d_model)

# ===============================================================================================



class MultiHeadAttentionBlock(nn.Module):
    """
    Implements Multi-Head Self-Attention as used in Transformer models.
    It splits input tensors into multiple heads, applies scaled dot-product attention,
    and recombines them into the original shape.
    """

    def __init__(self, d_model, num_heads, dropout= 0.1):
        """
        Initializes the MultiHeadAttention block.

        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout probability.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of each head
        self.dropout = nn.Dropout(dropout)

        # Linear projections for query, key, and value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final output projection layer
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        Computes scaled dot-product attention.

        Args:
            query (Tensor): Query tensor of shape (batch, num_heads, seq_len, d_k).
            key (Tensor): Key tensor of shape (batch, num_heads, seq_len, d_k).
            value (Tensor): Value tensor of shape (batch, num_heads, seq_len, d_k).
            mask (Tensor, optional): Attention mask (batch, 1, 1, seq_len).
            dropout (nn.Dropout, optional): Dropout layer.

        Returns:
            Tensor: Attention-weighted values.
            Tensor: Attention scores.
        """
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if not none
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attention_scores = attention_scores.softmax(dim=-1)

        # Apply dropout if not none
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_length, d_model).
            k (Tensor): Key tensor of shape (batch_size, seq_length, d_model).
            v (Tensor): Value tensor of shape (batch_size, seq_length, d_model).
            mask (Tensor, optional): Attention mask (batch, 1, 1, seq_len).

        Returns:
            Tensor: Output of multi-head attention (batch_size, seq_length, d_model).
        """
        batch_size = q.shape[0]

        # Apply linear projections
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) --> (batch_size, num_heads, seq_len, d_k)
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention mechanism
        x, self.attention_scores = self.attention(query, key, value, mask, dropout=self.dropout)

        # Concatenate heads and project back
        # (batch_size, num_heads, seq_len, d_k) --> (batch_size, seq_len, num_heads, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Apply final linear transformation
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by Layer Normalization.
    This is used in Transformer models to allow gradients to flow 
    easily through the network while stabilizing training.
    """

    def __init__(self, d_model, dropout):
        """
        Initializes the residual connection block.

        Args:
            d_model (int): Model dimension for Layer Normalization.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.norm = LayerNorm(d_model=d_model)  # Ensure LayerNorm supports d_model if needed
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Forward pass of residual connection.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            sublayer (Callable): Function/layer to apply on normalized input.

        Returns:
            Tensor: Output tensor after residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))  # Residual connection

class EncoderBlock(nn.Module):
    """
    A single encoder block in a Transformer model. 
    It consists of a self-attention mechanism followed by a feed-forward network, 
    both wrapped with residual connections and layer normalization.
    """

    def __init__(self, self_attention, 
                 feed_forward, dropout):
        """
        Initializes the encoder block.

        Args:
            self_attention (MultiHeadAttentionBlock): Multi-head self-attention module.
            feed_forward (FeedForwardNetwork): Feed-forward neural network.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        
        # Use nn.ModuleList to properly register residual layers as PyTorch submodules
        self.residuals = nn.ModuleList([ResidualConnection(d_model=self.self_attention.d_model, dropout=dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Forward pass of the encoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (Tensor): Masking tensor for self-attention.

        Returns:
            Tensor: Processed tensor of shape (batch_size, seq_len, d_model).
        """
        # Apply self-attention with residual connection
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        
        # Apply feed-forward network with residual connection
        x = self.residuals[1](x, self.feed_forward)

        return x


class Encoder(nn.Module):
    """
    Transformer Encoder: Stacks multiple EncoderBlocks and applies final Layer Normalization.
    """

    def __init__(self, layers: list):
        """
        Initializes the Transformer Encoder.

        Args:
            layers (list): A list of EncoderBlock modules.
        """
        super().__init__()
        
        # Ensure layers are properly registered as PyTorch submodules
        self.layers = nn.ModuleList(layers)  

        # Final layer normalization
        self.norm = LayerNorm(layers[0].self_attention.d_model)

    def forward(self, x, mask):
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor (batch_size, seq_len, d_model).
            mask (Tensor): Mask tensor for self-attention.

        Returns:
            Tensor: Processed encoder output (batch_size, seq_len, d_model).
        """
        # Pass through each EncoderBlock
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final layer normalization
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    A single decoder block in a Transformer model. 
    This block consists of self-attention, cross-attention (with encoder output), 
    and a feed-forward network, each followed by residual connections and layer normalization.
    """

    def __init__(self, self_attention, 
                 cross_attention, 
                 feed_forward, dropout):
        """
        Initializes the DecoderBlock.

        Args:
            self_attention (MultiHeadAttentionBlock): Multi-head self-attention mechanism for the decoder.
            cross_attention (MultiHeadAttentionBlock): Multi-head attention mechanism to attend to encoder output.
            feed_forward (FeedForwardBlock): Feed-forward network applied to the decoder output.
            dropout (float): Dropout probability to prevent overfitting.
        """
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward

        # Create a list of residual connection blocks for each sublayer (self-attention, cross-attention, feed-forward)
        self.residuals = nn.ModuleList([
            ResidualConnection(d_model=self.self_attention.d_model, dropout=dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the decoder block, applying self-attention, cross-attention, and feed-forward network.

        Args:
            x (Tensor): The input tensor to the decoder (batch_size, seq_len, d_model).
            encoder_output (Tensor): Output tensor from the encoder (batch_size, seq_len, d_model).
            src_mask (Tensor): The mask for the source sequence to prevent attention to padding tokens (batch_size, 1, 1, seq_len).
            tgt_mask (Tensor): The mask for the target sequence to prevent attention to future tokens (batch_size, 1, seq_len, seq_len).

        Returns:
            Tensor: The output tensor after applying the self-attention, cross-attention, and feed-forward transformations.
        """
        # Apply self-attention with residual connection
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))  # Self-attention

        # Apply cross-attention with encoder output and residual connection
        x = self.residuals[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))  # Cross-attention

        # Apply feed-forward network with residual connection
        x = self.residuals[2](x, lambda x: self.feed_forward(x))  # Feed-forward

        return x

class Decoder(nn.Module):
    """
    Transformer Decoder: Stacks multiple DecoderBlocks and applies final Layer Normalization.
    It consists of a series of decoder blocks, each applying self-attention, cross-attention (to encoder output), 
    and a feed-forward network, followed by residual connections and layer normalization.
    """

    def __init__(self, layers):
        """
        Initializes the Decoder module with a list of DecoderBlocks.

        Args:
            layers (list): A list of DecoderBlock modules. Each DecoderBlock applies self-attention, 
                           cross-attention, and a feed-forward network, with residual connections and normalization.
        """
        super().__init__()
        
        # Register all decoder layers as submodules of the Decoder class
        self.layers = nn.ModuleList(layers)

        # Final layer normalization after all decoder layers
        self.norm = LayerNorm(layers[0].self_attention.d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the decoder. Processes input through multiple DecoderBlocks.

        Args:
            x (Tensor): Input tensor (batch_size, seq_len, d_model), typically the target sequence embeddings.
            encoder_output (Tensor): Output tensor from the encoder (batch_size, seq_len, d_model).
            src_mask (Tensor): Source sequence mask (batch_size, 1, 1, seq_len), to prevent attention to padding tokens in the source.
            tgt_mask (Tensor): Target sequence mask (batch_size, 1, seq_len, seq_len), to prevent attention to future tokens in the target.

        Returns:
            Tensor: Output tensor after applying the decoder layers (batch_size, seq_len, d_model).
        """
        # Pass the input through each decoder block
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Apply final layer normalization
        return self.norm(x)
class ProjectionLayer(nn.Module):
    """
    A projection layer that maps the output of the decoder to the vocabulary space.
    It is used to convert the decoder's output into logits for each token in the vocabulary.
    The output is then passed through a softmax function to produce probabilities.
    """

    def __init__(self, d_model, vocab_size):
        """
        Initializes the projection layer.

        Args:
            d_model (int): The dimension of the model (i.e., the output size from the decoder).
            vocab_size (int): The size of the vocabulary, i.e., the number of possible tokens.
        """
        super().__init__()
        
        # Linear transformation that projects from d_model dimension to vocab_size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass through the projection layer. It maps the model's output to vocabulary logits and applies log softmax.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model), typically from the decoder.

        Returns:
            Tensor: Log-probabilities of each token in the vocabulary, of shape (batch_size, seq_len, vocab_size).
        """
        # Apply linear projection and log-softmax to obtain log-probabilities
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformers(nn.Module):
    """
    The main Transformer model.
    
    It consists of:
    - Input Embeddings for source & target sequences
    - Positional Encoding to maintain word order
    - Encoder to process the input sequence
    - Decoder to generate predictions
    - A final Projection Layer to map outputs to vocabulary
    
    This model is commonly used for tasks like language translation.
    """

    def __init__(self, encoder, decoder, src_embedd, tgt_embedd, 
                 src_pos_enc, tgt_pos_enc, projLayer):
        """
        Initializes the Transformer model.

        Args:
            encoder (Encoder): The encoder part of the transformer.
            decoder (Decoder): The decoder part of the transformer.
            src_embedd (InputEmbedding): Word embeddings for the input sequence.
            tgt_embedd (InputEmbedding): Word embeddings for the target sequence.
            src_pos_enc (PositionalEncoding): Positional encoding for the input sequence.
            tgt_pos_enc (PositionalEncoding): Positional encoding for the target sequence.
            projLayer (ProjectionLayer): The final layer that maps output to vocabulary size.
        """
        super().__init__()
        self.src_embedd = src_embedd
        self.tgt_embedd = tgt_embedd
        self.src_pos_enc = src_pos_enc
        self.tgt_pos_enc = tgt_pos_enc
        self.encoder = encoder
        self.decoder = decoder
        self.proj = projLayer

    def encode(self, src, src_mask):
        """
        Encodes the input sequence by applying embeddings, positional encoding, 
        and then passing it through the encoder.

        Args:
            src (Tensor): Input sequence (batch_size, seq_length).
            src_mask (Tensor): Mask to avoid attending to padding tokens.

        Returns:
            Tensor: Encoded representation of the input sequence.
        """
        src = self.src_embedd(src)
        src = self.src_pos_enc(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Decodes the target sequence using the encoder output.

        Args:
            encoder_output (Tensor): The processed input sequence from the encoder.
            src_mask (Tensor): Mask for the source sequence.
            tgt (Tensor): Target sequence (batch_size, seq_length).
            tgt_mask (Tensor): Mask for the target sequence to prevent looking ahead.

        Returns:
            Tensor: The decoder's output, which will be mapped to words.
        """
        tgt = self.tgt_embedd(tgt)
        tgt = self.tgt_pos_enc(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def linearlayer(self, x):
        """
        Applies the final linear layer to map the model's output to vocabulary words.

        Args:
            x (Tensor): The decoder’s output.

        Returns:
            Tensor: Logits representing word probabilities.
        """
        return self.proj(x)
    
def buildTransformer(src_vocab, tgt_vocab, src_seq_len, tgt_seq_len, 
                     d_model=512, num_blocks=6, num_heads=8, 
                     dropout=0.1, d_ffn=2048):
    """
    Creates a Transformer model with the given parameters.

    Args:
        src_vocab (int): Number of words in the source language.
        tgt_vocab (int): Number of words in the target language.
        src_seq_len (int): Maximum length of the input sequence.
        tgt_seq_len (int): Maximum length of the target sequence.
        d_model (int, optional): Size of each embedding vector. Default is 512.
        num_blocks (int, optional): Number of encoder and decoder layers. Default is 6.
        num_heads (int, optional): Number of attention heads. Default is 8.
        dropout (float, optional): Dropout rate for regularization. Default is 0.1.
        d_ffn (int, optional): Size of the hidden layer in the feedforward network. Default is 2048.

    Returns:
        Transformers: A fully constructed Transformer model.
    """
    
    # Word embeddings for input and target sequences
    src_embedd = InputEmbedding(d_model, src_vocab)
    tgt_embedd = InputEmbedding(d_model, tgt_vocab)

    # Positional encodings for input and target sequences
    src_pos_enc = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_enc = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder layers
    encoder_blocks = []
    for _ in range(num_blocks):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ffn, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create decoder layers
    decoder_blocks = []
    for _ in range(num_blocks):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ffn, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, 
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Construct the encoder and decoder
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)

    # Final projection layer that maps model outputs to vocabulary
    proj_layer = ProjectionLayer(d_model, tgt_vocab)

    # Assemble the Transformer model
    transformer = Transformers(encoder, decoder, src_embedd, tgt_embedd, src_pos_enc, tgt_pos_enc, proj_layer)

    # Initialize model parameters for better training stability
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

def build_and_wrap_transformer(*args, rank, world_size, **kwargs):
    """
    Creates a Transformer model and wraps it with DistributedDataParallel.

    Args:
        *args: Positional arguments to be passed to buildTransformer.
        rank (int): The rank of the current process (GPU).
        world_size (int): Total number of processes (GPUs).
        **kwargs: Keyword arguments to be passed to buildTransformer.

    Returns:
        Transformer or nn.DistributedDataParallel: The Transformer model wrapped with DDP.
    """
    # Build the transformer model
    transformer = buildTransformer(*args, **kwargs)
    
    # Move model to the appropriate device (rank)
    transformer = transformer.to(rank)
    
    # Wrap the model with DistributedDataParallel
    transformer = nn.parallel.DistributedDataParallel(transformer, device_ids=[rank], output_device=rank)

    return transformer
