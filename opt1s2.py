import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time

class CustomEmbeddingLayer(nn.Module):
    def __init__(self, input_size):
        super(CustomEmbeddingLayer, self).__init__()
        self.embedding = nn.Parameter(torch.ones(input_size, input_size))  # Initialize weights to 1

    def forward(self, x):
        return torch.matmul(x, self.embedding)

class CustomLinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomLinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_size, output_size))  # Initialize weights to 1
        self.bias = nn.Parameter(torch.ones(output_size))  # Initialize bias to 1

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

class TransformerLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(TransformerLayer, self).__init__()
        self.query_weight = nn.Parameter(torch.ones(input_size, input_size))  # Initialize query weight to 1
        self.key_weight = nn.Parameter(torch.ones(input_size, input_size))  # Initialize key weight to 1
        self.value_weight = nn.Parameter(torch.ones(input_size, input_size))  # Initialize value weight to 1
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.feed_forward = nn.Sequential(
            CustomLinearLayer(input_size, hidden_size),  # Custom linear layer with weights initialized to 1
            nn.ReLU(),
            CustomLinearLayer(hidden_size, input_size)  # Custom linear layer with weights initialized to 1
        )
        self.layer_norm2 = nn.LayerNorm(input_size)
        self.num_heads = num_heads

    def forward(self, x, cache=None):
        residual = x
        x = self.layer_norm1(x)
        query = torch.matmul(x, self.query_weight)  # Query matrix
        key = torch.matmul(x, self.key_weight)  # Key matrix
        value = torch.matmul(x, self.value_weight)  # Value matrix
        # Split the query, key, and value matrices into multiple heads
        batch_size, seq_length, _ = query.size()
        query = query.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        # Calculate attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (input_size ** 0.5)  # scaled dot-product attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        # Apply attention to values
        attended_values = torch.matmul(attn_probs, value)
        # Concatenate heads and transpose back
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        x += residual
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x += residual
        return x, attn_probs

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size, num_gpu_batches):
        super(Transformer, self).__init__()
        self.embedding_layer = CustomEmbeddingLayer(input_size)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.output_layer = CustomLinearLayer(hidden_size, output_size)  # Custom linear layer with weights initialized to 1
        self.num_gpu_batches = num_gpu_batches

    def forward(self, x):
        batch_size, seq_length, input_size = x.size()
        # Reshape input tensor for multiple GPU batches
        num_batches = batch_size // self.num_gpu_batches
        x = x.view(self.num_gpu_batches, num_batches, seq_length, input_size).transpose(0, 1)  # [num_batches, num_gpu_batches, seq_length, input_size]
        # Initialize cache for the first batch
        cache = None
        for i, batch in enumerate(x):
            for j, layer in enumerate(self.transformer_layers):
                if j == 0:
                    # For the first layer, use cache from the previous batch if available
                    x[i], cache = layer(batch, cache)
                else:
                    x[i], cache = layer(x[i], cache)
        x = self.output_layer(x[:, -1, :])  # Take only the last token's representation
        return x

OPT_CONFIG = {
    "name": 'opt-1.3',
    "max_seq_len": 2048,
    "num_hidden_layers": 24,
    "n_head": 32,
    "hidden_size": 2048,
    "input_dim": 2048,
    "ffn_embed_dim": 2048 * 4,
    "pad": 1,
    "activation_fn": 'relu',
    "vocab_size": 50272,
    "layer_norm_eps": 0.00001,
    "pad_token_id": 1,
    "dtype": np.float16,
    "num_gpu_batches": 4
}

# Example usage
input_size = OPT_CONFIG['input_dim']  # Input size
hidden_size = OPT_CONFIG['hidden_size']  # Hidden size
num_layers = OPT_CONFIG['num_hidden_layers']  # Number of transformer layers
num_heads = OPT_CONFIG['n_head']  # Number of attention heads
output_size = 1  # Output size
num_gpu_batches = OPT_CONFIG["num_gpu_batches"]

transformer_model = Transformer(input_size, hidden_size, num_layers, num_heads, output_size, num_gpu_batches)

# Step 1: Generate a fake dataset of 10 prompts
fake_dataset = ["This is prompt number {}".format(i) for i in range(10)]

# Step 2: Preprocess the dataset and convert it into tensor format
# Here, we'll just tokenize the prompts (you may have a different preprocessing step)
def tokenize(prompt):
    return [1] * (input_size)  # Placeholder for tokenization result

dataset_tensors = [tokenize(prompt) for prompt in fake_dataset]

# Step 3: Run the dataset through the model and measure the time taken
start_time = time.time()

with torch.no_grad():
    for tensor in tqdm(dataset_tensors, desc="Processing prompts"):
        print("Input prompt:", tensor)  # Print input prompt
        output = transformer_model(torch.tensor(tensor).view(1, input_size, input_size))  # Add batch dimension
        print("Output:", output)  # Print model output

end_time = time.time()

# Step 4: Calculate the throughput
num_prompts = len(fake_dataset)
time_taken = end_time - start_time
throughput = num_prompts / time_taken

print("Number of prompts:", num_prompts)
print("Time taken:", time_taken, "seconds")
print("Throughput:", throughput, "prompts per second")
