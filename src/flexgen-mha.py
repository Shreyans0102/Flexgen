import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List

from transformers import AutoTokenizer

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(InputEmbedding, self).__init__() 
        self.input_embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = self.input_embed(x)
        return x


class OutputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(OutputEmbedding, self).__init__() 
        self.output_embed = nn.Linear(d_model, vocab_size)


    def forward(self, x):
        x = self.output_embed(x)
        return x

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
    

    def forward(self, x):
        x = self.norm1(x)
        x = nn.ReLU()(self.linear1(x))
        x = self.linear2(x)
        x = self.norm2(x)
        return x
    
    def initialize_weights(self):
        self.linear1.weight.fill_(1)
        self.linear2.weight.fill_(1)


    
    

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.q_weight = nn.Linear(d_model, d_internal)
        self.k_weight = nn.Linear(d_model, d_internal)
        self.v_weight = nn.Linear(d_model, d_model)
        self.mlp = MLP(d_model) 
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def initialize_weights(self):
        with torch.no_grad():
            ### Here we initiaize to 1, but realistically the pre-trainied models weights should be read
            self.q_weight.weight.fill_(1)
            self.k_weight.weight.fill_(1)
            self.v_weight.weight.fill_(1)
            self.mlp.initialize_weights()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_weight.to(device)
        self.k_weight.to(device)
        self.v_weight.to(device)

        
        #### Remeber to send this on the GPU
        layer_weights = {
            "weight_q": self.q_weight,
            "weight_k": self.k_weight,
            "weight_v": self.v_weight,
        }
        return layer_weights


    def forward(self, k, i, hidden, weights, cache):
        if i == 1:
            input_vecs = data[k]
        else:
            input_vecs = hidden
        ## K, Q, V calculations
        q = weights["q_weight"](input_vecs)
        k = weights["k_weight"](input_vecs)
        v = weights["v_weight"](input_vecs)

        # attention_map = torch.matmul(q, torch.transpose(k, 0, 1)) ## Make sure transpose return the correct dim
        # sm_normalized_attention_map = nn.Softmax()(attention_map / (self.d_internal ** 0.5))
        # h = torch.matmul(sm_normalized_attention_map, v)
        h = self.mha(q, k, v)
 
        ## residual connection
        x_residual = h + input_vecs

        ## Feed Forward layers``
        x = self.linear1(x_residual)
        x = nn.ReLU()(x)
        x = self.linear2(x)

        ## Add residual connection
        x = x + x_residual

        hidden = x
        return hidden, k, v



class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_internal, num_classes, num_layers, gpu_batch_size, num_gpu_batches, n_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_dim = d_model

        self.attention_maps = []
        self.input_embed = InputEmbedding(self.d_model, self.vocab_size)
        self.transformer_layers = nn.ModuleList([TransformerLayer(self.d_model, self.d_internal, n_heads) for i in range(self.num_layers)])
        self.out_linear = OutputEmbedding(self.d_model, self.vocab_size)

        
        
        # raise Exception("Implement me")

    def forward(self, indices):
        self.attention_maps = []
        x = self.input_embed(indices)
        for transformer_layer in self.transformer_layers:
            x, attention_map = transformer_layer(x)
            self.attention_maps.append(attention_map)
        x = self.linear(x)
        x = nn.Softmax()(x)

        return x, self.attention_maps
    
    def load_all_weights(self):
        all_weights = []
        for j in range(self.num_layers):
            all_weights.append(self.transformer_layers[j].initialize_weights())
            #print(f"layer {j} initialized")
        return all_weights

# class FlexGen:
#     def __init__(self):
#         self.weight_home = []
#         self.prompt_len = 10
#         self.num_prompts = 2

#     #copyied
#     def get_test_inputs(self, prompt_len, num_prompts, tokenizer):
#         prompts = ["Paris is the capital city of"]
#         input_ids = tokenizer(prompts, padding="max_length",
#                             max_length=prompt_len).input_ids
#         return (input_ids[0],) * num_prompts

#     ## Return something like (ids)*num_prompts 
#     def get_test_inputs(self, tokenizer):
#         test_inputs = self.get_test_inputs(10, 2, tokenizer)

    



def main():
    """
        Main function to configure and run the transformer model.
        This function sets up the configuration for the transformer model, initializes the model,
        loads the weights, tokenizes the input prompts, and processes the input through the transformer
        layers while managing the weights and activations.
        Configuration:
            - name: Name of the model.
            - max_seq_len: Maximum sequence length.
            - num_hidden_layers: Number of hidden layers in the transformer.
            - n_head: Number of attention heads.
            - hidden_size: Size of the hidden layers.
            - input_dim: Dimension of the input embeddings.
            - ffn_embed_dim: Dimension of the feed-forward network embeddings.
            - pad: Padding token.
            - activation_fn: Activation function to use.
            - vocab_size: Size of the vocabulary.
            - layer_norm_eps: Epsilon value for layer normalization.
            - pad_token_id: Padding token ID.
            - dtype: Data type for the model parameters.
            - num_gpu_batches: Number of GPU batches.
            - gpu_batch_size: Size of each GPU batch.
            - prompt_len: Length of the input prompts.
        Steps:
            1. Initialize the transformer model with the given configuration.
            2. Load all weights for the transformer model.
            3. Tokenize the input prompts.
            4. Create input tensors and divide them into batches.
            5. Initialize activation, key, and value storage tensors.
            6. Move weights to the CPU and process each layer and batch.
            7. Compute attention scores and activations, and store them appropriately.
        Note:
            This function assumes the existence of a `Transformer` class and an `AutoTokenizer` class
            from the `transformers` library.
    """
    opt_config = {
            "name":'opt-1.3',
            "max_seq_len": 2048,
            "num_hidden_layers":24, 
            "n_head":32,
            "hidden_size":512, #2048
            "input_dim": 512, #2048 
            "ffn_embed_dim":2048 * 4,
            "pad":  1,
            "activation_fn" : 'relu',
            "vocab_size":  50272,
            "layer_norm_eps":  0.00001,
            "pad_token_id":  1,
            "dtype": np.float16,
            "num_gpu_batches" : 8,
            "gpu_batch_size" : 16, #32
            "prompt_len": 32, #512
        }

    transformer = Transformer(
        opt_config["vocab_size"],
        opt_config["input_dim"], 
        opt_config["hidden_size"], 
        opt_config["vocab_size"], 
        opt_config["num_hidden_layers"], 
        opt_config["gpu_batch_size"], 
        opt_config["num_gpu_batches"],
        opt_config["n_head"]
        )
    all_weights = transformer.load_all_weights()
    num_prompts = opt_config["num_gpu_batches"] * opt_config["gpu_batch_size"]

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")

    def get_test_inputs(prompt_len, num_prompts, tokenizer):
        prompts = ["Paris is the capital city of"]
        input_ids = tokenizer(prompts, padding="max_length",
                            max_length=prompt_len).input_ids
        return (input_ids[0],) * num_prompts

    input = get_test_inputs(opt_config["prompt_len"], num_prompts, tokenizer)
    input = torch.tensor(input)
    batches = torch.chunk(input, chunks=8)

    """
    
    Prefill

InputEMB

For layer 1 : j 
	Load layer j in CPU
	For Batch 1 - k
		if not layer 1
			get activations of layer j-1 for batch k from GPU
			x = Activation
		Else:
			x = Output from InputEmb
		
		- K = W_k*x, Q = W_q*x, V = W_v*x  
		- Attention Score -> Softmax(K.Q).V
		- Send Attention Score to GPU
		- Append the KV values to kv_home[j][k]
		- Use attention score on GPU to calculate activation on GPU 
			- Overwrite the Activation[k]

		 
OutputEmb	
--------------

"""

    act_home = torch.empty(opt_config["num_gpu_batches"], opt_config["gpu_batch_size"], opt_config["prompt_len"], opt_config["input_dim"])
    print(act_home.shape)
    k_home = torch.empty(opt_config["num_hidden_layers"], opt_config["num_gpu_batches"], opt_config["gpu_batch_size"], opt_config["prompt_len"], opt_config["input_dim"])
    v_home = torch.empty(opt_config["num_hidden_layers"], opt_config["num_gpu_batches"], opt_config["gpu_batch_size"], opt_config["prompt_len"], opt_config["input_dim"])
    def move_weights(layer_weights, device):
        for k, v in layer_weights.items():
            if k == "weight_l1" or "weight_l2":
                continue 
            v.to(device)

    for j in range(opt_config["num_hidden_layers"]):
        move_weights(all_weights[j], "cpu")
        for k in range(opt_config["num_gpu_batches"]):
            if j != 0:
                x = act_home[k]
            else:
                x = transformer.input_embed(batches[k])
            print(j)
            K = all_weights[j]["weight_k"](x)
            Q = all_weights[j]["weight_q"](x)
            V = all_weights[j]["weight_v"](x)

            atten_map, _ = transformer.transformer_layers[j].mha(Q,K,V, need_weights=False)
            #move_weights(atten_map, "cuda: 0")

            k_home[j][k] = K
            v_home[j][k] = V

            #send atten_map, x, MLP layer to GPU for MLP stuff 
            activation = transformer.transformer_layers[j].mlp(atten_map)
            print(activation.shape)
            act_home[k] = activation



            # attention_map = torch.matmul(q, torch.transpose(k, 0, 1)) ## Make sure transpose return the correct dim
            # sm_normalized_attention_map = nn.Softmax()(attention_map / (self.d_internal ** 0.5))
            # h = torch.matmul(sm_normalized_attention_map, v)
            #atten_score = compute_mha(w_k, w_q, w_v, x)





if __name__ == "__main__":
    main()

