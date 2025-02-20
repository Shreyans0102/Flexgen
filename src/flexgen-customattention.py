
"""
This module implements a Transformer model with custom caching and weight loading mechanisms based on FlexGen.
Classes:
    ValueHolder: A class to hold a single value with methods to store, pop, and clear the value.
    Transformer: A class representing the Transformer model with methods for forward pass, weight loading, and caching.
    TransformerLayer: A class representing a single layer of the Transformer model with methods for forward pass and weight initialization.
Functions:
    offload_to_cpu(): Placeholder function for offloading computations to CPU.
    array_1d(a, cls): Creates a 1D array of instances of the given class.
    array_2d(a, b, cls): Creates a 2D array of instances of the given class.
    array_3d(a, b, c, cls): Creates a 3D array of instances of the given class.
    array_4d(a, b, c, d, cls): Creates a 4D array of instances of the given class.
    get_test_inputs(prompt_len, num_prompts, tokenizer): Generates test input IDs for the given prompt length and number of prompts using the specified tokenizer.
    main(): Main function to initialize the tokenizer, generate test inputs, create the Transformer model, load weights, and run a forward pass.
"""

import time

import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List

from transformers import AutoTokenizer


hyperparams = {"model" :"facebook/opt-6.7b",
               "path" : "_DUMMY_",
                "prompt-len" : 256,
                "gen-len" : 32,
                "percent" : [100, 0, 100, 0, 100, 0],
                "gpu-batch-size" : 4,
                "overlap" :  False
                }

opt_config = {
            "name":'opt-1.3',
            "max_seq_len": 2048,
            "num_hidden_layers":24, 
            "n_head":32,
            "hidden_size":2048, 
            "input_dim":2048, 
            "ffn_embed_dim":2048 * 4,
            "pad":  1,
            "activation_fn" : 'relu',
            "vocab_size":  50272,
            "layer_norm_eps":  0.00001,
            "pad_token_id":  1,
            "dtype": np.float16,
            "num_gpu_batches" : 8,
            "gpu_batch_size" : 32
        }

def offload_to_cpu():
    return 


class ValueHolder:
    def __init__(self):
        self.val = None

    def store(self, val):
        assert self.val is None
        self.val = val

    def pop(self):
        ret = self.val
        self.val = None
        return ret

    def clear(self):
        self.val = None


def array_1d(a, cls):
    return [cls() for _ in range(a)]


def array_2d(a, b, cls):
    return [[cls() for _ in range(b)] for _ in range(a)]


def array_3d(a, b, c, cls):
    return [[[cls() for _ in range(c)] for _ in range(b)] for _ in range(a)]


def array_4d(a, b, c, d, cls):
    return [[[[cls() for _ in range(d)] for _ in range(c)] for _ in range(b)] for _ in range(a)]



class Transformer(nn.Module):
    """
    A Transformer model for sequence-to-sequence tasks.
    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model.
        d_internal (int): Dimension of the internal layers.
        num_classes (int): Number of output classes.
        num_layers (int): Number of transformer layers.
        gpu_batch_size (int): Batch size for GPU processing.
        num_gpu_batches (int): Number of GPU batches.
    Attributes:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model.
        d_internal (int): Dimension of the internal layers.
        num_classes (int): Number of output classes.
        num_layers (int): Number of transformer layers.
        attention_maps (list): List to store attention maps.
        embed (nn.Embedding): Embedding layer.
        transformer_layers (nn.ModuleList): List of transformer layers.
        linear (nn.Linear): Linear layer for output.
        cache_home (array_2d): Cache for home.
        cache_read_buf (array_2d): Cache read buffer.
        cache_write_buf (array_2d): Cache write buffer.
        weight_home (array_1d): Weight home.
        weight_read_buf (array_1d): Weight read buffer.
        attention_mask (array_1d): Attention mask.
    Methods:
        forward(indices):
            Forward pass of the transformer model.
        main_loop(num_layers, generation_length=32, num_GPU_batches=1):
            Main loop for processing the transformer model.
        load_all_weights():
            Load all weights and initialize them.
        load_weights(i, j, k):
            Load weights for a specific layer and batch.
        store_activations(i, j, k):
            Store activations for a specific layer and batch.
        store_cache(i, j, k):
            Store cache for a specific layer and batch.
        load_cache(i, j, k):
            Load cache for a specific layer and batch.
        load_activation(i, j, k):
            Load activation for a specific layer and batch.
        compute(i, j, k):
            Compute the output for a specific layer and batch.
    """
    def __init__(self, vocab_size, d_model, d_internal, num_classes, num_layers, gpu_batch_size, num_gpu_batches):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.attention_maps = []
        self.embed = nn.Embedding(vocab_size, d_model)
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for i in range(num_layers)])
        self.linear = nn.Linear(d_model, num_classes)


        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)
        
        
        # raise Exception("Implement me")

    def forward(self, indices):
        self.attention_maps = []
        x = self.embed(indices)
        print(x.shape)
        print(x)
        for transformer_layer in self.transformer_layers:
            x, attention_map = transformer_layer(x)
            self.attention_maps.append(attention_map)
        x = self.linear(x)
        x = nn.Softmax()(x)

        return x, self.attention_maps
    

    def main_loop(self, num_layers, generation_length = 32, num_GPU_batches = 1):
        generation_length = generation_length
        num_GPU_batches = num_GPU_batches
        for i in range(generation_length):
            for j in range(num_layers):
                for k in range(num_GPU_batches):
                    self.load_weights(i, j + 1, k)

                    self.store_activations(i, j, k-1)
                    self.store_cache(i, j, k+1)

                    self.load_cache(i, k, k+1)
                    self.load_activation(i, k, k+1)

                    self.compute(i,j,k)

                    #synchronize()

    # Function to load all the weights and store it in a weight matrix
    def load_all_weights(self):
        for j in range(self.num_layers):
            self.transformer_layers[j].initialize_weights(j, self.weight_home)
            print(f"layer {j} initialized")



    def load_weights(self, i, j, k):
        return 

    def store_activations(self, i, j, k):
        return

    def store_cache(self, i, j, k):
        return

    def load_cache(self, i, j, k):
        return

    def load_activation(self, i, j, k):
        return 

    def compute(self, i, j, k):
        curr_layer = self.transformer_layers[j]
        hidden, k, v = curr_layer()
        
        return




class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.q_weight = nn.Linear(d_model, d_internal)
        self.k_weight = nn.Linear(d_model, d_internal)
        self.v_weight = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def initialize_weights(self, j, weight_home):
        with torch.no_grad():
            ### Here we initiaize to 1, but realistically the pre-trainied models weights should be read
            self.q_weight.weight.fill_(1)
            self.k_weight.weight.fill_(1)
            self.v_weight.weight.fill_(1)
            self.linear1.weight.fill_(1)
            self.linear2.weight.fill_(1)
        
        layer_weights = {
            "layer_num" : j,
            "weight_q" : self.q_weight,
            "weight_k" : self.k_weight,
            "weight_v" : self.v_weight,
            "weight_l1": self.linear1,
            "weight_l2": self.linear2 
        }
        weight_home[j].store(layer_weights)


    def forward(self, k, i, hidden, weights, cache):
        if i == 1:
            input_vecs = data[k]
        else:
            input_vecs = hidden
        ## K, Q, V calculations
        q = weights["q_weight"](input_vecs)
        k = weights["k_weight"](input_vecs)
        v = weights["v_weight"](input_vecs)

        attention_map = torch.matmul(q, torch.transpose(k, 0, 1)) ## Make sure transpose return the correct dim
        sm_normalized_attention_map = nn.Softmax()(attention_map / (self.d_internal ** 0.5))
        h = torch.matmul(sm_normalized_attention_map, v)

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



#copyied
def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts






def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")

    ## Return something like (ids)*num_prompts 
    test_inputs = get_test_inputs(10, 2, tokenizer)

    opt_config = {
            "name":'opt-1.3',
            "max_seq_len": 2048,
            "num_hidden_layers":24, 
            "n_head":32,
            "hidden_size":2048, 
            "input_dim":2048, 
            "ffn_embed_dim":2048 * 4,
            "pad":  1,
            "activation_fn" : 'relu',
            "vocab_size":  50272,
            "layer_norm_eps":  0.00001,
            "pad_token_id":  1,
            "dtype": np.float16,
            "num_gpu_batches" : 8,
            "gpu_batch_size" : 32
        }

    transformer = Transformer(
        opt_config["vocab_size"],
        opt_config["input_dim"], 
        opt_config["hidden_size"], 
        opt_config["vocab_size"], 
        opt_config["num_hidden_layers"], 
        opt_config["gpu_batch_size"], 
        opt_config["num_gpu_batches"])
    # print(transformer)
    transformer.load_all_weights()
    print(test_inputs[0])
    transformer(torch.tensor(test_inputs[0]))


if __name__ == "__main__":
    main()