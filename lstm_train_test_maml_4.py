"""lstm_train_test.py runs the LSTM baselines training/inference on forecasting dataset.

Note: The training code for these baselines is covered under the patent <PATENT_LINK>.

Example usage:
python lstm_train_test.py 
    --model_path saved_models/lstm.pth.tar 
    --test_features ../data/forecasting_data_test.pkl 
    --train_features ../data/forecasting_data_train.pkl 
    --val_features ../data/forecasting_data_val.pkl 
    --use_delta --normalize
"""

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union

import argparse
import joblib
from joblib import Parallel, delayed
import numpy as np
import pickle as pkl
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from logger import Logger
import utils.baseline_config as config
import utils.baseline_utils_maml as baseline_utils
from utils.lstm_utils_maml import ModelUtils, LSTMDataset, LSTMDataset_maml, LSTMDataset_maml_simplified
from torch.utils.data import TensorDataset

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
global_step = 0
best_loss = float("inf")
np.random.seed(100)

ROLLOUT_LENS = [1, 10, 30]


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=2048,
                        help="Test batch size")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position",
    )
    parser.add_argument(
        "--train_features",
        default="",
        type=str,
        help="path to the file which has train features.",
    )
    parser.add_argument(
        "--val_features",
        default="",
        type=str,
        help="path to the file which has val features.",
    )
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--use_map",
                        action="store_true",
                        help="Use the map based features")
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")
    parser.add_argument("--test",
                        action="store_true",
                        help="If true, only run the inference")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=512,
                        help="Training batch size")
    parser.add_argument("--val_batch_size",
                        type=int,
                        default=512,
                        help="Val batch size")
    parser.add_argument("--end_epoch",
                        type=int,
                        default=5000,
                        help="Last epoch")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument(
        "--traj_save_path",
        required=False,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )
    parser.add_argument("--maml",
                        action="store_true",
                        help="Use Meta Learning")
    parser.add_argument("--per_step_maml_optimization",
                        action="store_true",
                        help="Use MAML++")
    parser.add_argument("--num_target_samples",
                        type=int,
                        default=1,
                        help="Number of target samples in the outside MAML loop")
    parser.add_argument("--shot",
                        type=int,
                        default=0,
                        help="Number of samples in the inside MAML loop")
    parser.add_argument("--num_training_steps_per_iter",
                        type=int,
                        default=1,
                        help="Number of steps inside an iter in a MAML loop")
    parser.add_argument("--second_order",
                        type=bool,
                        default=True,
                        help="Use second order gradients for MAML")
    parser.add_argument("--minibatch_size",
                         type=int,
                         default=25,
                         help="Number of steps inside an iter in a MAML loop")
    parser.add_argument("--num_workers",
                         type=int,
                         default=8,
                         help="Number of CPU threads used for the dataloader")
    parser.add_argument("--use_lslr",
                        action="store_true",
                        help="Use LSLR learning rule")
    parser.add_argument("--use_attention",
                        action="store_true",
                        help="Use attention decoder")
    parser.add_argument("--use_learnable_lr",
                        action="store_true",
                        help="Use learnable LR during LSLR")
    parser.add_argument("--min_lr",
                         type=float,
                         default=.001,
                         help="Number of CPU threads used for the dataloader")
    parser.add_argument("--num_layers",
                         type=int,
                         default=1,
                         help="Number of CPU threads used for the dataloader")

    return parser.parse_args()


class MetaLinearLayer(nn.Module):
    def __init__(self, input_size, num_filters, use_bias):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()
        c = input_size

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_filters, c), requires_grad=True)
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters), requires_grad=True)

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            #params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            pass
            #print('no inner loop params', self)

            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        # print(x.shape)
        return F.linear(input=x, weight=weight, bias=bias)

class MetaLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(MetaLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = MetaLinearLayer(input_size, 4 * hidden_size, use_bias=bias)
        self.h2h = MetaLinearLayer(hidden_size,  4 * hidden_size, use_bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden, params=None):
        if hidden is None:
            hidden = self._init_hidden(x)
        h, c = hidden
        # h = h.view(h.size(1), -1)
        # c = c.view(c.size(1), -1)
        # x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x, (None if (params == None) else params['i2h'])) + \
                    self.h2h(h, (None if (params == None) else params['h2h']))

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)

        h_t = torch.mul(o_t, c_t.tanh())

        # h_t = h_t.view(1, h_t.size(0), -1)
        # c_t = c_t.view(1, c_t.size(0), -1)
        # return h_t, (h_t, c_t)
        return (h_t, c_t)
    
    @staticmethod
    def _init_hidden(input_):
        h = torch.zeros_like(input_.view(1, input_.size(1), -1))
        c = torch.zeros_like(input_.view(1, input_.size(1), -1))
        return h, c

def create_init_params(num_layers, batch_size, hidden_size, model_utils):
    init_params = dict()
    for i in range(1, num_layers+1):
        init_params['lstm{}'.format(i)] = model_utils.init_hidden(batch_size,hidden_size)
    return init_params

class MetaEncoderRNN(nn.Module):
    """Encoder Network."""
    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 16,
                 num_layers: int = 1
                 ):
        """Initialize the encoder network.

        Args:
            input_size: number of features in the input
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM

        """
        super(MetaEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.enclinear1 = MetaLinearLayer(input_size, embedding_size, use_bias=True)
        self.enclstms = nn.ModuleDict()
        self.enclstms['lstm1'] = MetaLSTMCell(embedding_size, hidden_size)

        for i in range(1, self.num_layers):
            self.enclstms['lstm{}'.format(i+1)] = MetaLSTMCell(hidden_size, hidden_size)

        #self.lstm1 = MetaLSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden_in, param=None):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 

        """
        param_dict = None if param == None else self.preprocess_param_dict(param)
        #embedded = F.relu(self.enclinear1(x, (None if param == None else param_dict['enclinear1'])))
        #hidden = self.lstm['lstm1'](embedded, hidden, (None if param == None else param_dict['lstm1']))
        embedded = F.leaky_relu(self.enclinear1(x, (None if param == None else param_dict['enclinear1'])))
        hidden_out = dict()
        #import pdb; pdb.set_trace();
        hidden_out['lstm1'] = self.enclstms['lstm1'](embedded, hidden_in['lstm1'], (None if param == None else param_dict['lstm1']))

        for i in range(1, self.num_layers):
            key = 'lstm{}'.format(i+1)
            prev_key = 'lstm{}'.format(i)
            hidden_out[key] = self.enclstms[key](hidden_out[prev_key][0], hidden_in[key], (None if param == None else param_dict[key]))

        output = hidden_out['lstm{}'.format(self.num_layers)][0]
        
        return hidden_out, output
    
    def preprocess_param_dict(self, param_dict):
        reordered_dict = {}
        reordered_dict['enclinear1'] = {}
        for i in range(1, self.num_layers+1):
            reordered_dict['lstm{}'.format(i)] = {}
            reordered_dict['lstm{}'.format(i)]['i2h'] ={}
            reordered_dict['lstm{}'.format(i)]['h2h'] ={}
        for name, param in param_dict.items():
            key = (name.replace('module.', '')) if 'module.' in name else name
            names_split = key.split('.')
            if names_split[0] == 'enclinear1':
                reordered_dict['enclinear1'][names_split[1]] = param
            elif names_split[0] == 'enclstms':
                reordered_dict[names_split[1]][names_split[2]][names_split[3]] = param
        return reordered_dict

class MetaDecoderRNN(nn.Module):
    """Encoder Network."""
    def __init__(self,
                 embedding_size=8,
                 hidden_size=16,
                 output_size=2,
                 num_layers = 1,
                 ):

        """Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output
        """
        super(MetaDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers 

        self.declinear1 = MetaLinearLayer(output_size, embedding_size, use_bias=True)
        #self.lstm1 = MetaLSTMCell(embedding_size, hidden_size)
        self.declstms = nn.ModuleDict()
        self.declstms['lstm1'] = MetaLSTMCell(embedding_size, hidden_size)

        for i in range(1, self.num_layers):
            self.declstms['lstm{}'.format(i+1)] = MetaLSTMCell(hidden_size, hidden_size)

        self.declinear2 = MetaLinearLayer(hidden_size, output_size, use_bias=True)

    def forward(self, x: torch.FloatTensor, hidden_in, param=None):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 

        """
        #import pdb; pdb.set_trace()
        param_dict = None if param == None else self.preprocess_param_dict(param)
        embedded = F.leaky_relu(self.declinear1(x, (None if param == None else param_dict['declinear1'])))

        #hidden = self.lstm1(embedded, hidden, (None if param == None else param_dict['lstm1']))
        hidden_out = dict()
        hidden_out['lstm1'] = self.declstms['lstm1'](embedded, hidden_in['lstm1'], (None if param == None else param_dict['lstm1']))

        for i in range(1, self.num_layers):
            key = 'lstm{}'.format(i+1)
            prev_key = 'lstm{}'.format(i)
            hidden_out[key] = self.declstms[key](hidden_out[prev_key][0], hidden_in[key], (None if param == None else param_dict[key]))

        output = self.declinear2(hidden_out['lstm{}'.format(self.num_layers)][0], (None if param == None else param_dict['declinear2']))
        return output, hidden_out
    
    def preprocess_param_dict(self, param_dict):
        reordered_dict = {}
        reordered_dict['declinear1'] = {}
        reordered_dict['declinear2'] = {}
        for i in range(1, self.num_layers+1):
            reordered_dict['lstm{}'.format(i)] = {}
            reordered_dict['lstm{}'.format(i)]['i2h'] ={}
            reordered_dict['lstm{}'.format(i)]['h2h'] ={}
        for name, param in param_dict.items():
            key = (name.replace('module.', '')) if 'module.' in name else name
            names_split = key.split('.')
            if names_split[0] == 'declinear1':
                reordered_dict['declinear1'][names_split[1]] = param
            elif names_split[0] == 'declstms':
                reordered_dict[names_split[1]][names_split[2]][names_split[3]] = param
            elif names_split[0] == 'declinear2':
                reordered_dict['declinear2'][names_split[1]] = param
        return reordered_dict

class MetaAttDecoderRNN(nn.Module):
    """Encoder Network."""
    def __init__(self,
                 embedding_size=8,
                 hidden_size=16,
                 output_size=2,
                 num_layers = 1,
                 obs_len= 20,
                 ):

        """Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output
        """
        super(MetaAttDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers 

        self.declinear1 = MetaLinearLayer(output_size, embedding_size, use_bias=True)
        #self.lstm1 = MetaLSTMCell(embedding_size, hidden_size)
        self.declstms = nn.ModuleDict()
        self.declstms['lstm1'] = MetaLSTMCell(embedding_size, hidden_size)

        for i in range(1, self.num_layers):
            self.declstms['lstm{}'.format(i+1)] = MetaLSTMCell(hidden_size, hidden_size)

        self.declinear2 = MetaLinearLayer(hidden_size, output_size, use_bias=True)

        self.attn = MetaLinearLayer(embedding_size + (hidden_size * num_layers), obs_len, use_bias=True) 
        self.attn_combine = MetaLinearLayer(hidden_size + embedding_size , embedding_size, use_bias=True)

    def forward(self, x: torch.FloatTensor, hidden_in, encoder_outputs, param=None):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
            param: meta params
            encoder_outputs: all previous encoder output hidden states
        Returns:
            hidden: final hidden 

        """
        #import pdb; pdb.set_trace()
        param_dict = None if param == None else self.preprocess_param_dict(param)
        embedded = F.leaky_relu(self.declinear1(x, (None if param == None else param_dict['declinear1'])))

        # Attention stuff:
        attn_input = embedded
        
        #import pdb; pdb.set_trace()
        for i in range(1, self.num_layers+1):
            key = 'lstm{}'.format(i)
            # We only want to concatenate hidden, not c
            attn_input = torch.cat((attn_input, hidden_in[key][0]), dim=1)

        attn_weights = torch.nn.functional.softmax(self.attn(attn_input, (None if param == None else param_dict['attn'])), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        attn_combined_input = torch.cat((attn_applied.squeeze(1), embedded), dim=1) 
        attn_combined_output = torch.nn.functional.relu(self.attn_combine(attn_combined_input, (None if param == None else param_dict['attn_combine'])))

        # end of attention stuff
        
        #hidden = self.lstm1(embedded, hidden, (None if param == None else param_dict['lstm1']))
        hidden_out = dict()
        hidden_out['lstm1'] = self.declstms['lstm1'](attn_combined_output, hidden_in['lstm1'], (None if param == None else param_dict['lstm1']))
        
        for i in range(1, self.num_layers):
            key = 'lstm{}'.format(i+1)
            prev_key = 'lstm{}'.format(i)
            hidden_out[key] = self.declstms[key](hidden_out[prev_key][0], hidden_in[key], (None if param == None else param_dict[key]))

        output = self.declinear2(hidden_out['lstm{}'.format(self.num_layers)][0], (None if param == None else param_dict['declinear2']))
        return output, hidden_out
    
    def preprocess_param_dict(self, param_dict):
        reordered_dict = {}
        reordered_dict['declinear1'] = {}
        reordered_dict['declinear2'] = {}
        reordered_dict['attn'] = {}
        reordered_dict['attn_combine'] = {}
        for i in range(1, self.num_layers+1):
            reordered_dict['lstm{}'.format(i)] = {}
            reordered_dict['lstm{}'.format(i)]['i2h'] ={}
            reordered_dict['lstm{}'.format(i)]['h2h'] ={}
        for name, param in param_dict.items():
            key = (name.replace('module.', '')) if 'module.' in name else name
            names_split = key.split('.')
            if names_split[0] == 'declinear1':
                reordered_dict['declinear1'][names_split[1]] = param
            elif names_split[0] == 'declstms':
                reordered_dict[names_split[1]][names_split[2]][names_split[3]] = param
            elif names_split[0] == 'declinear2':
                reordered_dict['declinear2'][names_split[1]] = param
            elif names_split[0] == 'attn':
                reordered_dict['attn'][names_split[1]] = param
            elif names_split[0] == 'attn_combine':
                reordered_dict['attn_combine'][names_split[1]] = param

        return reordered_dict

class EncoderRNN(nn.Module):
    """Encoder Network."""
    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 16):
        """Initialize the encoder network.

        Args:
            input_size: number of features in the input
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM

        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        return hidden


class DecoderRNN(nn.Module):
    """Decoder Network."""
    def __init__(self, embedding_size=8, hidden_size=16, output_size=2):
        """Initialize the decoder network.

        Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output

        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            output: output from lstm
            hidden: final hidden state

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden

def plot_grad_flow(encoder_parameters, decoder_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers_enc = []
    for n, p in encoder_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers_enc.append('encoder.'+n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    for n, p in decoder_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers_enc.append('decoder.'+n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers_enc, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])



def train(
        train_loader: Any,
        epoch: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        decoder: Any,
        encoder_optimizer: Any,
        decoder_optimizer: Any,
        model_utils: ModelUtils,
        rollout_len: int = 30,
) -> None:
    """Train the lstm network.

    Args:
        train_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        rollout_len: current prediction horizon

    """
    args = parse_arguments()
    global global_step

    for i, (_input, target, helpers) in enumerate(train_loader):
        _input = _input.to(device)
        target = target.to(device)

        # Set to train mode
        encoder.train()
        decoder.train()

        # Zero the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        output_length = target.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Initialize losses
        loss = 0

        # Encode observed trajectory
        #import pdb; pdb.set_trace();
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(target.shape).to(device)

        # Decode hidden state in future trajectory
        for di in range(rollout_len):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Update loss
            loss += criterion(decoder_output[:, :2], target[:, di, :2])

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get average loss for pred_len
        loss = loss / rollout_len
        # Backpropagate
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        if global_step % 1000 == 0:

            # Log results
            print(
                f"Train -- Epoch:{epoch}, loss:{loss}, Rollout:{rollout_len}")

            logger.scalar_summary(tag="Train/loss",
                                  value=loss.item(),
                                  step=epoch)

        global_step += 1

def lstm_forward(
    num_layers,
    encoder: Any,
    decoder: Any,
    encoder_params: Any,
    decoder_params: Any,
    input_seq: Any,
    target_seq: Any,
    obs_len: int,
    pred_len: int,
    criterion: Any,
    model_utils: ModelUtils,
    use_attention: bool,
):
    batch_size =  input_seq.shape[0]
    # Initialize losses
    loss = 0

    encoder_hidden = create_init_params(num_layers,
                                        batch_size,
                                        encoder.module.hidden_size if use_cuda else encoder.hidden_size,
                                        model_utils,)
    if use_attention:
        # For attention
        # Output from encoder is batch_size x hidden_size
        encoder_outputs = torch.zeros(batch_size, obs_len, encoder.module.hidden_size if use_cuda else encoder.hidden_size, device=device)

    # Encode observed trajectory
    for ei in range(obs_len):
        encoder_input = input_seq[:, ei, :]
        encoder_hidden, encoder_out = encoder(encoder_input, encoder_hidden, encoder_params)
        if use_attention:
            encoder_outputs[:,ei,:] = encoder_out

    # Initialize decoder input with last coordinate in encoder
    decoder_input = encoder_input[:, :2]

    # Initialize decoder hidden state as encoder hidden state
    decoder_hidden = encoder_hidden

    decoder_outputs = torch.zeros(target_seq.shape).to(device)

    # Decode hidden state in future trajectory
    for di in range(pred_len):
        if use_attention:
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                 decoder_hidden, encoder_outputs, decoder_params)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                 decoder_hidden, decoder_params)
        decoder_outputs[:, di, :] = decoder_output

        # Update loss
        loss += criterion(decoder_output[:, :2], target_seq[:, di, :2])

        # Use own predictions as inputs at next step
        decoder_input = decoder_output

    # Get average loss for pred_len
    loss = loss / pred_len
    
    return loss, decoder_outputs

def lstm_infer_forward(
    num_layers,
    encoder: Any,
    decoder: Any,
    encoder_params: Any,
    decoder_params: Any,
    input_seq: Any,
    target_seq: Any,
    obs_len: int,
    pred_len: int,
    model_utils: ModelUtils,
    use_attention: bool,
):
    batch_size =  input_seq.shape[0]
    # Initialize losses

    encoder_hidden = create_init_params(num_layers,
                                        batch_size,
                                        encoder.module.hidden_size if use_cuda else encoder.hidden_size,
                                        model_utils,)

    if use_attention:
        # For attention
        # Output from encoder is batch_size x hidden_size
        encoder_outputs = torch.zeros(batch_size, obs_len, encoder.module.hidden_size if use_cuda else encoder.hidden_size, device=device)

    #import pdb; pdb.set_trace()
    # Encode observed trajectory
    for ei in range(obs_len):
        encoder_input = input_seq[:, ei, :]
        encoder_hidden, encoder_out = encoder(encoder_input, encoder_hidden, encoder_params)
        if use_attention:
            encoder_outputs[:,ei,:] = encoder_out

    # Initialize decoder input with last coordinate in encoder
    decoder_input = encoder_input[:, :2]

    # Initialize decoder hidden state as encoder hidden state
    decoder_hidden = encoder_hidden

    decoder_outputs = torch.zeros(target_seq.shape).to(device)

    # Decode hidden state in future trajectory
    for di in range(pred_len):
        if use_attention:
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                 decoder_hidden, encoder_outputs, decoder_params)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                 decoder_hidden, decoder_params)
        decoder_outputs[:, di, :] = decoder_output

        # Use own predictions as inputs at next step
        decoder_input = decoder_output

    return decoder_outputs

def get_named_params_dicts(
    model: Any,
):
    params_dict = dict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            #key = (name.replace('module.', '')) if 'module.' in name else name
            params_dict[name] = param.to(device)
    
    return params_dict

def get_raw_named_params_dicts(
    model: Any,
):
    params_dict = dict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_dict[name] = param.to(device)
    
    return params_dict

def update_params(
    param_dict,
    grad_dict,
    lr_dict = None,
):
    args = parse_arguments()
    updated_weights_dict = dict()
    for key in grad_dict.keys():
        updated_weights_dict[key] = param_dict[key] - (args.lr if lr_dict == None else lr_dict[key.replace('.', '-')]) * grad_dict[key]

    return updated_weights_dict

def update_state_params(
    state_dict,
    param_dict,
    grad_dict,
):
    args = parse_arguments()
    state_dict_copy = copy.deepcopy(state_dict)
    for key in grad_dict.keys():
        state_dict_copy[key] = param_dict[key] - args.lr * grad_dict[key]

    return state_dict_copy

def maml_inner_loop_update(
    loss,
    encoder,
    decoder,
    encoder_params,
    decoder_params,
    use_second_order,
    encoder_learning_rule = None,
    decoder_learning_rule = None,
    step = 0,
    mamlpp = False,
    encoder_lr_dict = None,
    decoder_lr_dict = None,
):
    #import pdb; pdb.set_trace();
    zero_grad(encoder, encoder_params)
    zero_grad(decoder, decoder_params)

    if mamlpp:
        encoder_grads = torch.autograd.grad(loss, encoder_params.values(), retain_graph = True)
        decoder_grads = torch.autograd.grad(loss, decoder_params.values(), retain_graph = True)
        for grad in encoder_grads:
            grad = grad.detach()
        for grad in decoder_grads:
            grad = grad.detach()
    else:
        encoder_grads = torch.autograd.grad(loss, encoder_params.values(), create_graph=use_second_order, retain_graph=True)
        decoder_grads = torch.autograd.grad(loss, decoder_params.values(), create_graph=use_second_order)

    encoder_grads_wrt_param_names = dict(zip(encoder_params.keys(), encoder_grads))
    decoder_grads_wrt_param_names = dict(zip(decoder_params.keys(), decoder_grads))

    if encoder_learning_rule == None:
        encoder_params = update_params(encoder_params, encoder_grads_wrt_param_names)
    else:
        encoder_params = encoder_learning_rule.update_params(encoder_params, encoder_grads_wrt_param_names, step, encoder_lr_dict)
        
    if decoder_learning_rule == None:
        decoder_params = update_params(decoder_params, decoder_grads_wrt_param_names) 
    else:
        decoder_params = decoder_learning_rule.update_params(decoder_params, decoder_grads_wrt_param_names, step, decoder_lr_dict)


    return encoder_params, decoder_params

def update_model_grads(
    model,
    grad_dict
):
    for name, param in model.named_parameters():
       for name_, grad in grad_dict:
           if name == name_ :
               param.grad = grad

def clamp_grads(model):
    for name, param in model.named_parameters():
         if param.requires_grad:
             param.grad.data.clamp_(-10, 10)  

def zero_grad(model, params=None):
    if params is None:
        for param in model.parameters():
            if param.requires_grad == True:
                if param.grad is not None:
                    if torch.sum(param.grad) > 0:
                        #print(param.grad)
                        param.grad.zero_()
    else:
        for name, param in params.items():
            if param.requires_grad == True:
                if param.grad is not None:
                    if torch.sum(param.grad) > 0:
                        #print(param.grad)
                        param.grad.zero_()
                        params[name].grad = None

def get_per_step_loss_importance_vector(args, current_epoch):
    """
    Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
    loss towards the optimization loss.
    :return: A tensor to be used to compute the weighted average of the loss, useful for
    the MSL (Multi Step Loss) mechanism.
    """
    loss_weights = np.ones(shape=(args.num_training_steps_per_iter)) * (
            1.0 / args.num_training_steps_per_iter)
    decay_rate = 1.0 / args.num_training_steps_per_iter / args.end_epoch
    min_value_for_non_final_losses = 0.03 / args.num_training_steps_per_iter
    for i in range(len(loss_weights) - 1):
        loss_weights[i] = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)

    curr_value = np.minimum(
        loss_weights[-1] + (current_epoch * (args.num_training_steps_per_iter - 1) * decay_rate),
        1.0 - ((args.num_training_steps_per_iter - 1) * min_value_for_non_final_losses))
    loss_weights[-1] = curr_value
    loss_weights = torch.Tensor(loss_weights).to(device=device)
    return loss_weights

def maml_infer_forward(
        args : Any,
        data_batch: Any,
        epoch: int,
        criterion: Any,
        encoder: Any,
        decoder: Any,
        model_utils: ModelUtils,
        encoder_learning_rules = None,
        decoder_learning_rules = None,
):

    rollout_len = args.pred_len
    (support_input_seqs, support_obs_seqs, _, _, _) = data_batch

        # Copy the model for MAML inner loop
    shot = args.shot if args.shot <= args.train_batch_size else args.train_batch_size
    support_input_seq = support_input_seqs[:shot, :, :, :].squeeze(dim=1).to(device)
    support_obs_seq = support_obs_seqs[:shot, :, :, :].squeeze(dim=1).to(device)

    encoder_copy_params = get_named_params_dicts(encoder)
    decoder_copy_params = get_named_params_dicts(decoder)

    encoder_learning_rule, encoder_lr_dict = (None, None) if encoder_learning_rules is None else encoder_learning_rules
    decoder_learning_rule, decoder_lr_dict = (None, None) if decoder_learning_rules is None else decoder_learning_rules

    for iter_ in range(args.num_training_steps_per_iter):
        support_loss, supprt_pred = lstm_forward(
            args.num_layers,
            encoder,
            decoder,
            encoder_copy_params,
            decoder_copy_params,
            support_input_seq,
            support_obs_seq,
            args.obs_len,
            rollout_len,
            criterion,
            model_utils,
            use_attention=args.use_attention,
        )

        encoder_copy_params, decoder_copy_params = maml_inner_loop_update(
            support_loss, 
            encoder, decoder,
            encoder_copy_params, decoder_copy_params,
            False,
            encoder_learning_rule, decoder_learning_rule,
            iter_,
            args.per_step_maml_optimization,
            encoder_lr_dict, decoder_lr_dict,
        )

    return encoder_copy_params, decoder_copy_params

def maml_forward(
        args : Any,
        data_batch: Any,
        epoch: int,
        criterion: Any,
        encoder: Any,
        decoder: Any,
        model_utils: ModelUtils,
        per_step_loss_importance_vecor,
        second_order = False,
        rollout_len: int = 30,
        encoder_learning_rule = None,
        decoder_learning_rule = None,
):

    #import pdb; pdb.set_trace();
    (support_input_seqs, support_obs_seqs, train_input_seq, train_obs_seq, _) = data_batch

    train_input_seq = train_input_seq.squeeze(1).to(device)
    train_obs_seq = train_obs_seq.squeeze(1).to(device)

        # Copy the model for MAML inner loop
    shot = args.shot if args.shot <= args.train_batch_size else args.train_batch_size
    support_input_seq = support_input_seqs[:shot, :, :, :].squeeze(dim=1).to(device)
    support_obs_seq = support_obs_seqs[:shot, :, :, :].squeeze(dim=1).to(device)

    encoder_copy_params = get_named_params_dicts(encoder)
    decoder_copy_params = get_named_params_dicts(decoder)

    train_loss = None
    encoder_dict = get_named_params_dicts(encoder)
    decoder_dict = get_named_params_dicts(decoder)

    train_preds = None
    total_losses = []
    for iter_ in range(args.num_training_steps_per_iter):
        support_loss, supprt_pred = lstm_forward(
            args.num_layers,
            encoder,
            decoder,
            encoder_copy_params,
            decoder_copy_params,
            support_input_seq,
            support_obs_seq,
            args.obs_len,
            rollout_len,
            criterion,
            model_utils,
            use_attention=args.use_attention
        )

        total_losses.append(per_step_loss_importance_vecor[iter_] * support_loss)

        encoder_copy_params, decoder_copy_params = maml_inner_loop_update(
            support_loss, 
            encoder, decoder,
            encoder_copy_params, decoder_copy_params,
            args.second_order,
            encoder_learning_rule, decoder_learning_rule,
            iter_,
            args.per_step_maml_optimization,
        )

    train_loss, train_preds = lstm_forward(
        args.num_layers,
        encoder,
        decoder,
        encoder_copy_params,
        decoder_copy_params,
        train_input_seq,
        train_obs_seq,
        args.obs_len,
        rollout_len,
        criterion,
        model_utils,
        args.use_attention,
    ) 
    total_losses.append(train_loss)

    #loss = train_loss
    loss = torch.sum(torch.stack(total_losses)).to(device)
    if args.test is False:
        return loss
    else:
        return loss, train_preds

def train_maml_oversimplified(
        train_loader: Any,
        epoch: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        decoder: Any,
        encoder_optimizers: Any,
        decoder_optimizers: Any,
        model_utils: ModelUtils,
        loader_len,
        rollout_len: int = 30,
        encoder_learning_rule = None,
        decoder_learning_rule = None,
) -> None:
    """Train the lstm network.

    Args:
        train_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        rollout_len: current prediction horizon

    """
    args = parse_arguments()
    global global_step
    #import pdb; pdb.set_trace();
    (encoder_optimizer, encoder_scheduler) = encoder_optimizers
    (decoder_optimizer, decoder_scheduler) = decoder_optimizers
    
    per_step_loss_importance_vecor = get_per_step_loss_importance_vector(args, epoch) 
            
    with tqdm(total=loader_len, desc='Epoch: {}'.format(epoch), position=0) as pbar:
        for i, data_batch in enumerate(train_loader):
            encoder.train()
            decoder.train()
            zero_grad(encoder)
            zero_grad(decoder)
            pbar.update(1)

            loss = maml_forward(
                args,
                data_batch,
                epoch,
                criterion,
                encoder,
                decoder,
                model_utils,
                per_step_loss_importance_vecor,
                args.second_order,
                rollout_len,
                encoder_learning_rule,
                decoder_learning_rule,
            )
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Backpropagate
            loss.backward()
            #plot_grad_flow(encoder.named_parameters(), decoder.named_parameters())

            clamp_grads(encoder)
            clamp_grads(decoder)

            encoder_optimizer.step()
            decoder_optimizer.step()

            if encoder_scheduler is not None:
                encoder_scheduler.step(epoch=epoch)
            if decoder_scheduler is not None:
                decoder_scheduler.step(epoch=epoch)

            if i % 100 == 0:
                pbar.write(f"Train -- Optimizer loop:{i} Epoch:{epoch}, avg loss:{loss.detach().item()}, Rollout:{rollout_len}")
            if global_step % 1000 == 0:

                logger.scalar_summary(tag="Train/loss",
                                      value=loss.detach().item(),
                                      step=epoch)

            global_step += 1

def validate_maml(
        val_loader: Any,
        epoch: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        decoder: Any,
        encoder_optimizers: Any,
        decoder_optimizers: Any,
        model_utils: ModelUtils,
        loader_len,
        prev_loss: float,
        decrement_counter: int,
        rollout_len: int = 30,
        encoder_learning_rule = None,
        decoder_learning_rule = None,
) -> Tuple[float, int]:
    """Validate the lstm network.

    Args:
        val_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        prev_loss: Loss in the previous validation run
        decrement_counter: keeping track of the number of consecutive times loss increased in the current rollout
        rollout_len: current prediction horizon

    """
    args = parse_arguments()
    global best_loss
    total_loss = []

    per_step_loss_importance_vecor = get_per_step_loss_importance_vector(args, epoch)

    encoder_optimizer, encoder_scheduler = encoder_optimizers
    decoder_optimizer, decoder_scheduler = decoder_optimizers

    with tqdm(total=loader_len, desc='Epoch: {}'.format(epoch), position=0) as pbar:
        for i, data_batch in enumerate(val_loader):
            encoder.eval()
            decoder.eval()
            pbar.update(1)

            loss = 0
            
            loss = maml_forward(
                args = args,
                data_batch = data_batch,
                epoch = epoch,
                criterion = criterion,
                encoder = encoder,
                decoder = decoder,
                model_utils = model_utils,
                per_step_loss_importance_vecor = per_step_loss_importance_vecor,
                second_order = False,
                rollout_len = rollout_len,
                encoder_learning_rule = encoder_learning_rule,
                decoder_learning_rule = decoder_learning_rule,
            )

            total_loss.append(loss.detach().item())

            if i % 10 == 0:

                pbar.write(
                    f"Val -- Epoch:{epoch}, loss:{loss.detach().item()}, Rollout: {rollout_len}",
                )

    # Save
    val_loss = sum(total_loss) / len(total_loss)
    pbar.write(
        f"Val -- Epoch:{epoch}, avg loss:{val_loss}, Rollout: {rollout_len}",
    )

    if val_loss <= best_loss:
        pbar.write(
            f"Val -- Epoch:{epoch}, New best loss: {val_loss}",
        )
        best_loss = val_loss
        if args.use_map:
            save_dir = "saved_models/lstm_map/best_val"
        elif args.use_social:
            save_dir = "saved_models/lstm_social/best_val"
        else:
            save_dir = "saved_models/lstm/best_val"

        os.makedirs(save_dir, exist_ok=True)
        model_utils.save_checkpoint(
            save_dir,
            {
                "epoch": epoch + 1,
                "rollout_len": rollout_len,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "best_loss": val_loss,
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                #"encoder_learning_rate": encoder_scheduler.get_lr(),
                #"decoder_learning_rate": decoder_scheduler.get_lr(),
                "encoder_learning_rule_dict":encoder_learning_rule.get_lr_dict(),
                "decoder_learning_rule_dict":decoder_learning_rule.get_lr_dict(),
            },
        )

    # Save all models too for good measure
    if args.use_map:
        save_dir = "saved_models/lstm_map/all"
    elif args.use_social:
        save_dir = "saved_models/lstm_social/all"
    else:
        save_dir = "saved_models/lstm/all"

    os.makedirs(save_dir, exist_ok=True)
    model_utils.save_checkpoint(
        save_dir,
        {
            "epoch": epoch + 1,
            "rollout_len": rollout_len,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "best_loss": val_loss,
            "encoder_optimizer": encoder_optimizer.state_dict(),
            "decoder_optimizer": decoder_optimizer.state_dict(),
            #"encoder_learning_rate": encoder_scheduler.get_lr(),
            #"decoder_learning_rate": decoder_scheduler.get_lr(),
            "encoder_learning_rule_dict":encoder_learning_rule.get_lr_dict(),
            "decoder_learning_rule_dict":decoder_learning_rule.get_lr_dict(),
        },
    )

    logger.scalar_summary(tag="Val/loss", value=val_loss, step=epoch)

    # Keep track of the loss to change preiction horizon
    if val_loss <= prev_loss:
        decrement_counter = 0
    else:
        decrement_counter += 1

    return val_loss, decrement_counter

def validate(
        val_loader: Any,
        epoch: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        decoder: Any,
        encoder_optimizer: Any,
        decoder_optimizer: Any,
        model_utils: ModelUtils,
        prev_loss: float,
        decrement_counter: int,
        rollout_len: int = 30,
) -> Tuple[float, int]:
    """Validate the lstm network.

    Args:
        val_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        prev_loss: Loss in the previous validation run
        decrement_counter: keeping track of the number of consecutive times loss increased in the current rollout
        rollout_len: current prediction horizon

    """
    args = parse_arguments()
    global best_loss
    total_loss = []

    for i, (_input, target, helpers) in enumerate(val_loader):

        _input = _input.to(device)
        target = target.to(device)

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        output_length = target.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Initialize loss
        loss = 0

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(target.shape).to(device)

        # Decode hidden state in future trajectory
        for di in range(output_length):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Update losses for all benchmarks
            loss += criterion(decoder_output[:, :2], target[:, di, :2])

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get average loss for pred_len
        loss = loss / output_length
        total_loss.append(loss)

        if i % 10 == 0:

            cprint(
                f"Val -- Epoch:{epoch}, loss:{loss}, Rollout: {rollout_len}",
                color="green",
            )

    # Save
    val_loss = sum(total_loss) / len(total_loss)

    if val_loss <= best_loss:
        best_loss = val_loss
        if args.use_map:
            save_dir = "saved_models/lstm_map"
        elif args.use_social:
            save_dir = "saved_models/lstm_social"
        else:
            save_dir = "saved_models/lstm"

        os.makedirs(save_dir, exist_ok=True)
        model_utils.save_checkpoint(
            save_dir,
            {
                "epoch": epoch + 1,
                "rollout_len": rollout_len,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "best_loss": val_loss,
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
            },
        )

    logger.scalar_summary(tag="Val/loss", value=val_loss.item(), step=epoch)

    # Keep track of the loss to change preiction horizon
    if val_loss <= prev_loss:
        decrement_counter = 0
    else:
        decrement_counter += 1

    return val_loss, decrement_counter


def infer_absolute(
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
):
    """Infer function for non-map LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance

    """
    args = parse_arguments()
    forecasted_trajectories = {}

    for i, (_input, target, helpers) in enumerate(test_loader):

        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(
            (batch_size, args.pred_len, 2)).to(device)

        # Decode hidden state in future trajectory
        for di in range(args.pred_len):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get absolute trajectory
        abs_helpers = {}
        abs_helpers["REFERENCE"] = np.array(helpers_dict["DELTA_REFERENCE"])
        abs_helpers["TRANSLATION"] = np.array(helpers_dict["TRANSLATION"])
        abs_helpers["ROTATION"] = np.array(helpers_dict["ROTATION"])
        abs_inputs, abs_outputs = baseline_utils.get_abs_traj(
            _input.clone().cpu().numpy(),
            decoder_outputs.detach().clone().cpu().numpy(),
            args,
            abs_helpers,
        )

        for i in range(abs_outputs.shape[0]):
            seq_id = int(helpers_dict["SEQ_PATHS"][i])
            forecasted_trajectories[seq_id] = [abs_outputs[i]]

    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)

def infer_maml_abs_simplified(
        test_loader: Any,
        support_loader: Any,
        encoder: Any,
        decoder: Any,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
        epoch: int,
        loader_len : int,
        encoder_learning_rules = None, 
        decoder_learning_rules = None,

):
    """Infer function for map-based LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance
        epoch: Epoch at which we ended training at
    """
    
    forecasted_trajectories = {}
    args = parse_arguments()
    criterion = nn.MSELoss()
    total_loss = []

    for i, (support_batch, data_batch) in enumerate(zip(support_loader, test_loader)):

        encoder.eval()
        decoder.eval()

        #import pdb; pdb.set_trace()
        encoder_parameters, decoder_parameters = maml_infer_forward(
                                                   args = args,
                                                   data_batch = support_batch,
                                                   epoch = epoch,
                                                   criterion = criterion,
                                                   encoder = encoder,
                                                   decoder = decoder,
                                                   model_utils = model_utils,
                                                   encoder_learning_rules = encoder_learning_rules,
                                                   decoder_learning_rules = decoder_learning_rules,
                                                 )

        (_, _, _, _, helpers) = data_batch
        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]
    
        batch_size = data_batch[2].shape[0]
       
        with tqdm(total=batch_size, desc='Iterating over batch', position=1) as pbar2:
            for batch_idx in range(batch_size):
                pbar2.update(1)

                num_candidates = len(helpers_dict["CANDIDATE_CENTERLINES"][batch_idx])
                curr_centroids = helpers_dict["CENTROIDS"][batch_idx]
                seq_id = int(helpers_dict["SEQ_PATHS"][batch_idx])
                abs_outputs = []

                # Predict using every centerline candidate for the current trajectory
                for candidate_idx in range(num_candidates):
                    curr_centerline = helpers_dict["CANDIDATE_CENTERLINES"][
                        batch_idx][candidate_idx]
                    curr_nt_dist = helpers_dict["CANDIDATE_NT_DISTANCES"][
                        batch_idx][candidate_idx]

                    # Since this is test set all our inputs are gonna be None, gotta build
                    # them ourselves.
                    test_input_seq = torch.FloatTensor(
                                     np.expand_dims(curr_nt_dist[:args.obs_len].astype(float),
                                     0)).to(device)

                    test_target_seq = torch.zeros(test_input_seq.shape[0], 30, 2)

                    preds = lstm_infer_forward(
                              num_layers = args.num_layers,
                              encoder = encoder,
                              decoder = decoder,
                              encoder_params = encoder_parameters,
                              decoder_params = decoder_parameters,
                              input_seq = test_input_seq,
                              target_seq = test_target_seq,
                              obs_len = args.obs_len,
                              pred_len = args.pred_len,
                              model_utils = model_utils,
                              use_attention=args.use_attention,
                            )

                    abs_helpers = {}
                    abs_helpers["REFERENCE"] = np.expand_dims(
                        np.array(helpers_dict["CANDIDATE_DELTA_REFERENCES"]
                                 [batch_idx][candidate_idx]),
                        0,
                    )
                    abs_helpers["CENTERLINE"] = np.expand_dims(curr_centerline, 0)

                    abs_input, abs_output = baseline_utils.get_abs_traj(
                        test_input_seq.clone().cpu().numpy(),
                        preds.detach().clone().cpu().numpy(),
                        args,
                        abs_helpers,
                    )

                    # array of shape (1,30,2) to list of (30,2)
                    abs_outputs.append(abs_output[0])
                forecasted_trajectories[seq_id] = abs_outputs
    os.makedirs(forecasted_save_dir, exist_ok=True)
    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)

def infer_maml_map(
        test_loader: Any,
        support_loader: Any,
        encoder: Any,
        decoder: Any,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
        epoch: int,
        loader_len : int,
        support_loader_len: int,
):
    """Infer function for map-based LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance
        epoch: Epoch at which we ended training at
    """
    
    forecasted_trajectories = {}
    args = parse_arguments()
    per_step_loss_importance_vecor = get_per_step_loss_importance_vector(args, epoch) if args.num_training_steps_per_iter > 0 else None
    criterion = nn.MSELoss()
    total_loss = []

    with tqdm(total=loader_len, desc='Testing on epoch: {}'.format(epoch), position=0) as pbar:
        for i, data_batch in enumerate(test_loader):

            # Support task data batch:
            support_batch = next(iter(support_loader))

            encoder.eval()
            decoder.eval()
            pbar.update(1)

            loss = 0
            (support_input_seqs, support_obs_seqs, test_input_seq, test_obs_seq, helpers) = data_batch
            batch_helpers = list(zip(*helpers))

            helpers_dict = {}
            for k, v in config.LSTM_HELPER_DICT_IDX.items():
                helpers_dict[k] = batch_helpers[v]
        
            batch_size = test_input_seq.shape[0]
            input_length = test_input_seq.shape[2]
           
            with tqdm(total=batch_size, desc='Iterating over batch', position=1) as pbar2:
                for batch_idx in range(batch_size):
                    pbar2.update(1)
                    num_candidates = len(
                        helpers_dict["CANDIDATE_CENTERLINES"][batch_idx])
                    curr_centroids = helpers_dict["CENTROIDS"][batch_idx]
                    seq_id = int(helpers_dict["SEQ_PATHS"][batch_idx])
                    abs_outputs = []
                    # Predict using every centerline candidate for the current trajectory
                    #import pdb; pdb.set_trace();
                    for candidate_idx in range(num_candidates):
                        curr_centerline = helpers_dict["CANDIDATE_CENTERLINES"][
                            batch_idx][candidate_idx]
                        curr_nt_dist = helpers_dict["CANDIDATE_NT_DISTANCES"][
                            batch_idx][candidate_idx]

                        # Since this is test set all our inputs are gonna be None, gotta build
                        # them ourselves.
                        test_input_seq = torch.FloatTensor(
                        np.expand_dims(curr_nt_dist[:args.obs_len].astype(float),
                                       0)).to(device)

                        # Update support batch and feed to maml_forward
                        #import pdb; pdb.set_trace()
                        tempbatch = list(support_batch)
                        tempbatch[2] = test_input_seq.unsqueeze(0)
                        support_batch = tuple(tempbatch)

                        loss, preds = maml_forward(
                            args = args,
                            data_batch = support_batch,
                            epoch = epoch,
                            criterion = criterion,
                            encoder = encoder,
                            decoder = decoder,
                            model_utils = model_utils,
                            per_step_loss_importance_vecor = per_step_loss_importance_vecor,
                            second_order = False,
                            rollout_len = args.pred_len,
                            encoder_learning_rule = None,
                            decoder_learning_rule = None,
                        )
                        # Preds has been broadcasted to the shape of output, which means it has batch size,
                        # but actually it's just copied, so take one of the elements only
                        preds = preds[0,:,:].unsqueeze(0)
                        # Get absolute trajectory
                        abs_helpers = {}
                        abs_helpers["REFERENCE"] = np.expand_dims(
                            np.array(helpers_dict["CANDIDATE_DELTA_REFERENCES"]
                                     [batch_idx][candidate_idx]),
                            0,
                        )
                        abs_helpers["CENTERLINE"] = np.expand_dims(curr_centerline, 0)

                        abs_input, abs_output = baseline_utils.get_abs_traj(
                            test_input_seq.clone().cpu().numpy(),
                            preds.detach().clone().cpu().numpy(),
                            args,
                            abs_helpers,
                        )

                        # array of shape (1,30,2) to list of (30,2)
                        abs_outputs.append(abs_output[0])
                    forecasted_trajectories[seq_id] = abs_outputs
    os.makedirs(forecasted_save_dir, exist_ok=True)
    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)

def infer_maml_map_simplified(
        test_loader: Any,
        support_loader: Any,
        encoder: Any,
        decoder: Any,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
        epoch: int,
        loader_len : int,
        encoder_learning_rules = None, 
        decoder_learning_rules = None,

):
    """Infer function for map-based LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance
        epoch: Epoch at which we ended training at
    """
    
    forecasted_trajectories = {}
    args = parse_arguments()
    criterion = nn.MSELoss()
    total_loss = []

    for i, (support_batch, data_batch) in enumerate(zip(support_loader, test_loader)):

        encoder.eval()
        decoder.eval()

        #import pdb; pdb.set_trace()
        encoder_parameters, decoder_parameters = maml_infer_forward(
                                                   args = args,
                                                   data_batch = support_batch,
                                                   epoch = epoch,
                                                   criterion = criterion,
                                                   encoder = encoder,
                                                   decoder = decoder,
                                                   model_utils = model_utils,
                                                   encoder_learning_rules = encoder_learning_rules,
                                                   decoder_learning_rules = decoder_learning_rules,
                                                 )

        (_, _, social_inputs, _, helpers) = data_batch
        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]
    
        batch_size = data_batch[2].shape[0]
       
        with tqdm(total=batch_size, desc='Iterating over batch', position=1) as pbar2:
            for batch_idx in range(batch_size):
                pbar2.update(1)

                num_candidates = len(helpers_dict["CANDIDATE_CENTERLINES"][batch_idx])
                curr_centroids = helpers_dict["CENTROIDS"][batch_idx]
                seq_id = int(helpers_dict["SEQ_PATHS"][batch_idx])
                abs_outputs = []

                # Predict using every centerline candidate for the current trajectory
                for candidate_idx in range(num_candidates):
                    curr_centerline = helpers_dict["CANDIDATE_CENTERLINES"][
                        batch_idx][candidate_idx]
                    curr_nt_dist = helpers_dict["CANDIDATE_NT_DISTANCES"][
                        batch_idx][candidate_idx]

                    # Since this is test set all our inputs are gonna be None, gotta build
                    # them ourselves.
                    test_input_seq = torch.FloatTensor(
                                     np.expand_dims(curr_nt_dist[:args.obs_len].astype(float),
                                     0)).to(device)
                    if args.use_social: 
                    	test_input_seq = torch.cat((test_input_seq, social_inputs[batch_idx,:,:,2:].to(device)), dim=2)
                    test_target_seq = torch.zeros(test_input_seq.shape[0], 30, 2)

                    preds = lstm_infer_forward(
                              num_layers = args.num_layers,
                              encoder = encoder,
                              decoder = decoder,
                              encoder_params = encoder_parameters,
                              decoder_params = decoder_parameters,
                              input_seq = test_input_seq,
                              target_seq = test_target_seq,
                              obs_len = args.obs_len,
                              pred_len = args.pred_len,
                              model_utils = model_utils,
                              use_attention=args.use_attention,
                            )

                    abs_helpers = {}
                    abs_helpers["REFERENCE"] = np.expand_dims(
                        np.array(helpers_dict["CANDIDATE_DELTA_REFERENCES"]
                                 [batch_idx][candidate_idx]),
                        0,
                    )
                    abs_helpers["CENTERLINE"] = np.expand_dims(curr_centerline, 0)

                    abs_input, abs_output = baseline_utils.get_abs_traj(
                        test_input_seq.clone().cpu().numpy(),
                        preds.detach().clone().cpu().numpy(),
                        args,
                        abs_helpers,
                    )

                    # array of shape (1,30,2) to list of (30,2)
                    abs_outputs.append(abs_output[0])
                forecasted_trajectories[seq_id] = abs_outputs
    os.makedirs(forecasted_save_dir, exist_ok=True)
    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)

def infer_map(
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
):
    """Infer function for map-based LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance

    """
    args = parse_arguments()
    global best_loss
    forecasted_trajectories = {}
    for i, (_input, target, helpers) in enumerate(test_loader):

        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]

        # Iterate over every element in the batch
        for batch_idx in range(batch_size):
            num_candidates = len(
                helpers_dict["CANDIDATE_CENTERLINES"][batch_idx])
            curr_centroids = helpers_dict["CENTROIDS"][batch_idx]
            seq_id = int(helpers_dict["SEQ_PATHS"][batch_idx])
            abs_outputs = []

            # Predict using every centerline candidate for the current trajectory
            for candidate_idx in range(num_candidates):
                curr_centerline = helpers_dict["CANDIDATE_CENTERLINES"][
                    batch_idx][candidate_idx]
                curr_nt_dist = helpers_dict["CANDIDATE_NT_DISTANCES"][
                    batch_idx][candidate_idx]

                _input = torch.FloatTensor(
                    np.expand_dims(curr_nt_dist[:args.obs_len].astype(float),
                                   0)).to(device)

                # Initialize encoder hidden state
                encoder_hidden = model_utils.init_hidden(
                    1, encoder.module.hidden_size
                    if use_cuda else encoder.hidden_size)

                # Encode observed trajectory
                for ei in range(input_length):
                    encoder_input = _input[:, ei, :]
                    encoder_hidden, encoder_out = encoder(encoder_input, encoder_hidden)

                # Initialize decoder input with last coordinate in encoder
                decoder_input = encoder_input[:, :2]

                # Initialize decoder hidden state as encoder hidden state
                decoder_hidden = encoder_hidden

                decoder_outputs = torch.zeros((1, args.pred_len, 2)).to(device)

                # Decode hidden state in future trajectory
                for di in range(args.pred_len):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    decoder_outputs[:, di, :] = decoder_output

                    # Use own predictions as inputs at next step
                    decoder_input = decoder_output

                # Get absolute trajectory
                abs_helpers = {}
                abs_helpers["REFERENCE"] = np.expand_dims(
                    np.array(helpers_dict["CANDIDATE_DELTA_REFERENCES"]
                             [batch_idx][candidate_idx]),
                    0,
                )
                abs_helpers["CENTERLINE"] = np.expand_dims(curr_centerline, 0)

                abs_input, abs_output = baseline_utils.get_abs_traj(
                    _input.clone().cpu().numpy(),
                    decoder_outputs.detach().clone().cpu().numpy(),
                    args,
                    abs_helpers,
                )

                # array of shape (1,30,2) to list of (30,2)
                abs_outputs.append(abs_output[0])
            forecasted_trajectories[seq_id] = abs_outputs

    os.makedirs(forecasted_save_dir, exist_ok=True)
    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)


def infer_helper(
        curr_data_dict: Dict[str, Any],
        support_data_dict: Dict[str, Any],
        start_idx: int,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        model_utils: ModelUtils,
        forecasted_save_dir: str,
        epoch: int,
        encoder_learning_rules = None, 
        decoder_learning_rules = None,
):
    """Run inference on the current joblib batch.

    Args:
        curr_data_dict: Data dictionary for the current joblib batch
        start_idx: Start idx of the current joblib batch
        encoder: Encoder network instance
        decoder: Decoder network instance
        model_utils: ModelUtils instance
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        epoch: The epoch which we stopped training at
    """
    args = parse_arguments()

    curr_test_dataset = LSTMDataset_maml_simplified(curr_data_dict, args, "test", 0)
    curr_test_loader = torch.utils.data.DataLoader(
        curr_test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        collate_fn=model_utils.my_collate_fn_maml,
    )
    test_loader_len = len(iter(curr_test_loader))
   
    # For now use test batch size because test is same as val
    curr_support_dataset = LSTMDataset_maml_simplified(support_data_dict, args, "val", 0)
    curr_support_loader = torch.utils.data.DataLoader(
        curr_support_dataset,
        shuffle=False,
        # TODO fix later
        batch_size=args.test_batch_size,
        collate_fn=model_utils.my_collate_fn_maml,
    )

    if args.use_map:
        if args.maml:
            print(f"#### LSTM+map maml inference at index {start_idx} ####")
            infer_maml_map_simplified(
                curr_test_loader,
                curr_support_loader,
                encoder,
                decoder,
                start_idx,
                forecasted_save_dir,
                model_utils,
                epoch,
                test_loader_len,
                encoder_learning_rules = encoder_learning_rules, 
                decoder_learning_rules = decoder_learning_rules,
            )
        else:
            print(f"#### LSTM+map inference at index {start_idx} ####")
            infer_map(
                curr_test_loader,
                encoder,
                decoder,
                start_idx,
                forecasted_save_dir,
                model_utils,
            )

    else:
        print(f"#### LSTM+social inference at {start_idx} ####"
              ) if args.use_social else print(
                  f"#### LSTM inference at {start_idx} ####")
        infer_absolute(
            curr_test_loader,
            encoder,
            decoder,
            start_idx,
            forecasted_save_dir,
            model_utils,
        )


def main():
    """Main."""
    args = parse_arguments()

    #if not baseline_utils.validate_args(args):
    #    return

    print(f"Using all ({joblib.cpu_count()}) CPUs....")
    if use_cuda:
        print(f"Using all ({torch.cuda.device_count()}) GPUs...")

    model_utils = ModelUtils()

    # key for getting feature set
    # Get features
    if args.use_map and args.use_social:
        baseline_key = "map_social"
    elif args.use_map:
        baseline_key = "map"
    elif args.use_social:
        baseline_key = "social"
    else:
        baseline_key = "none"

    # Get data
    data_dict = baseline_utils.get_data(args, baseline_key)

    # Get model
    criterion = nn.MSELoss()
    if args.maml:
        encoder = MetaEncoderRNN(
            input_size=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]), num_layers=args.num_layers)
        if args.use_attention:
            decoder = MetaAttDecoderRNN(output_size=2, num_layers=args.num_layers, obs_len = args.obs_len)
        else:
            decoder = MetaDecoderRNN(output_size=2, num_layers=args.num_layers)
    else:
        encoder = EncoderRNN(
            input_size=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]))
        decoder = DecoderRNN(output_size=2)
    if use_cuda:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    encoder.to(device)
    decoder.to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, amsgrad=False)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, amsgrad=False)
    #encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=encoder_optimizer, T_max=args.end_epoch,
    #                                                          eta_min=args.min_lr)
    #decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=decoder_optimizer, T_max=args.end_epoch,
    #                                                          eta_min=args.min_lr)
    encoder_scheduler = None
    decoder_scheduler = None
    encoder_optimizers = (encoder_optimizer, encoder_scheduler)
    decoder_optimizers = (decoder_optimizer, decoder_scheduler)

    encoder_learning_rule_dict = None
    decoder_learning_rule_dict = None

    # If model_path provided, resume from saved checkpoint
    if args.model_path is not None and os.path.isfile(args.model_path):
        epoch, rollout_len, _, encoder_learning_rule_dict, decoder_learning_rule_dict = model_utils.load_checkpoint(
            args.model_path, encoder, decoder, encoder_optimizer,
            decoder_optimizer)
        start_epoch = epoch + 1
        start_rollout_idx = ROLLOUT_LENS.index(rollout_len)

    else:
        start_epoch = 0
        start_rollout_idx = 0
    
    encoder_learning_rule_and_dict = (None, None)
    decoder_learning_rule_and_dict = (None, None)
    if args.use_lslr:
        encoder_learning_rule = baseline_utils.LSLRGradientDescentLearningRule(
            device, args.num_training_steps_per_iter, args.use_learnable_lr, 1e-3,
        )
        decoder_learning_rule = baseline_utils.LSLRGradientDescentLearningRule(
            device, args.num_training_steps_per_iter, args.use_learnable_lr, 1e-3,
        )
        encoder_learning_rule.initialise(get_named_params_dicts(encoder))
        decoder_learning_rule.initialise(get_named_params_dicts(decoder))

        if encoder_learning_rule_dict and decoder_learning_rule_dict:
            encoder_learning_rule_and_dict = (encoder_learning_rule, encoder_learning_rule_dict)
            decoder_learning_rule_and_dict = (decoder_learning_rule, decoder_learning_rule_dict)

    else:
        encoder_learning_rule = None
        decoder_learning_rule = None

    if not args.test:

        # Tensorboard logger
        log_dir = os.path.join(os.getcwd(), "lstm_logs", baseline_key)

        # Get PyTorch Dataset
        train_dataset = None
        val_dataset = None
        if args.maml: 
            seed = 0 # np.random.randint(10)
            train_dataset = LSTMDataset_maml_simplified(data_dict, args, "train", seed)
            val_dataset = LSTMDataset_maml_simplified(data_dict, args, "val", seed)
        else:
            train_dataset = LSTMDataset(data_dict, args, "train")
            val_dataset = LSTMDataset(data_dict, args, "val")

        # Setting Dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            num_workers=8,
            shuffle=True,
            drop_last=False,
            collate_fn=model_utils.my_collate_fn_maml if args.maml else model_utils.my_collate_fn,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            num_workers=8,
            drop_last=False,
            shuffle=False,
            collate_fn=model_utils.my_collate_fn_maml if args.maml else model_utils.my_collate_fn,
        )


        print("Training begins ...")

        decrement_counter = 0

        #import pdb; pdb.set_trace();
        epoch = start_epoch
        global_start_time = time.time()
        train_loader_len = len(iter(train_loader))
        val_loader_len = len(iter(val_loader))
        #import pdb; pdb.set_trace();
        for i in range(start_rollout_idx, len(ROLLOUT_LENS)):
            rollout_len = ROLLOUT_LENS[i]
            logger = Logger(log_dir, name="{}".format(rollout_len))
            best_loss = float("inf")
            prev_loss = best_loss
            while epoch < args.end_epoch:
                start = time.time()
                if args.maml:
                    train_maml_oversimplified(
                        train_loader,
                        epoch,
                        criterion,
                        logger,
                        encoder,
                        decoder,
                        encoder_optimizers,
                        decoder_optimizers,
                        model_utils,
                        train_loader_len,
                        rollout_len,
                        encoder_learning_rule,
                        decoder_learning_rule,
                    )
                else:
                    train(
                        train_loader,
                        epoch,
                        criterion,
                        logger,
                        encoder,
                        decoder,
                        encoder_optimizer,
                        decoder_optimizer,
                        model_utils,
                        rollout_len,
                    )
                end = time.time()

                print(
                    f"Training epoch completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                )

                epoch += 1
                if epoch % 5 == 0:
                    start = time.time()
                    if args.maml:
                        #import pdb; pdb.set_trace();
                        prev_loss, decrement_counter = validate_maml(
                            val_loader,
                            epoch,
                            criterion,
                            logger,
                            encoder,
                            decoder,
                            encoder_optimizers,
                            decoder_optimizers,
                            model_utils,
                            val_loader_len,
                            prev_loss,
                            decrement_counter,
                            rollout_len,
                            encoder_learning_rule,
                            decoder_learning_rule,
                        )

                    else:
                        prev_loss, decrement_counter = validate(
                            val_loader,
                            epoch,
                            criterion,
                            logger,
                            encoder,
                            decoder,
                            encoder_optimizer,
                            decoder_optimizer,
                            model_utils,
                            prev_loss,
                            decrement_counter,
                            rollout_len,
                        )
                    end = time.time()
                    print(
                        f"Validation completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                    )

                    # If val loss increased 3 times consecutively, go to next rollout length
                    if decrement_counter > 3:
                        break

    else:

        start_time = time.time()

        temp_save_dir = tempfile.mkdtemp()

        #import pdb; pdb.set_trace();
        test_size = data_dict["test_input"].shape[0]
        test_data_subsets = baseline_utils.get_test_data_dict_subset(
            data_dict, args)
        
        # We also use val input because support tasks need to train too even during test
        # Luckily they are the same size since test is actually using val data csv files
        support_size = data_dict["val_input"].shape[0]
        support_data_subsets = baseline_utils.get_support_data_dict_subset(
            data_dict, args)
        
        # test_batch_size should be lesser than joblib_batch_size
        Parallel(n_jobs=6, verbose=2)(
            delayed(infer_helper)(test_data_subsets[i], support_data_subsets[i], i, encoder, decoder,
                                  model_utils, temp_save_dir, epoch, encoder_learning_rule_and_dict, 
				  decoder_learning_rule_and_dict)
            for i in range(0, test_size, args.joblib_batch_size))

        baseline_utils.merge_saved_traj(temp_save_dir, args.traj_save_path)
        shutil.rmtree(temp_save_dir)

        end = time.time()
        print(f"Test completed in {(end - start_time) / 60.0} mins")
        print(f"Forecasted Trajectories saved at {args.traj_save_path}")


if __name__ == "__main__":
    main()
