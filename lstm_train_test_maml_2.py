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
                        default=512,
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
                         help="Number of steps inside an iter in a MAML loop")
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
        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

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
        out = F.linear(input=x, weight=weight, bias=bias)
        return out


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

class MetaEncoderRNN(nn.Module):
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
        super(MetaEncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = MetaLinearLayer(input_size, embedding_size, use_bias=True)
        self.lstm1 = MetaLSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden, param=None):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 

        """
        param_dict = None if param == None else self.preprocess_param_dict(param)
        embedded = F.relu(self.linear1(x, (None if param == None else param_dict['linear1'])))
        hidden = self.lstm1(embedded, hidden, (None if param == None else param_dict['lstm1']))
        return hidden
    
    def preprocess_param_dict(self, param_dict):
        reordered_dict = {}
        reordered_dict['linear1'] = {}
        reordered_dict['lstm1'] ={}
        reordered_dict['lstm1']['i2h'] ={}
        reordered_dict['lstm1']['h2h'] ={}
        for name, param in param_dict.items():
            names_split = name.split('.')
            if names_split[0] == 'linear1':
                reordered_dict['linear1'][names_split[1]] = param
            elif names_split[0] == 'lstm1':
                if names_split[1] == 'i2h':
                    reordered_dict['lstm1']['i2h'][names_split[2]] = param
                elif names_split[1] == 'h2h':
                    reordered_dict['lstm1']['h2h'][names_split[2]] = param
        return reordered_dict

class MetaDecoderRNN(nn.Module):
    """Encoder Network."""
    def __init__(self, embedding_size=8, hidden_size=16, output_size=2):

        """Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output
        """
        super(MetaDecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = MetaLinearLayer(output_size, embedding_size, use_bias=True)
        self.lstm1 = MetaLSTMCell(embedding_size, hidden_size)
        self.linear2 = MetaLinearLayer(hidden_size, output_size, use_bias=True)

    def forward(self, x: torch.FloatTensor, hidden, param=None):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 

        """
        param_dict = None if param == None else self.preprocess_param_dict(param)
        embedded = F.relu(self.linear1(x, (None if param == None else param_dict['linear1'])))
        hidden = self.lstm1(embedded, hidden, (None if param == None else param_dict['lstm1']))
        output = self.linear2(hidden[0], (None if param == None else param_dict['linear2']))
        return output, hidden
    
    def preprocess_param_dict(self, param_dict):
        reordered_dict = {}
        reordered_dict['linear1'] = {}
        reordered_dict['linear2'] = {}
        reordered_dict['lstm1'] ={}
        reordered_dict['lstm1']['i2h'] ={}
        reordered_dict['lstm1']['h2h'] ={}
        for name, param in param_dict.items():
            names_split = name.split('.')
            if names_split[0] == 'linear1':
                reordered_dict['linear1'][names_split[1]] = param
            elif names_split[0] == 'lstm1':
                if names_split[1] == 'i2h':
                    reordered_dict['lstm1']['i2h'][names_split[2]] = param
                elif names_split[1] == 'h2h':
                    reordered_dict['lstm1']['h2h'][names_split[2]] = param
            elif names_split[0] == 'linear2':
                reordered_dict['linear2'][names_split[1]] = param
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
):
    batch_size =  input_seq.shape[0]
    # Initialize losses
    loss = 0

    # Initialize encoder hidden state
    encoder_hidden = model_utils.init_hidden(
        batch_size,
        encoder.module.hidden_size if use_cuda else encoder.hidden_size)


    # Encode observed trajectory
    for ei in range(obs_len):
        encoder_input = input_seq[:, ei, :]
        encoder_hidden = encoder(encoder_input, encoder_hidden, encoder_params)

    # Initialize decoder input with last coordinate in encoder
    decoder_input = encoder_input[:, :2]

    # Initialize decoder hidden state as encoder hidden state
    decoder_hidden = encoder_hidden

    decoder_outputs = torch.zeros(target_seq.shape).to(device)

    # Decode hidden state in future trajectory
    for di in range(pred_len):
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

def get_named_params_dicts(
    model: Any,
):
    params_dict = dict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            key = (name.replace('module.', '')) if 'module.' in name else name
            params_dict[key] = param.to(device)
    
    return params_dict

def update_params(
    param_dict,
    grad_dict,
):
    args = parse_arguments()
    updated_weights_dict = dict()
    for key in grad_dict.keys():
        updated_weights_dict[key] = param_dict[key] - args.lr * grad_dict[key]

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
):
    zero_grad(encoder, encoder_params)
    zero_grad(decoder, decoder_params)

    encoder_grads = torch.autograd.grad(loss, encoder_params.values(), create_graph=use_second_order)
    decoder_grads = torch.autograd.grad(loss, decoder_params.values(), create_graph=use_second_order)

    encoder_grads_wrt_param_names = dict(zip(encoder_params.keys(), encoder_grads))
    decoder_grads_wrt_param_names = dict(zip(decoder_params.keys(), decoder_grads))

    encoder_params = update_params(encoder_params, encoder_grads_wrt_param_names)
    decoder_params = update_params(decoder_params, decoder_grads_wrt_param_names) 

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

def train_maml(
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
    encoder.zero_grad()
    decoder.zero_grad()
    encoder.train()
    decoder.train()
    
    for i, (support_input_seqs, support_obs_seqs, train_input_seqs, train_obs_seqs, helpers) in enumerate(train_loader):
        support_input_seqs = support_input_seqs.to(device)
        support_obs_seqs = support_obs_seqs.to(device)
        train_input_seqs = train_input_seqs.to(device)
        train_obs_seqs = train_obs_seqs.to(device)
        
        loss_list = []
        for j, (support_input_seq, support_obs_seq, train_input_seq, train_obs_seq) in enumerate(
                                                                                        zip(support_input_seqs,
                                                                                            support_obs_seqs,
                                                                                            train_input_seqs,
                                                                                            train_obs_seqs)):
            # Copy the model for MAML inner loop
            encoder_copy_params = get_named_params_dicts(encoder)
            decoder_copy_params = get_named_params_dicts(decoder)

            train_loss = None
            encoder_dict = get_named_params_dicts(encoder)
            decoder_dict = get_named_params_dicts(decoder)
       
            for iter in range(args.num_training_steps_per_iter):
                support_loss, supprt_pred = lstm_forward(
                    encoder,
                    decoder,
                    encoder_copy_params,
                    decoder_copy_params,
                    support_input_seq,
                    support_obs_seq,
                    args.obs_len,
                    args.pred_len,
                    criterion,
                    model_utils
                )

                encoder_copy_params, decoder_copy_params = maml_inner_loop_update(
                    support_loss, encoder, decoder, encoder_copy_params, decoder_copy_params, args.second_order
                )

                if(iter == args.num_training_steps_per_iter - 1):
                    train_loss, train_preds = lstm_forward(
                        encoder,
                        decoder,
                        encoder_copy_params,
                        decoder_copy_params,
                        train_input_seq,
                        train_obs_seq,
                        args.obs_len,
                        args.pred_len,
                        criterion,
                        model_utils
                    ) 


            # Zero the gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = train_loss

            if global_step % 1000 == 0:
                loss_list.append(loss.item())

            # Backpropagate
            loss.backward()
            clamp_grads(encoder)
            clamp_grads(decoder)

            encoder_optimizer.step()
            decoder_optimizer.step()
            if j % 100 == 0:
                print(
                    f"Training inner loop {j}-- Epoch:{epoch}, loss:{loss}, Rollout:{rollout_len}")

        if global_step % 1000 == 0:

            # Log results
            avg_loss = np.array(loss_list).mean()
            print(
                f"--------------------------------------\n\
                Train -- Epoch:{epoch}, avg loss:{avg_loss}, Rollout:{rollout_len}\n\
                --------------------------------------")

            logger.scalar_summary(tag="Train/loss",
                                  value=avg_loss,
                                  step=epoch)

        global_step += 1

def train_maml_simplified(
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
    encoder.zero_grad()
    decoder.zero_grad()
    encoder.train()
    decoder.train()
    import pdb; pdb.set_trace();
    for i, (support_input_seqs, support_obs_seqs, train_input_seqs, train_obs_seqs, helpers) in enumerate(train_loader):
        #support_input_seqs = support_input_seqs.to(device)
        #support_obs_seqs = support_obs_seqs.to(device)
        #train_input_seqs = train_input_seqs.to(device)
        #train_obs_seqs = train_obs_seqs.to(device)

        maml_dataset = TensorDataset(support_input_seqs, support_obs_seqs, train_input_seqs, train_obs_seqs)
        maml_dataloader = torch.utils.data.DataLoader(maml_dataset, batch_size = args.minibatch_size, num_workers = args.num_workers)
        
        loss_list = []
        for j, (batch_support_input_seq, batch_support_obs_seq, train_input_seq, train_obs_seq) in enumerate(maml_dataloader):
            # Copy the model for MAML inner loop
            shot = args.shot if batch_support_input_seq.shape[0] >= args.shot else batch_support_input_seq.shape[0]
            support_input_seq = batch_support_input_seq[:shot, :].squeeze(dim=1)
            support_obs_seq = batch_support_obs_seq[:shot, :].squeeze(dim=1)

            support_input_seq = support_input_seq.to(device)
            support_obs_seq = support_obs_seq.to(device)
            train_input_seq = train_input_seq.squeeze(dim=1).to(device)
            train_obs_seq = train_obs_seq.squeeze(dim=1).to(device)


            encoder_copy_params = get_named_params_dicts(encoder)
            decoder_copy_params = get_named_params_dicts(decoder)

            train_loss = None
            encoder_dict = get_named_params_dicts(encoder)
            decoder_dict = get_named_params_dicts(decoder)
       
            for iter in range(args.num_training_steps_per_iter):
                support_loss, supprt_pred = lstm_forward(
                    encoder,
                    decoder,
                    encoder_copy_params,
                    decoder_copy_params,
                    support_input_seq,
                    support_obs_seq,
                    args.obs_len,
                    args.pred_len,
                    criterion,
                    model_utils
                )

                encoder_copy_params, decoder_copy_params = maml_inner_loop_update(
                    support_loss, encoder, decoder, encoder_copy_params, decoder_copy_params, args.second_order
                )

                if(iter == args.num_training_steps_per_iter - 1):
                    train_loss, train_preds = lstm_forward(
                        encoder,
                        decoder,
                        encoder_copy_params,
                        decoder_copy_params,
                        train_input_seq,
                        train_obs_seq,
                        args.obs_len,
                        args.pred_len,
                        criterion,
                        model_utils
                    ) 


            # Zero the gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = train_loss

            if global_step % 1000 == 0:
                loss_list.append(loss.item())

            # Backpropagate
            loss.backward()
            clamp_grads(encoder)
            clamp_grads(decoder)

            encoder_optimizer.step()
            decoder_optimizer.step()
            #if j % 100 == 0:
            print(
                f"Training inner loop {j}-- Epoch:{epoch}, loss:{loss}, Rollout:{rollout_len}")

        if global_step % 1000 == 0:

            # Log results
            avg_loss = np.array(loss_list).mean()
            print(
                f"--------------------------------------\n\
                Train -- Epoch:{epoch}, avg loss:{avg_loss}, Rollout:{rollout_len}\n\
                --------------------------------------")

            logger.scalar_summary(tag="Train/loss",
                                  value=avg_loss,
                                  step=epoch)

        global_step += 1

def train_maml_oversimplified(
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
    encoder.zero_grad()
    decoder.zero_grad()
    encoder.train()
    decoder.train()
    import pdb; pdb.set_trace();
    for i, (support_input_seqs, support_obs_seqs, train_input_seq, train_obs_seq, helpers) in enumerate(train_loader):
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
       
        for iter in range(args.num_training_steps_per_iter):
            support_loss, supprt_pred = lstm_forward(
                encoder,
                decoder,
                encoder_copy_params,
                decoder_copy_params,
                support_input_seq,
                support_obs_seq,
                args.obs_len,
                rollout_len,
                criterion,
                model_utils
            )

            encoder_copy_params, decoder_copy_params = maml_inner_loop_update(
                support_loss, encoder, decoder, encoder_copy_params, decoder_copy_params, args.second_order
            )

            if(iter == args.num_training_steps_per_iter - 1):
                train_loss, train_preds = lstm_forward(
                    encoder,
                    decoder,
                    encoder_copy_params,
                    decoder_copy_params,
                    train_input_seq,
                    train_obs_seq,
                    args.obs_len,
                    rollout_len,
                    criterion,
                    model_utils
                ) 


        # Zero the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = train_loss

        # Backpropagate
        loss.backward()
        clamp_grads(encoder)
        clamp_grads(decoder)

        encoder_optimizer.step()
        decoder_optimizer.step()
        #if i % 100 == 0:
        print(
            f"Train -- Optimizer loop:{i} Epoch:{epoch}, avg loss:{loss}, Rollout:{rollout_len}")

        if global_step % 1000 == 0:

            # Log results
            print(
                f"Train -- Epoch:{epoch}, avg loss:{loss}, Rollout:{rollout_len}")

            logger.scalar_summary(tag="Train/loss",
                                  value=loss.item(),
                                  step=epoch)

        global_step += 1


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
                    encoder_hidden = encoder(encoder_input, encoder_hidden)

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
        start_idx: int,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        model_utils: ModelUtils,
        forecasted_save_dir: str,
):
    """Run inference on the current joblib batch.

    Args:
        curr_data_dict: Data dictionary for the current joblib batch
        start_idx: Start idx of the current joblib batch
        encoder: Encoder network instance
        decoder: Decoder network instance
        model_utils: ModelUtils instance
        forecasted_save_dir: Directory where forecasted trajectories are to be saved

    """
    args = parse_arguments()
    curr_test_dataset = LSTMDataset(curr_data_dict, args, "test")
    curr_test_loader = torch.utils.data.DataLoader(
        curr_test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        collate_fn=model_utils.my_collate_fn,
    )

    if args.use_map:
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

    if not baseline_utils.validate_args(args):
        return

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
            input_size=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]))
        decoder = MetaDecoderRNN(output_size=2)
    else:
        encoder = EncoderRNN(
            input_size=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]))
        decoder = DecoderRNN(output_size=2)
    if use_cuda:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    encoder.to(device)
    decoder.to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    # If model_path provided, resume from saved checkpoint
    if args.model_path is not None and os.path.isfile(args.model_path):
        epoch, rollout_len, _ = model_utils.load_checkpoint(
            args.model_path, encoder, decoder, encoder_optimizer,
            decoder_optimizer)
        start_epoch = epoch + 1
        start_rollout_idx = ROLLOUT_LENS.index(rollout_len) + 1

    else:
        start_epoch = 0
        start_rollout_idx = 0

    if not args.test:

        # Tensorboard logger
        log_dir = os.path.join(os.getcwd(), "lstm_logs", baseline_key)

        # Get PyTorch Dataset
        train_dataset = None
        if args.maml: 
            seed = 0 # np.random.randint(10)
            train_dataset = LSTMDataset_maml_simplified(data_dict, args, "train", seed)
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
            drop_last=False,
            shuffle=False,
            collate_fn=model_utils.my_collate_fn,
        )

        print("Training begins ...")

        decrement_counter = 0

        #import pdb; pdb.set_trace();
        epoch = start_epoch
        global_start_time = time.time()
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
                        encoder_optimizer,
                        decoder_optimizer,
                        model_utils,
                        rollout_len,
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
                    if decrement_counter > 2:
                        break

    else:

        start_time = time.time()

        temp_save_dir = tempfile.mkdtemp()

        test_size = data_dict["test_input"].shape[0]
        test_data_subsets = baseline_utils.get_test_data_dict_subset(
            data_dict, args)

        # test_batch_size should be lesser than joblib_batch_size
        Parallel(n_jobs=-2, verbose=2)(
            delayed(infer_helper)(test_data_subsets[i], i, encoder, decoder,
                                  model_utils, temp_save_dir)
            for i in range(0, test_size, args.joblib_batch_size))

        baseline_utils.merge_saved_traj(temp_save_dir, args.traj_save_path)
        shutil.rmtree(temp_save_dir)

        end = time.time()
        print(f"Test completed in {(end - start_time) / 60.0} mins")
        print(f"Forecasted Trajectories saved at {args.traj_save_path}")


if __name__ == "__main__":
    main()
