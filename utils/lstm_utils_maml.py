"""lstm_utils.py contains utility functions for running LSTM Baselines."""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from collections import OrderedDict
import utils.baseline_config as config

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class LSTMDataset(Dataset):
    """PyTorch Dataset for LSTM Baselines."""
    def __init__(self, data_dict: Dict[str, Any], args: Any, mode: str):
        """Initialize the Dataset.

        Args:
            data_dict: Dict containing all the data
            args: Arguments passed to the baseline code
            mode: train/val/test mode

        """
        self.data_dict = data_dict
        self.args = args
        self.mode = mode

        # Get input
        self.input_data = data_dict["{}_input".format(mode)]
        if mode != "test":
            self.output_data = data_dict["{}_output".format(mode)]
        self.data_size = self.input_data.shape[0]

        # Get helpers
        self.helpers = self.get_helpers()
        self.helpers = list(zip(*self.helpers))

    def __len__(self):
        """Get length of dataset.

        Returns:
            Length of dataset

        """
        return self.data_size

    def __getitem__(self, idx: int
                    ) -> Tuple[torch.FloatTensor, Any, Dict[str, np.ndarray]]:
        """Get the element at the given index.

        Args:
            idx: Query index

        Returns:
            A list containing input Tensor, Output Tensor (Empty if test) and viz helpers. 

        """
        return (
            torch.FloatTensor(self.input_data[idx]),
            torch.empty(1) if self.mode == "test" else torch.FloatTensor(
                self.output_data[idx]),
            self.helpers[idx],
        )

    def get_helpers(self) -> Tuple[Any]:
        """Get helpers for running baselines.

        Returns:
            helpers: Tuple in the format specified by LSTM_HELPER_DICT_IDX

        Note: We need a tuple because DataLoader needs to index across all these helpers simultaneously.

        """
        helper_df = self.data_dict[f"{self.mode}_helpers"]
        candidate_centerlines = helper_df["CANDIDATE_CENTERLINES"].values
        candidate_nt_distances = helper_df["CANDIDATE_NT_DISTANCES"].values
        xcoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["X"]].astype("float")
        ycoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["Y"]].astype("float")
        centroids = np.stack((xcoord, ycoord), axis=2)
        _DEFAULT_HELPER_VALUE = np.full((centroids.shape[0]), None)
        city_names = np.stack(helper_df["FEATURES"].values
                              )[:, :, config.FEATURE_FORMAT["CITY_NAME"]]
        seq_paths = helper_df["SEQUENCE"].values
        translation = (helper_df["TRANSLATION"].values
                       if self.args.normalize else _DEFAULT_HELPER_VALUE)
        rotation = (helper_df["ROTATION"].values
                    if self.args.normalize else _DEFAULT_HELPER_VALUE)

        use_candidates = self.args.use_map and self.mode == "test"

        candidate_delta_references = (
            helper_df["CANDIDATE_DELTA_REFERENCES"].values
            if self.args.use_map and use_candidates else _DEFAULT_HELPER_VALUE)
        delta_reference = (helper_df["DELTA_REFERENCE"].values
                           if self.args.use_delta and not use_candidates else
                           _DEFAULT_HELPER_VALUE)

        helpers = [None for i in range(len(config.LSTM_HELPER_DICT_IDX))]

        # Name of the variables should be the same as keys in LSTM_HELPER_DICT_IDX
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers[v] = locals()[k.lower()]

        return tuple(helpers)


class LSTMDataset_maml(Dataset):
    """PyTorch Dataset for LSTM Baselines."""
    def __init__(self, data_dict: Dict[str, Any], args: Any, mode: str, seed: int):
        """Initialize the Dataset.
        Args:
            data_dict: Dict containing all the data
            args: Arguments passed to the baseline code
            mode: train/val/test mode

        """
        self.data_dict = data_dict
        self.args = args
        self.mode = mode
        self.seed = seed
        if mode == "train":
            self.obs_len = args.obs_len
            self.num_target_samples = args.num_target_samples
            self.shot = args.shot


        # Get input
        self.input_data = data_dict["{}_input".format(mode)]
        if mode != "test":
            self.output_data = data_dict["{}_output".format(mode)]
        self.data_size = self.input_data.shape[0]

        # Get helpers
        self.helpers = self.get_helpers()
        self.helpers = list(zip(*self.helpers))

    def __len__(self):
        """Get length of dataset.

        Returns:
            Length of dataset

        """
        return self.data_size

    def __getitem__(self, idx: int
                    ) -> Any:
        """Get the element at the given index.

        Args:
            idx: Query index

        Returns:
        if train
            A list containing (support_set, input) Tensor, (support_set_target, Output) Tensor (Empty if test) and viz helpers. 
        else
            A list containing input Tensor, Output Tensor (Empty if test) and viz helpers. 

        """
        if self.mode == "train":
            rng = np.random.RandomState(self.seed + idx)
            samples_list = np.array(rng.choice(self.data_size, self.num_target_samples+self.shot, replace=False))
            train_input_set = self.input_data[samples_list, :]
            train_output_set = self.output_data[samples_list, :]

            support_input_seq = train_input_set[:self.shot]
            support_obs_seq   = train_output_set[:self.shot]
            train_input_seq = train_input_set[self.shot:]
            train_obs_seq   = train_output_set[self.shot:]

            return(
                torch.FloatTensor(support_input_seq),
                torch.FloatTensor(support_obs_seq), 
                torch.FloatTensor(train_input_seq),
                torch.FloatTensor(train_obs_seq),
                [self.helpers[idx] for idx in samples_list[self.shot:]],
            )
        else:
            return (
                torch.FloatTensor(self.input_data[idx]),
                torch.empty(1) if self.mode == "test" else torch.FloatTensor(
                    self.output_data[idx]),
                self.helpers[idx],
            )

    def get_helpers(self) -> Tuple[Any]:
        """Get helpers for running baselines.

        Returns:
            helpers: Tuple in the format specified by LSTM_HELPER_DICT_IDX

        Note: We need a tuple because DataLoader needs to index across all these helpers simultaneously.

        """
        helper_df = self.data_dict[f"{self.mode}_helpers"]
        candidate_centerlines = helper_df["CANDIDATE_CENTERLINES"].values
        candidate_nt_distances = helper_df["CANDIDATE_NT_DISTANCES"].values
        xcoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["X"]].astype("float")
        ycoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["Y"]].astype("float")
        centroids = np.stack((xcoord, ycoord), axis=2)
        _DEFAULT_HELPER_VALUE = np.full((centroids.shape[0]), None)
        city_names = np.stack(helper_df["FEATURES"].values
                              )[:, :, config.FEATURE_FORMAT["CITY_NAME"]]
        seq_paths = helper_df["SEQUENCE"].values
        translation = (helper_df["TRANSLATION"].values
                       if self.args.normalize else _DEFAULT_HELPER_VALUE)
        rotation = (helper_df["ROTATION"].values
                    if self.args.normalize else _DEFAULT_HELPER_VALUE)

        use_candidates = self.args.use_map and self.mode == "test"

        candidate_delta_references = (
            helper_df["CANDIDATE_DELTA_REFERENCES"].values
            if self.args.use_map and use_candidates else _DEFAULT_HELPER_VALUE)
        delta_reference = (helper_df["DELTA_REFERENCE"].values
                           if self.args.use_delta and not use_candidates else
                           _DEFAULT_HELPER_VALUE)

        helpers = [None for i in range(len(config.LSTM_HELPER_DICT_IDX))]

        # Name of the variables should be the same as keys in LSTM_HELPER_DICT_IDX
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers[v] = locals()[k.lower()]

        return tuple(helpers)


class LSTMDataset_maml_simplified(Dataset):
    """PyTorch Dataset for LSTM Baselines."""
    def __init__(self, data_dict: Dict[str, Any], args: Any, mode: str, seed: int):
        """Initialize the Dataset.
        Args:
            data_dict: Dict containing all the data
            args: Arguments passed to the baseline code
            mode: train/val/test mode

        """
        self.data_dict = data_dict
        self.args = args
        self.mode = mode
        self.seed = seed
            
        self.obs_len = args.obs_len
        self.num_target_samples = args.num_target_samples
        self.shot = args.shot


        # Get input
        self.input_data = data_dict["{}_input".format(mode)]
        if mode != "test":
            self.output_data = data_dict["{}_output".format(mode)]
        self.data_size = self.input_data.shape[0]

        # Get helpers
        self.helpers = self.get_helpers()
        self.helpers = list(zip(*self.helpers))

    def __len__(self):
        """Get length of dataset.

        Returns:
            Length of dataset

        """
        return self.data_size

    def __getitem__(self, idx: int
                    ) -> Any:
        """Get the element at the given index.

        Args:
            idx: Query index

        Returns:
        if train
            A list containing (support_set, input) Tensor, (support_set_target, Output) Tensor (Empty if test) and viz helpers. 
        else
            A list containing input Tensor, Output Tensor (Empty if test) and viz helpers. 

        """
        rng = np.random.RandomState(self.seed + idx)
        #samples_list = np.concatenate((np.array(rng.choice(self.data_size, self.shot, replace=False)), np.array([idx])))
        samples_list = np.concatenate((rng.randint(0, self.data_size, 1), np.array([idx])))
        train_input_set = self.input_data[samples_list, :]
        support_input_seq = train_input_set[:1]
        train_input_seq = train_input_set[1:]

        train_output_set = self.output_data[samples_list, :] if self.mode != 'test' else None
        support_obs_seq   = train_output_set[:1] if self.mode !='test' else torch.empty(1)
        train_obs_seq   = train_output_set[1:] if self.mode != 'test' else torch.empty(1)

        return(
            torch.FloatTensor(support_input_seq),
            torch.FloatTensor(support_obs_seq), 
            torch.FloatTensor(train_input_seq),
            torch.FloatTensor(train_obs_seq),
            self.helpers[idx],
        )


    def get_helpers(self) -> Tuple[Any]:
        """Get helpers for running baselines.

        Returns:
            helpers: Tuple in the format specified by LSTM_HELPER_DICT_IDX

        Note: We need a tuple because DataLoader needs to index across all these helpers simultaneously.

        """
        helper_df = self.data_dict[f"{self.mode}_helpers"]
        candidate_centerlines = helper_df["CANDIDATE_CENTERLINES"].values
        candidate_nt_distances = helper_df["CANDIDATE_NT_DISTANCES"].values
        xcoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["X"]].astype("float")
        ycoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["Y"]].astype("float")
        centroids = np.stack((xcoord, ycoord), axis=2)
        _DEFAULT_HELPER_VALUE = np.full((centroids.shape[0]), None)
        city_names = np.stack(helper_df["FEATURES"].values
                              )[:, :, config.FEATURE_FORMAT["CITY_NAME"]]
        seq_paths = helper_df["SEQUENCE"].values
        translation = (helper_df["TRANSLATION"].values
                       if self.args.normalize else _DEFAULT_HELPER_VALUE)
        rotation = (helper_df["ROTATION"].values
                    if self.args.normalize else _DEFAULT_HELPER_VALUE)

        use_candidates = self.args.use_map and self.mode == "test"

        candidate_delta_references = (
            helper_df["CANDIDATE_DELTA_REFERENCES"].values
            if self.args.use_map and use_candidates else _DEFAULT_HELPER_VALUE)
        delta_reference = (helper_df["DELTA_REFERENCE"].values
                           if self.args.use_delta and not use_candidates else
                           _DEFAULT_HELPER_VALUE)

        helpers = [None for i in range(len(config.LSTM_HELPER_DICT_IDX))]

        # Name of the variables should be the same as keys in LSTM_HELPER_DICT_IDX
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers[v] = locals()[k.lower()]

        return tuple(helpers)


class ModelUtils:
    """Utils for LSTM baselines."""
    def save_checkpoint(self, save_dir: str, state: Dict[str, Any]) -> None:
        """Save checkpoint file.
        
        Args:
            save_dir: Directory where model is to be saved
            state: State of the model

        """
        filename = "{}/LSTM_rollout{}.pth.tar".format(save_dir,
                                                      state["rollout_len"])
        torch.save(state, filename)

    def load_checkpoint(
            self,
            checkpoint_file: str,
            encoder: Any,
            decoder: Any,
            encoder_optimizer: Any,
            decoder_optimizer: Any,
    ) -> Tuple[int, int, float, dict]:
        """Load the checkpoint.

        Args:
            checkpoint_file: Path to checkpoint file
            encoder: Encoder model
            decoder: Decoder model 

        Returns:
            epoch: epoch when the model was saved.
            rollout_len: horizon used
            best_loss: loss when the checkpoint was saved

        """
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            rollout_len = checkpoint["rollout_len"]
           
            encoder_learning_rule_dict = checkpoint["encoder_learning_rule_dict"]
            decoder_learning_rule_dict = checkpoint["decoder_learning_rule_dict"]

            # Remove "module" from encoder and decoder state dictionary keys
            newdict = OrderedDict()
            for key, value in checkpoint["encoder_state_dict"].items():
                newdict[key.replace('module.','')] = value    
            checkpoint["encoder_state_dict"] = newdict
            
            newdict = OrderedDict()
            for key, value in checkpoint["decoder_state_dict"].items():
                newdict[key.replace('module.','')] = value    
            checkpoint["decoder_state_dict"] = newdict

            if use_cuda:
                encoder.module.load_state_dict(
                    checkpoint["encoder_state_dict"])
                decoder.module.load_state_dict(
                    checkpoint["decoder_state_dict"])
            else:
                encoder.load_state_dict(checkpoint["encoder_state_dict"])
                decoder.load_state_dict(checkpoint["decoder_state_dict"])
            
            encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
            decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
            print(
                f"=> loaded checkpoint {checkpoint_file} (epoch: {epoch}, loss: {best_loss})"
            )
        else:
            print(f"=> no checkpoint found at {checkpoint_file}")

        return epoch, rollout_len, best_loss, encoder_learning_rule_dict, decoder_learning_rule_dict

    def my_collate_fn(self, batch: List[Any]) -> List[Any]:
        """Collate function for PyTorch DataLoader.

        Args:
            batch: Batch data

        Returns: 
            input, output and helpers in the format expected by DataLoader

        """
        _input, output, helpers = [], [], []

        for item in batch:
            _input.append(item[0])
            output.append(item[1])
            helpers.append(item[2])
        _input = torch.stack(_input)
        output = torch.stack(output)
        return [_input, output, helpers]

    def my_collate_fn_maml(self, batch: List[Any]) -> List[Any]:
        """Collate function for PyTorch DataLoader.

        Args:
            batch: Batch data

        Returns: 
            input, output and helpers in the format expected by DataLoader

        """
        support_input, support_output, train_input, train_output, helpers = [], [], [], [], []

        for item in batch:
            support_input.append(item[0])
            support_output.append(item[1])
            train_input.append(item[2])
            train_output.append(item[3])
            helpers.append(item[4])
        support_input = torch.stack(support_input)
        support_output = torch.stack(support_output)
        train_input = torch.stack(train_input)
        train_output = torch.stack(train_output)
        return [support_input, support_output, train_input, train_output, helpers]

    def init_hidden(self, batch_size: int,
                    hidden_size: int) -> Tuple[Any, Any]:
        """Get initial hidden state for LSTM.

        Args:
            batch_size: Batch size
            hidden_size: Hidden size of LSTM

        Returns:
            Initial hidden states

        """
        return (
            torch.zeros(batch_size, hidden_size).to(device),
            torch.zeros(batch_size, hidden_size).to(device),
        )
