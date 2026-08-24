"""torch_lhco.py - This file defines the LHCODataset class.
It is a modification of the original OmniLearn code that furnishes
pytorch data loaders instead of tensorflow data loaders.
It is used to train L-GATr models on the LHCO dataset.

Author: Kevin Greif
Last updated 6/2/25
python3
"""

import copy
import h5py
import numpy as np
import torch
from sklearn.utils import shuffle
from scipy.stats import norm


class LHCODataset:

    def __init__(self, path, rank=0, size=1, nevts=-1, **kwargs):
        """
        Initialize the LHCODataset.

        Args:
            path (str): Path to the LHCO data file.
            rank (int): Rank of the current process (for distributed training).
            size (int): Total number of processes (for distributed training).
            nevts (int): Number of events to load from the dataset.
                If negative, all events are loaded.
            **kwargs: Additional keyword arguments.
        """

        super().__init__()
        self.path = path
        self.rank = rank
        self.size = size
        assert self.rank < self.size, "Rank must be less than size"
        if nevts < 0:
            self.nevts = h5py.File(self.path, "r")["data"].shape[0]
        else:
            self.nevts = nevts

        # Load data from the HDF5 file
        self.jet = h5py.File(self.path, "r")["jet"][rank:int(self.nevts):size]
        self.X = h5py.File(self.path, "r")["data"][rank:int(self.nevts):size]
        # Min pT cut, require log(pT) > 0 or pT > 1 GeV
        self.X = self.X * (self.X[:, :, :, 3:4] > -0.0)
        self.mask = self.X[:, :, :, 2] != 0

        # Calculate px, py, pz, and E for each particle
        pt = np.exp(self.X[:, :, :, 3])
        E = np.exp(self.X[:, :, :, 5])
        px = pt * np.cos(self.X[:, :, :, 1] + self.jet[:, :, None, 2])
        py = pt * np.sin(self.X[:, :, :, 1] + self.jet[:, :, None, 2])
        pz = E * np.tanh(self.X[:, :, :, 0] + self.jet[:, :, None, 1])
        self.V = np.expand_dims(self.mask, -1) * np.stack([E, px, py, pz], -1)

        # For labels, we use the di-jet mass (mjj) as the target.
        if "pid" in h5py.File(self.path, "r"):
            self.raw_y = h5py.File(self.path, "r")["pid"][rank:int(self.nevts):size]
        else:
            self.raw_y = self.get_dimass(self.jet)

        self.y = self.prep_mjj(self.raw_y)

        # Initialize the weights to a vector of ones
        self.w = np.ones((self.nevts, 1), dtype=np.float32)

        # Define data shapes
        self.num_part = self.X.shape[2]
        self.num_feat = self.X.shape[3]
        self.num_jet = self.jet.shape[1]
        self.num_classes = 1

        # Labels will be set to 0 then later updated for signal events
        self.label = np.zeros((self.y.shape[0], 1))

        # Define statistics of the input distributions for normalization
        self.mean_part = [
            0.0,
            0.0,
            -0.022,
            2.13,
            -0.021,
            2.368,
            0.261,
        ]
        self.std_part = [
            0.25,
            0.25,
            0.066,
            1.429,
            0.065,
            1.44,
            0.235,
        ]

        self.mean_jet = [
            1.28724651e03,
            -4.81260266e-05,
            0.0,
            2.05052711e02,
            5.72253125e01,
        ]
        self.std_jet = [244.15460668, 0.74111563, 1.0, 151.10313677, 29.44343823]

    def make_ptdata(self):
        """make_ptdata - Generates a pytorch dataset from the LHCO data file.
        This runs preprocessing automatically since it's assumed to be used
        for input to some model.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset containing the LHCO data.
        """

        # Run preprocessing
        X = self.preprocess(self.X, self.mask).astype(np.float32)
        mask = self.mask
        V = self.V
        jet = self.add_noise(self.jet)
        jet = self.preprocess_jet(self.jet).astype(np.float32)

        # Pad scalar features so that the jet and constituent scalars
        # have the same dimension
        jet_extended = np.concatenate(
            (jet, np.zeros((*jet.shape[:-1], X.shape[-1]))),
            axis=-1,
        )
        X = np.concatenate(
            (np.zeros((*X.shape[:-1], jet.shape[-1])), X),
            axis=-1,
        )

        # Add a one hot distinguishing particle and jet tokens
        jet_extended = np.concatenate(
            (np.zeros((*jet_extended.shape[:-1], 1)), jet_extended),
            axis=-1,
        )
        X = np.concatenate((np.ones((*X.shape[:-1], 1)), X), axis=-1)

        # Add the jet information as an extra token for each jet
        X = np.concatenate((jet_extended[:, :, np.newaxis, :], X), axis=2)
        mask = np.concatenate(
            (np.ones((*jet_extended.shape[:-1], 1), dtype=bool), mask),
            axis=-1,
        )
        V = np.concatenate((np.sum(V, axis=2, keepdims=True), V), axis=2)

        # Add another one hot to distinguish the two jets in the event
        one_hot_jet = np.concatenate(
            [
                np.zeros((X.shape[0], 1, X.shape[2], 1)),  # 0s for first component
                np.ones((X.shape[0], 1, X.shape[2], 1)),  # 1s for second component
            ],
            axis=1,
        )
        X = np.concatenate([one_hot_jet, X], axis=-1)

        # Append event features to all particles within the event
        X_event = np.repeat(
            np.repeat(np.expand_dims(self.y, (1, 2, 3)), X.shape[1], 1),
            X.shape[2],
            2,
        )
        X = np.concatenate([X, X_event], -1)

        # Flatten off the jet dimension
        X, _ = self.flatten_event(X, mask)
        V, mask = self.flatten_event(V, mask)
        jet = jet.reshape((jet.shape[0], jet.shape[1] * jet.shape[2]))

        # Store simple per-token mask (no class token, no pairwise expansion)
        self.mask = mask

        # Add multivector dimension to V
        V = np.expand_dims(V, -2)

        # Build dataset
        dset = torch.utils.data.TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(V.astype(np.float32)),
            torch.from_numpy(self.mask.astype(np.float32)),
            torch.from_numpy(self.label.astype(np.float32)),
            torch.from_numpy(self.w.astype(np.float32)),
        )
        return dset

    def combine(self, datasets):
        """combine - Combines multiple datasets into a single dataset.
        Note the datasets added onto the original are assigned a label of 1.
        This method is meant to add signal datasets to the background dataset
        to create a combined dataset for training.

        Args:
            datasets (list): List of LHCODataset instances to combine.

        Returns:
            None: The method modifies the current instance in place.
        """

        for dataset in datasets:
            self.nevts += dataset.nevts
            self.X = np.concatenate([self.X, dataset.X], 0)
            self.V = np.concatenate([self.V, dataset.V], 0)
            self.mask = np.concatenate([self.mask, dataset.mask], 0)
            self.jet = np.concatenate([self.jet, dataset.jet], 0)
            self.label = np.concatenate(
                [self.label, np.ones((dataset.y.shape[0], 1))], 0
            )
            self.y = np.concatenate([self.y, dataset.y], 0)
            self.w = np.concatenate([self.w, dataset.w], 0)
        self.X, self.V, self.mask, self.jet, self.label, self.y, self.w = shuffle(
            self.X, self.V, self.mask, self.jet, self.label, self.y, self.w
        )

    def flatten_event(self, x, mask):
        """flatten_event - Flattens the event data (which is some numpy array
        with shape (num_events, 2, num_particles, num_features)) into a 3D
        array with shape (num_events, max_particles, num_features).

        Note the particles from the two jets will be listed one after the other
        in the flattened array, so the first half of the particles will be from
        the first jet and the second half will be from the second jet.

        Arguments:
            x (np.ndarray): Input data with shape
                (num_events, 2, num_particles, num_features).
            mask (np.ndarray): Mask indicating valid particles with shape
                (num_events, 2, num_particles).
        Returns:
            np.ndarray: Flattened input data with shape
                (num_events, max_particles, num_features).
            np.ndarray: Flattened mask with shape
                (num_events, max_particles).
        """

        N_events, _, N_parts, N_feats = x.shape

        # Flatten the event data
        flat_data = x.reshape(N_events, 2 * N_parts, N_feats)
        flat_mask = mask.reshape(N_events, 2 * N_parts)

        # Find the maximum number of particles across all events
        max_parts = np.max(np.sum(flat_mask, axis=1))

        # Create a new array with the shape (num_events, max_parts, num_features)
        out_data = np.zeros((N_events, max_parts, N_feats), dtype=x.dtype)
        out_mask = np.zeros((N_events, max_parts), dtype=mask.dtype)

        # Fill the new array with the flattened data
        for i in range(N_events):
            valid_indices = np.where(flat_mask[i, :] > 0)[0]
            out_data[i, : len(valid_indices), :] = flat_data[i, valid_indices, :]
            out_mask[i, : len(valid_indices)] = flat_mask[i, valid_indices]

        return out_data, out_mask

    # This method pre-processes the di-jet mass (mjj) into a normalized format.
    def prep_mjj(self, mjj, mjjmin=2300, mjjmax=5000):
        new_mjj = (mjj - mjjmin) / (mjjmax - mjjmin)
        # new_mjj = (np.log(mjj) - np.log(mjjmin))/(np.log(mjjmax) - np.log(mjjmin))
        new_mjj = 2 * new_mjj - 1.0
        return new_mjj.astype(np.float32)

    # Below are methods for preprocessing and reverting preprocessing
    def preprocess(self, x, mask):
        num_feat = x.shape[-1]
        return (
            np.expand_dims(mask, -1)
            * (x[:, :, :, :num_feat] - self.mean_part[:num_feat])
            / self.std_part[:num_feat]
        )

    def preprocess_jet(self, x):
        # Transform phi from uniform to gaussian
        new_x = copy.deepcopy(x)
        new_x[:, :, 2] = norm.ppf(0.5 * (1.0 + x[:, :, 2] / np.pi))
        return (new_x - self.mean_jet) / self.std_jet

    def revert_preprocess(self, x, mask):
        num_feat = x.shape[-1]
        new_part = np.expand_dims(mask, -1) * (
            x[:, :, :num_feat] * self.std_part[:num_feat] + self.mean_part[:num_feat]
        )
        # log pt rel and log e rel  should always be negative or 0
        new_part[:, :, 2] = np.minimum(new_part[:, :, 2], 0.0)
        return new_part

    def revert_preprocess_jet(self, x):
        new_x = self.std_jet * x + self.mean_jet
        # Recover phi
        new_x[:, :, 2] = np.pi * (2 * norm.cdf(new_x[:, :, 2]) - 1.0)
        new_x[:, :, 2] = np.clip(new_x[:, :, 2], -np.pi, np.pi)
        # Convert multiplicity back into integers
        new_x[:, :, -1] = np.round(new_x[:, :, -1])
        new_x[:, :, -1] = np.clip(new_x[:, :, -1], 2, self.num_part)
        return new_x

    # For calculating the di-jet mass (mjj) from the jet data, if ever needed
    def get_dimass(self, jets):
        jet_e = np.sqrt(
            jets[:, 0, 3] ** 2 + jets[:, 0, 0] ** 2 * np.cosh(jets[:, 0, 1]) ** 2
        )
        jet_e += np.sqrt(
            jets[:, 1, 3] ** 2 + jets[:, 1, 0] ** 2 * np.cosh(jets[:, 1, 1]) ** 2
        )
        jet_px = jets[:, 0, 0] * np.cos(jets[:, 0, 2]) + jets[:, 1, 0] * np.cos(
            jets[:, 1, 2]
        )
        jet_py = jets[:, 0, 0] * np.sin(jets[:, 0, 2]) + jets[:, 1, 0] * np.sin(
            jets[:, 1, 2]
        )
        jet_pz = jets[:, 0, 0] * np.sinh(jets[:, 0, 1]) + jets[:, 1, 0] * np.sinh(
            jets[:, 1, 1]
        )
        mjj = np.sqrt(np.abs(jet_px**2 + jet_py**2 + jet_pz**2 - jet_e**2))
        return mjj

    # This function adds some noise to the final dimension of an input tensor x.
    # It is used to simulate some variability in the jet multiplicity.
    def add_noise(self, x):
        # Add noise to the jet multiplicity
        noise = np.random.uniform(-0.5, 0.5, x.shape[0])
        x[:, :, -1] += noise[:, None]
        return x
