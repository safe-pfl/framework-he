from sympy import public
from typing import Tuple, List
from xmkckks import Rq
import numpy as np
from constants.framework import GAUSSIAN_DISTRIBUTION

from core.federated_base import FederatedBase
from utils.log import Log
from validators.config_validator import ConfigValidator
import copy as py_copy
from core.train import train
import torch
from typing import List
from copy import deepcopy
from utils.encryption.pad_to_power_of_2 import pad_to_power_of_2
from heapq import nlargest


class Client(FederatedBase):
    def __init__(
        self,
        model,
        optimizer_fn,
        id_num,
        train_data_loader,
        evaluation_data_loader,
        config: "ConfigValidator",
        log: "Log",
    ):
        super().__init__(model, config, log)

        # Use default PyTorch precision (float32) for training
        self.model = self.model.float()  # Ensure model is in float32

        self.optimizer = optimizer_fn(self.model.parameters())

        self.model_shape = None
        self.model_size = None
        self.train_loader = train_data_loader
        self.eval_loader = evaluation_data_loader

        self.gradients = {}
        self.id = id_num

        self.rlwe = deepcopy(self.config.RUNTIME_CONFIG.rlwe)
        self.public_key = None
        self.secret_key = None
        self.aggregated_public_key = None
        self.current_cluster_idx = None
        self.importance_mask = None

        # Store keys for each cluster to avoid regenerating them
        self.cluster_keys = {}  # Maps cluster_idx to (secret_key, public_key) tuple

        self.log.info(f"client no: {self.id} initialized")

    # Step 1) Server sends shared vector_a to clients and they all send back vector_b
    def generate_pubkey(
        self, vector_a: List[int], cluster_idx: int = None
    ) -> List[int]:
        # Check if we already have keys for this cluster
        if cluster_idx in self.cluster_keys:
            self.secret_key, self.public_key = self.cluster_keys[cluster_idx]
            self.current_cluster_idx = cluster_idx
            self.log.info(
                f"client id: {self.id} reusing existing public key for cluster {cluster_idx}"
            )
            return self.public_key[0].poly_to_list()

        vector_a = self.rlwe.list_to_poly(vector_a, "q")
        self.rlwe.set_vector_a(vector_a)

        (self.secret_key, self.public_key) = self.rlwe.generate_keys()
        self.current_cluster_idx = cluster_idx

        # Store the keys for this cluster
        self.cluster_keys[cluster_idx] = (self.secret_key, self.public_key)

        self.log.info(
            f"client id: {self.id} generated public key for cluster {cluster_idx}"
        )
        self.log.info(self.public_key)

        return self.public_key[0].poly_to_list()

    # Step 2) Server sends aggregated publickey aggregated_public_key to clients and receive boolean confirmation
    def store_aggregated_pubkey(self, aggregated_public_key: List[int]) -> bool:
        aggregated_pubkey = self.rlwe.list_to_poly(aggregated_public_key, "q")
        self.aggregated_public_key = (aggregated_pubkey, self.rlwe.get_vector_a())
        self.log.info(
            f"client id: {self.id} stored aggregated_public_key for cluster {self.current_cluster_idx}"
        )
        return True

    def create_importance_mask(self) -> np.ndarray:
        """
        Creates a binary mask based on the sensitivity percentage and gradients.
        The mask will have 1s for the most important parameters and 0s for the rest.
        """
        if not self.gradients:
            self.log.warn(
                "No gradients available for creating importance mask. Using all parameters."
            )
            # If no gradients available, use all parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            return np.ones(total_params, dtype=np.int32)

        # Calculate how many parameters to keep based on sensitivity percentage
        total_params = len(self.gradients)
        params_to_keep = int(
            np.ceil(self.config.SENSITIVITY_PERCENTAGE * total_params / 100)
        )

        # Get the indices of the most important parameters
        grads = self.gradients.items()
        important_indices = [
            k for k, _ in nlargest(params_to_keep, grads, key=lambda x: x[1])
        ]

        # Create binary mask
        mask = np.zeros(total_params, dtype=np.int32)
        mask[important_indices] = 1

        self.log.info(
            f"Created importance mask with {params_to_keep}/{total_params} parameters ({self.config.SENSITIVITY_PERCENTAGE}%)"
        )
        return mask

    # Step 3) After round, encrypt flat list of parameters into two lists (c0, c1)
    def encrypt_parameters(self) -> Tuple[List[int], List[int]]:
        if self.aggregated_public_key is None:
            raise ValueError(
                f"Client {self.id} has no aggregated public key for encryption"
            )

        # Flatten model parameters using PyTorch
        flat_weights = torch.cat(
            [param.data.view(-1) for param in self.model.parameters()]
        )
        self.model_shape = [param.size() for param in self.model.parameters()]

        # Create or update importance mask based on gradients if needed
        if self.importance_mask is None or len(self.importance_mask) != len(
            flat_weights
        ):
            self.importance_mask = self.create_importance_mask()
            self.log.info(f"Created new importance mask for client {self.id}")

        # Convert to numpy for easier manipulation
        flat_weights_np = flat_weights.cpu().numpy()
        total_params = len(flat_weights_np)

        # Apply mask: only encrypt important parameters
        masked_weights = np.zeros_like(flat_weights_np)

        # If mask is shorter than weights (due to padding in previous steps), extend it
        if len(self.importance_mask) < total_params:
            extended_mask = np.zeros(total_params, dtype=np.int32)
            extended_mask[: len(self.importance_mask)] = self.importance_mask
            self.importance_mask = extended_mask

        # Apply the mask - keep important parameters, zero out unimportant ones
        important_indices = np.where(self.importance_mask == 1)[0]
        masked_weights[important_indices] = flat_weights_np[important_indices]

        # Convert back to torch tensor
        masked_flat_weights = torch.tensor(masked_weights, device=flat_weights.device)

        # Count how many parameters we're actually encrypting
        important_count = np.sum(self.importance_mask)
        self.log.info(
            f"Client {self.id} encrypting {important_count}/{total_params} parameters ({important_count/total_params*100:.2f}%)"
        )

        # Pad list until length 2**20 with random numbers that mimic the weights
        flattened_weights, self.model_size = pad_to_power_of_2(
            masked_flat_weights, self.rlwe.n, self.config.XMKCKKS_WEIGHT_DECIMALS
        )
        self.log.info(
            f"Client {self.id} encrypting parameters for cluster {self.current_cluster_idx}"
        )

        # Turn list into polynomial
        poly_weights = Rq(np.array(flattened_weights.cpu()), self.rlwe.t)

        # Encrypt the polynomial using cluster's aggregated key
        c0, c1 = self.rlwe.encrypt(poly_weights, self.aggregated_public_key)
        c0 = list(c0.poly.coeffs)
        c1 = list(c1.poly.coeffs)
        return c0, c1

    # Step 4) Use csum1 to calculate partial decryption share di
    def compute_decryption_share(self, csum1) -> List[int]:
        if self.secret_key is None:
            raise ValueError(f"Client {self.id} has no secret key for decryption")

        # Use the same std as in RLWE initialization (3)
        std = 3
        csum1_poly = self.rlwe.list_to_poly(csum1, "q")

        # Generate error with controlled magnitude
        error_array = np.random.randn(self.rlwe.n)
        # Clip extreme values in the error
        error_array = np.clip(error_array, -2, 2) * std

        error = Rq(np.round(error_array), self.config.RUNTIME_CONFIG.q)
        decryption_share = self.rlwe.decrypt(csum1_poly, self.secret_key, error)
        decryption_share = list(
            decryption_share.poly.coeffs
        )  # decryption_share is poly_t not poly_q
        return decryption_share

    def synchronize_with_server(self, server):
        self.model.load_state_dict(server.model.state_dict())

    def compute_weight_update(
        self,
        be_ready_for_clustering,
        epochs=None,
        loader=None,
    ):
        # Add gradient clipping to optimizer
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        epochs = self.config.NUMBER_OF_EPOCHS if epochs is None else epochs

        # If we need gradients for clustering, use the track_gradients option
        if be_ready_for_clustering:
            _updated_model, train_stats, accumulated_grads = train(
                self.model,
                self.train_loader if not loader else loader,
                self.optimizer,
                epochs,
                self.device,
                self.log,
                track_gradients=True,
            )

            # Process the accumulated gradients
            all_grads = []
            for grad in accumulated_grads:
                if grad is not None:
                    all_grads.append(grad.view(-1).cpu())

            if all_grads:
                combined_grads = torch.cat(all_grads).numpy()
                self.gradients = {i: val for i, val in enumerate(combined_grads)}
                self.log.info(f"Gradients computed with {len(self.gradients)} entries.")

                # Create importance mask based on these gradients
                self.importance_mask = self.create_importance_mask()
                self.log.info(f"Created importance mask for client {self.id}")
            else:
                self.log.warn("No gradients were computed.")
                self.gradients = {}
        else:
            _updated_model, train_stats = train(
                self.model,
                self.train_loader if not loader else loader,
                self.optimizer,
                epochs,
                self.device,
                self.log,
                track_gradients=False,
            )

        self.model.load_state_dict(_updated_model.state_dict())
        del _updated_model

        self.log.info(
            f"training done for client no {self.id} with loss of {train_stats}"
        )

        return train_stats
