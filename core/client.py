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

        self.log.info(f"client no: {self.id} initialized")

    # Step 1) Server sends shared vector_a to clients and they all send back vector_b
    def generate_pubkey(
        self, vector_a: List[int], cluster_idx: int = None
    ) -> List[int]:
        vector_a = self.rlwe.list_to_poly(vector_a, "q")
        self.rlwe.set_vector_a(vector_a)

        (self.secret_key, self.public_key) = self.rlwe.generate_keys()
        self.current_cluster_idx = cluster_idx

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

        # Pad list until length 2**20 with random numbers that mimic the weights
        flattened_weights, self.model_size = pad_to_power_of_2(
            flat_weights, self.rlwe.n, self.config.XMKCKKS_WEIGHT_DECIMALS
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
        # print(f"client {self.id} c0 (first 10): {c0[:10]}")
        # print(f"client {self.id} c1 (first 10): {c1[:10]}")
        return c0, c1

    # Step 4) Use csum1 to calculate partial decryption share di
    def compute_decryption_share(self, csum1) -> List[int]:
        if self.secret_key is None:
            raise ValueError(f"Client {self.id} has no secret key for decryption")

        std = GAUSSIAN_DISTRIBUTION + 2  # Larger variance for security & error coverage
        csum1_poly = self.rlwe.list_to_poly(csum1, "q")
        error = Rq(
            np.round(std * np.random.randn(self.rlwe.n)), self.config.RUNTIME_CONFIG.q
        )
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

        epochs = self.config.NUMBER_OF_EPOCHS if epochs is None else epochs
        _updated_model, train_stats = train(
            self.model,
            self.train_loader if not loader else loader,
            self.optimizer,
            epochs,
            self.device,
            self.log,
        )

        self.model.load_state_dict(_updated_model.state_dict())
        del _updated_model

        self.log.info(
            f"training done for client no {self.id} with loss of {train_stats}"
        )

        if be_ready_for_clustering:
            criterion = torch.nn.CrossEntropyLoss().to(
                device=self.device, non_blocking=True
            )

            _model = py_copy.deepcopy(self.model)
            _model.eval()

            accumulated_grads = []
            for param in _model.parameters():
                if param.requires_grad:
                    accumulated_grads.append(
                        torch.zeros_like(param, device=self.device)
                    )
                else:
                    accumulated_grads.append(None)

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = _model(inputs)
                loss = criterion(outputs, labels.long())

                grads = torch.autograd.grad(
                    loss, _model.parameters(), allow_unused=True
                )

                for i, grad in enumerate(grads):
                    if grad is not None:
                        accumulated_grads[i] += grad.detach().abs()

            all_grads = []
            for grad in accumulated_grads:
                if grad is not None:
                    all_grads.append(grad.view(-1).cpu())

            if all_grads:
                combined_grads = torch.cat(all_grads).numpy()
                self.gradients = {i: val for i, val in enumerate(combined_grads)}
                self.log.info(f"Gradients computed with {len(self.gradients)} entries.")
            else:
                self.log.warn("No gradients were computed.")
                self.gradients = {}

            del _model
        return train_stats
