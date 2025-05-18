from copy import deepcopy

import numpy as np

from constants.framework import (
    GLOBAL_MODELS_SAVING_PATH,
    ENCRYPTION_HOMOMORPHIC_XMKCKKS,
)
from constants.distances_constants import *
import pandas as pd
from sklearn.cluster import AffinityPropagation
import torch
import copy as py_copy
import os
from xmkckks import Rq
from typing import List
from core.federated_base import FederatedBase
from utils.log import Log

from utils.similarities.pairwise_coordinate_similarity import (
    pairwise_coordinate_similarity,
)
from utils.similarities.pairwise_cosine_similarity import pairwise_cosine_similarity
from utils.similarities.pairwise_euclidean_similarity import (
    pairwise_euclidean_similarity,
)
from validators.config_validator import ConfigValidator
from core.client import Client


class Server(FederatedBase):
    def __init__(self, model, config: "ConfigValidator", log: "Log"):
        super().__init__(model, config, log)

        # Use default PyTorch precision (float32)
        self.model = self.model.float()

        self.model_cache = []
        self.distance_metric = self.config.DISTANCE_METRIC

        self.rlwe = deepcopy(self.config.RUNTIME_CONFIG.rlwe)
        # Store cluster-specific RLWE vectors and keys
        self.cluster_rlwe_vectors = {}  # Maps cluster_idx to RLWE vector
        self.cluster_aggregated_keys = {}  # Maps cluster_idx to aggregated public key

    def _generate_cluster_rlwe_vector(self, cluster_idx):
        if (
            self.config.ENCRYPTION_METHOD is not None
            and self.config.ENCRYPTION_METHOD == ENCRYPTION_HOMOMORPHIC_XMKCKKS
        ):
            self.log.info(f"Generating RLWE vector for cluster {cluster_idx}")
            rlwe = deepcopy(self.rlwe)
            rlwe.generate_vector_a()
            self.cluster_rlwe_vectors[cluster_idx] = rlwe.get_vector_a()
            return rlwe.get_vector_a()
        else:
            self.log.info("Passing generating RLWE vector")
            return None

    def get_cluster_rlwe_vector(self, cluster_idx):
        if cluster_idx not in self.cluster_rlwe_vectors:
            return self._generate_cluster_rlwe_vector(cluster_idx)
        return self.cluster_rlwe_vectors[cluster_idx]

    def store_cluster_aggregated_pubkey(self, cluster_idx, aggregated_public_key):
        aggregated_pubkey = self.rlwe.list_to_poly(aggregated_public_key, "q")
        self.cluster_aggregated_keys[cluster_idx] = (
            aggregated_pubkey,
            self.cluster_rlwe_vectors[cluster_idx],
        )
        self.log.info(
            f"Server stored aggregated_public_key for cluster {cluster_idx}: {self.cluster_aggregated_keys[cluster_idx]}"
        )
        return True

    def compute_pairwise_similarities(self, clients):
        self.log.info(
            f"Start compute pairwise similarities with metric: {self.distance_metric}"
        )

        if self.distance_metric == DISTANCE_COSINE:
            return pairwise_cosine_similarity(
                clients, self.config.DISTANCE_METRIC_ON_PARAMETERS, self.log
            )
        elif self.distance_metric == DISTANCE_COORDINATE:
            return pairwise_coordinate_similarity(
                clients, self.config.REMOVE_COMMON_IDS, self.config, self.log
            )
        elif self.distance_metric == DISTANCE_EUCLIDEAN:
            return pairwise_euclidean_similarity(
                clients, self.config.DISTANCE_METRIC_ON_PARAMETERS, self.log
            )
        else:
            raise ValueError(f"unsupported distance metric {self.distance_metric}")

    def cluster_clients(self, similarities):

        self.log.info("similarity matrix is that feeds the clustering")
        similarity_df = pd.DataFrame(similarities)
        self.log.info("\n" + similarity_df.to_string())

        clustering = AffinityPropagation(
            affinity="precomputed",
            random_state=42,
        ).fit(similarities)

        self.log.info(f"Cluster labels: {clustering.labels_}")

        del similarities

        return clustering

    def aggregate(self, clients: List[Client], use_encryption: bool = False):
        self.log.info(f"models to be aggregated count: {len(clients)}")

        models = [client.model for client in clients]
        device = next(models[0].parameters()).device

        if use_encryption:
            aggregated_public_key = clients[0].aggregated_public_key

            # Calculate total parameters size
            total_params = sum(p.numel() for p in models[0].parameters())
            # Find next power of 2 that can accommodate all parameters
            target_size = 2 ** (total_params - 1).bit_length()
            self.log.info(
                f"Model total parameters: {total_params}, Padded size: {target_size}"
            )

            # Encrypt each model's parameters with proper precision
            encrypted_models = []
            for model in models:
                model.to(device)

                flat_params = []
                for param in model.parameters():
                    flat_params.extend(param.data.view(-1).tolist())

                # Scale parameters to fixed-point with xmkckks_weight_decimals precision
                scale = 10**self.config.XMKCKKS_WEIGHT_DECIMALS

                # Apply clipping to prevent extreme values
                flat_params = [max(min(x, 1e3), -1e3) for x in flat_params]
                flat_params = [int(x * scale) for x in flat_params]

                # Pad to next power of 2 to ensure consistent size
                if len(flat_params) < target_size:
                    flat_params.extend([0] * (target_size - len(flat_params)))

                m = Rq(np.array(flat_params), self.rlwe.t)
                encrypted = self.rlwe.encrypt(m, aggregated_public_key)
                encrypted_models.append(encrypted)

            # Add first model's coefficients
            csum0 = np.zeros(target_size, dtype=np.int64)
            csum1 = np.zeros(target_size, dtype=np.int64)

            coeffs0 = encrypted_models[0][0].poly.coeffs
            coeffs1 = encrypted_models[0][1].poly.coeffs
            csum0[: len(coeffs0)] = coeffs0
            csum1[: len(coeffs1)] = coeffs1

            # Add remaining models' coefficients
            for i in range(1, len(encrypted_models)):
                coeffs0 = encrypted_models[i][0].poly.coeffs
                coeffs1 = encrypted_models[i][1].poly.coeffs
                csum0[: len(coeffs0)] += coeffs0
                csum1[: len(coeffs1)] += coeffs1

            # Convert back to polynomials
            csum0 = Rq(csum0, self.config.RUNTIME_CONFIG.q)
            csum1 = Rq(csum1, self.config.RUNTIME_CONFIG.q)

            # Partial decryption from each client
            partial_decryptions = []
            for client in clients:
                partial_dec = client.compute_decryption_share(csum1.poly.coeffs)
                # Pad partial decryption to full size
                padded_dec = np.zeros(target_size, dtype=np.int64)
                padded_dec[: len(partial_dec)] = partial_dec
                partial_decryptions.append(padded_dec)

            # Final decryption
            dec_sum = np.zeros(target_size, dtype=np.int64)
            dec_sum[: len(csum0.poly.coeffs)] = csum0.poly.coeffs

            for partial_dec in partial_decryptions:
                dec_sum += partial_dec

            dec_sum = Rq(dec_sum, self.rlwe.t)

            # Scale back from fixed-point after decryption
            scale = 10**-self.config.XMKCKKS_WEIGHT_DECIMALS
            decrypted_params = [x * scale for x in dec_sum.poly.coeffs]

            # Add post-processing to filter out extreme values that might have occurred during encryption
            decrypted_params = [max(min(x, 1e3), -1e3) for x in decrypted_params]

            # Normalize by dividing by the number of clients
            decrypted_params = [x / len(clients) for x in decrypted_params]

            # Reconstruct the averaged model
            avg_model = py_copy.deepcopy(models[0])

            # Assign decrypted and averaged parameters back to the model
            pointer = 0
            with torch.no_grad():
                for param in avg_model.parameters():
                    param_size = param.numel()
                    param_data = decrypted_params[pointer : pointer + param_size]
                    param.data = torch.tensor(param_data, device=device).reshape(
                        param.shape
                    )
                    pointer += param_size

            return avg_model

        else:
            # Non-encrypted aggregation uses float32
            for model in models:
                model.to(device)
                model.float()
            avg_model = py_copy.deepcopy(models[0])

            with torch.no_grad():
                for param_name, param in avg_model.named_parameters():
                    param.data.zero_()
                    for model in models:
                        param.data.add_(
                            model.state_dict()[param_name].data / len(models)
                        )

            return avg_model

    def aggregate_clusterwise(self, client_clusters, use_encryption):
        for cluster_idx, cluster in enumerate(client_clusters):
            idcs = [client.id for client in cluster]
            self.log.info(f"Aggregating clients: {idcs}")

            avg_model = self.aggregate(clients=cluster, use_encryption=use_encryption)

            if self.config.SAVE_GLOBAL_MODELS:
                _path = f"{GLOBAL_MODELS_SAVING_PATH}/{self.config.MODEL_TYPE}"
                os.makedirs(_path, exist_ok=True)
                global_id = f"cluster{cluster_idx}_aggregated"
                model_path = f"{_path}/global_{global_id}.pth"
                torch.save(avg_model.state_dict(), model_path)
                self.log.info(
                    f"Saved aggregated model for cluster {cluster_idx} to {model_path}"
                )

            for client in cluster:
                client.model.load_state_dict(avg_model.state_dict())
                # client.optimizer = torch.optim.Adam(client.model.parameters(), lr=0.001,  amsgrad=True)
                # client.optimizer = torch.optim.SGD(client.model.parameters(),lr=0.001, momentum=0.9, weight_decay=1e-4)
                # client.optimizer = torch.optim.SGD(client.model.parameters(),lr=0.001, momentum=0.9)

    def _get_vector_a_list(self) -> List[int]:
        return self.rlwe_vector_a_poly.poly_to_list()
