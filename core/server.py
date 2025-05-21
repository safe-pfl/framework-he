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
        if self.config.ENCRYPTION_METHOD == ENCRYPTION_HOMOMORPHIC_XMKCKKS:
            self.log.info("Server generating its main RLWE vector 'a'.")
            self.rlwe.generate_vector_a()
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
        self.log.info(f"Reusing existing RLWE vector for cluster {cluster_idx}")
        return self.cluster_rlwe_vectors[cluster_idx]

    def store_cluster_aggregated_pubkey(self, cluster_idx, aggregated_public_key):
        # Check if we already have the same key for this cluster
        if cluster_idx in self.cluster_aggregated_keys:
            existing_key = self.cluster_aggregated_keys[cluster_idx][0]
            # If keys are the same, no need to update
            if np.array_equal(
                existing_key.poly.coeffs,
                self.rlwe.list_to_poly(aggregated_public_key, "q").poly.coeffs,
            ):
                self.log.info(
                    f"Aggregated public key for cluster {cluster_idx} unchanged"
                )
                return True

        aggregated_pubkey = self.rlwe.list_to_poly(aggregated_public_key, "q")
        self.cluster_aggregated_keys[cluster_idx] = (
            aggregated_pubkey,
            self.cluster_rlwe_vectors[cluster_idx],
        )
        self.log.info(f"Server stored aggregated_public_key for cluster {cluster_idx}")
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

    def aggregate(
        self, clients: List[Client], use_encryption: bool = False, cluster_mask=None
    ):
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

            # Use provided cluster mask or combine importance masks from clients
            if cluster_mask is not None:
                combined_mask = cluster_mask
                self.log.info("Using provided cluster-specific importance mask")
            else:
                # Combine importance masks from all clients
                combined_mask = self.combine_importance_masks(clients)
                if combined_mask is None or len(combined_mask) < total_params:
                    # If no valid mask, create a mask that includes all parameters
                    combined_mask = np.ones(total_params, dtype=np.int32)
                    self.log.info(
                        "No valid importance masks found, using all parameters"
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

                # No aggressive clipping - use modest clipping only for numerical stability
                # This matches the approach in the reference code where clipping isn't as severe
                flat_params = [int(x * scale) for x in flat_params]

                # Pad to next power of 2 to ensure consistent size
                if len(flat_params) < target_size:
                    flat_params.extend([0] * (target_size - len(flat_params)))

                m = Rq(np.array(flat_params), self.rlwe.t)
                encrypted = self.rlwe.encrypt(m, aggregated_public_key)
                encrypted_models.append(encrypted)

            # Add first model's coefficients
            self.log.info("Starting to add model coefficients")
            csum0 = np.zeros(target_size, dtype=np.int64)
            csum1 = np.zeros(target_size, dtype=np.int64)

            coeffs0 = encrypted_models[0][0].poly.coeffs
            coeffs1 = encrypted_models[0][1].poly.coeffs
            csum0[: len(coeffs0)] = coeffs0
            csum1[: len(coeffs1)] = coeffs1
            self.log.info("Added first model coefficients")

            # Add remaining models' coefficients
            self.log.info(
                f"Adding remaining {len(encrypted_models)-1} models' coefficients"
            )
            for i in range(1, len(encrypted_models)):
                coeffs0 = encrypted_models[i][0].poly.coeffs
                coeffs1 = encrypted_models[i][1].poly.coeffs
                csum0[: len(coeffs0)] += coeffs0
                csum1[: len(coeffs1)] += coeffs1
            self.log.info("Finished adding all model coefficients")

            # Convert back to polynomials
            self.log.info("Starting conversion to Rq polynomials for csum0")
            csum0 = Rq(csum0, self.config.RUNTIME_CONFIG.q)
            self.log.info("Finished conversion to Rq for csum0, starting csum1")
            csum1 = Rq(csum1, self.config.RUNTIME_CONFIG.q)
            self.log.info("Finished conversion to Rq for csum1")

            # Partial decryption from each client
            self.log.info("Starting partial decryption process from clients")
            partial_decryptions = []
            for client_idx, client in enumerate(clients):
                self.log.info(
                    f"Computing decryption share for client {client_idx}/{len(clients)}"
                )
                partial_dec = client.compute_decryption_share(csum1.poly.coeffs)
                # Pad partial decryption to full size
                padded_dec = np.zeros(target_size, dtype=np.int64)
                padded_dec[: len(partial_dec)] = partial_dec
                partial_decryptions.append(padded_dec)
            self.log.info("Finished collecting all partial decryptions")

            # Final decryption
            self.log.info("Starting final decryption process")
            # Initialize a numpy array for the sum of decrypted parameters.
            # Using object type for dec_sum_accumulator to handle potentially very large intermediate integers if necessary,
            # though direct int64 summation followed by Rq modulo should also work if t fits in int64.
            # For safety, we ensure all numbers are treated correctly with respect to modulus t.
            dec_sum_accumulator = np.zeros(target_size, dtype=np.int64)

            # csum0 is Rq(..., q). Its coefficients are modulo q.
            # We need these coefficients modulo t for the plaintext reconstruction.
            plaintext_modulus_t = int(self.rlwe.t)

            self.log.info("Converting coefficients from modulo q to modulo t")
            csum0_coeffs_mod_q = csum0.poly.coeffs
            csum0_coeffs_mod_t = np.mod(csum0_coeffs_mod_q, plaintext_modulus_t)

            # Assign the (mod t) coefficients of c0_sum to the accumulator
            self.log.info("Assigning coefficients to accumulator")
            len_c0_coeffs = len(csum0_coeffs_mod_t)
            dec_sum_accumulator[:len_c0_coeffs] = csum0_coeffs_mod_t
            # Ensure remaining parts are zero if c0_coeffs_mod_t was shorter than target_size (should not happen if padding is consistent)
            if len_c0_coeffs < target_size:
                dec_sum_accumulator[len_c0_coeffs:] = 0

            # Add partial decryptions (which are already modulo t and padded to target_size)
            self.log.info("Adding partial decryptions to accumulator")
            for dec_idx, partial_dec_coeffs_arr in enumerate(partial_decryptions):
                self.log.info(
                    f"Adding partial decryption {dec_idx+1}/{len(partial_decryptions)}"
                )
                dec_sum_accumulator = np.add(
                    dec_sum_accumulator, partial_dec_coeffs_arr
                )
                # We can take modulo t here at each step to keep numbers smaller, or at the end via Rq.
                # Rq will handle the final modulo, so direct sum is fine if it doesn't overflow int64 significantly
                # before Rq constructor (which it shouldn't if t fits int64).

            # The dec_sum_accumulator now contains the sum of (c0_coeffs mod t) + sum(partial_shares mod t).
            # This sum itself needs to be interpreted modulo t.
            self.log.info("Creating final Rq object from accumulator")
            dec_sum = Rq(dec_sum_accumulator, plaintext_modulus_t)
            self.log.info("Finished creating final Rq object")

            # Scale back from fixed-point after decryption
            self.log.info("Scaling back parameters from fixed-point representation")
            scale = 10**-self.config.XMKCKKS_WEIGHT_DECIMALS
            decrypted_params = [x * scale for x in dec_sum.poly.coeffs]
            self.log.info("Finished scaling parameters")

            # Normalize by dividing by the number of clients without aggressive clipping
            self.log.info("Normalizing encrypted parameters by client count")
            decrypted_params = [x / len(clients) for x in decrypted_params]
            self.log.info("Finished encrypted parameter normalization")

            # For non-encrypted parameters, use standard averaging
            self.log.info("Processing non-encrypted parameters")
            non_encrypted_params = np.zeros(total_params)
            if np.any(combined_mask == 0):  # If there are any non-encrypted parameters
                non_encrypted_indices = np.where(combined_mask == 0)[0]
                self.log.info(
                    f"Found {len(non_encrypted_indices)} non-encrypted parameter indices"
                )

                # Vectorized approach for non-encrypted parameters
                self.log.info("Using vectorized approach for non-encrypted parameters")

                # First, flatten all models' parameters at once
                all_flat_params = []
                for model in models:
                    # Flatten parameters for this model
                    model_params = []
                    for param in model.parameters():
                        model_params.extend(param.data.view(-1).cpu().numpy())
                    all_flat_params.append(model_params)

                # Convert to numpy array for vectorized operations
                all_flat_params = np.array(
                    all_flat_params
                )  # shape: [num_models, num_params]

                # Calculate average directly for all non-encrypted parameters
                # Sum across models (axis 0) and divide by number of models
                non_encrypted_params_all = np.mean(all_flat_params, axis=0)

                # Assign only the non-encrypted parameters
                for idx in non_encrypted_indices:
                    if idx < len(non_encrypted_params_all):
                        non_encrypted_params[idx] = non_encrypted_params_all[idx]

                self.log.info(
                    "Finished vectorized processing of non-encrypted parameters"
                )
            self.log.info("Finished processing non-encrypted parameters")

            # Reconstruct the averaged model
            self.log.info("Reconstructing averaged model")
            avg_model = py_copy.deepcopy(models[0])

            # Assign decrypted and averaged parameters back to the model
            self.log.info("Assigning parameters to model")
            pointer = 0
            with torch.no_grad():
                for param_idx, param in enumerate(avg_model.parameters()):
                    if param_idx % 10 == 0:
                        self.log.info(f"Processing parameter layer {param_idx}")
                    param_size = param.numel()
                    # Get the relevant slice of parameters for this layer
                    param_slice = slice(pointer, pointer + param_size)

                    # Create a tensor to hold the aggregated parameters
                    param_data = np.zeros(param_size)

                    # Fill with encrypted parameters where mask is 1
                    encrypted_indices = np.where(combined_mask[param_slice] == 1)[0]
                    if len(encrypted_indices) > 0:
                        self.log.info(
                            f"Layer {param_idx}: Assigning {len(encrypted_indices)} encrypted parameters"
                        )
                        for i, idx in enumerate(encrypted_indices):
                            if pointer + idx < len(decrypted_params):
                                param_data[idx] = decrypted_params[pointer + idx]

                    # Fill with non-encrypted parameters where mask is 0
                    non_encrypted_indices = np.where(combined_mask[param_slice] == 0)[0]
                    if len(non_encrypted_indices) > 0:
                        self.log.info(
                            f"Layer {param_idx}: Assigning {len(non_encrypted_indices)} non-encrypted parameters"
                        )
                        for idx in non_encrypted_indices:
                            param_data[idx] = non_encrypted_params[pointer + idx]

                    # Update the model parameter
                    param.data = torch.tensor(param_data, device=device).reshape(
                        param.shape
                    )
                    pointer += param_size

            self.log.info("Finished model reconstruction. Aggregation complete.")
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
        # If using encryption, analyze cluster masks first
        cluster_masks = {}
        if (
            use_encryption
            and self.config.ENCRYPTION_METHOD == ENCRYPTION_HOMOMORPHIC_XMKCKKS
        ):
            self.log.info("Analyzing importance masks across clusters...")
            combined_mask, cluster_masks, mask_analysis = self.analyze_cluster_masks(
                client_clusters
            )

            # Log some key statistics about the masks
            if combined_mask is not None:
                total_params = mask_analysis["total_parameters"]
                combined_important = mask_analysis["combined_important_count"]
                self.log.info(
                    f"Overall, {combined_important}/{total_params} parameters "
                    f"({combined_important/total_params*100:.2f}%) are considered important by at least one cluster"
                )

                # Check if clusters have significantly different masks
                if len(client_clusters) > 1:
                    avg_jaccard = 0.0
                    count = 0
                    for analysis in mask_analysis["intersection_analysis"].values():
                        avg_jaccard += analysis["jaccard_similarity"]
                        count += 1

                    if count > 0:
                        avg_jaccard /= count
                        self.log.info(
                            f"Average Jaccard similarity between cluster masks: {avg_jaccard:.4f}"
                        )

                        if avg_jaccard < 0.5:
                            self.log.info(
                                "Clusters have significantly different important parameters (Jaccard < 0.5)"
                            )
                        elif avg_jaccard < 0.8:
                            self.log.info(
                                "Clusters have moderately different important parameters (0.5 ≤ Jaccard < 0.8)"
                            )
                        else:
                            self.log.info(
                                "Clusters have similar important parameters (Jaccard ≥ 0.8)"
                            )

                # Visualize the cluster masks
                if len(cluster_masks) > 0:
                    # Create directory for visualizations if it doesn't exist
                    import os

                    vis_dir = f"visualizations/{self.config.MODEL_TYPE}"
                    os.makedirs(vis_dir, exist_ok=True)

                    # Create visualization filename with round number if available
                    round_num = getattr(self, "current_round", 0)
                    vis_path = f"{vis_dir}/cluster_masks_round_{round_num}.png"

                    # Generate and save the visualization
                    self.visualize_cluster_masks(cluster_masks, save_path=vis_path)

        # Proceed with aggregation for each cluster
        for cluster_idx, cluster in enumerate(client_clusters):
            idcs = [client.id for client in cluster]
            self.log.info(f"Aggregating clients: {idcs}")

            # Use the cluster-specific mask if available
            cluster_specific_mask = (
                cluster_masks.get(cluster_idx) if cluster_masks else None
            )

            avg_model = self.aggregate(
                clients=cluster,
                use_encryption=use_encryption,
                cluster_mask=cluster_specific_mask,
            )

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

    def combine_importance_masks(self, clients, cluster_idx=None):
        """
        Combines importance masks from multiple clients into a single mask.
        The combined mask will have 1s for parameters that are important to any client.

        If cluster_idx is provided, this is part of a cluster-specific analysis.
        """
        if not clients:
            return None

        # Get the first client's mask to determine size
        first_client = clients[0]
        if first_client.importance_mask is None:
            # If no masks are available, return None
            return None

        mask_size = len(first_client.importance_mask)
        combined_mask = np.zeros(mask_size, dtype=np.int32)

        # Combine masks using logical OR
        for client in clients:
            if client.importance_mask is not None:
                # Make sure masks are the same size
                if len(client.importance_mask) != mask_size:
                    self.log.warn(
                        f"Client {client.id} has mask of different size. Adjusting..."
                    )
                    if len(client.importance_mask) < mask_size:
                        # Extend shorter mask
                        extended_mask = np.zeros(mask_size, dtype=np.int32)
                        extended_mask[: len(client.importance_mask)] = (
                            client.importance_mask
                        )
                        client.importance_mask = extended_mask
                    else:
                        # Truncate longer mask
                        client.importance_mask = client.importance_mask[:mask_size]

                # Combine masks
                combined_mask = np.logical_or(combined_mask, client.importance_mask)

        # Count parameters in combined mask
        important_count = np.sum(combined_mask)
        if cluster_idx is not None:
            self.log.info(
                f"Cluster {cluster_idx} mask includes {important_count}/{mask_size} parameters ({important_count/mask_size*100:.2f}%)"
            )
        else:
            self.log.info(
                f"Combined mask includes {important_count}/{mask_size} parameters ({important_count/mask_size*100:.2f}%)"
            )

        return combined_mask

    def analyze_cluster_masks(self, client_clusters):
        """
        Analyzes the importance masks across different clusters.
        Determines each cluster's mask separately and analyzes their intersections.

        Returns:
            combined_mask: The combined mask across all clusters
            cluster_masks: Dictionary mapping cluster_idx to that cluster's mask
            analysis: Dictionary containing analysis results
        """
        if not client_clusters:
            return None, {}, {}

        # Get first client to determine mask size
        first_client = client_clusters[0][0]
        if first_client.importance_mask is None:
            return None, {}, {}

        mask_size = len(first_client.importance_mask)

        # Generate mask for each cluster
        cluster_masks = {}
        for cluster_idx, cluster in enumerate(client_clusters):
            cluster_masks[cluster_idx] = self.combine_importance_masks(
                cluster, cluster_idx
            )

        # Create combined mask across all clusters
        combined_mask = np.zeros(mask_size, dtype=np.int32)
        for cluster_mask in cluster_masks.values():
            combined_mask = np.logical_or(combined_mask, cluster_mask)

        # Analyze intersections and differences
        analysis = {
            "total_parameters": mask_size,
            "combined_important_count": np.sum(combined_mask),
            "cluster_important_counts": {},
            "intersection_analysis": {},
            "unique_parameters": {},
        }

        # Count important parameters per cluster
        for cluster_idx, mask in cluster_masks.items():
            analysis["cluster_important_counts"][cluster_idx] = np.sum(mask)

        # Analyze pairwise intersections between clusters
        if len(cluster_masks) > 1:
            for i in range(len(cluster_masks)):
                for j in range(i + 1, len(cluster_masks)):
                    mask_i = cluster_masks[i]
                    mask_j = cluster_masks[j]

                    # Calculate intersection
                    intersection = np.logical_and(mask_i, mask_j)
                    intersection_count = np.sum(intersection)

                    # Calculate Jaccard similarity
                    union = np.logical_or(mask_i, mask_j)
                    union_count = np.sum(union)
                    jaccard = intersection_count / union_count if union_count > 0 else 0

                    analysis["intersection_analysis"][f"{i}_and_{j}"] = {
                        "intersection_count": int(intersection_count),
                        "intersection_percentage": float(
                            intersection_count / mask_size * 100
                        ),
                        "jaccard_similarity": float(jaccard),
                    }

                    self.log.info(
                        f"Clusters {i} and {j} have {intersection_count} parameters in common "
                        + f"({intersection_count/mask_size*100:.2f}% of total, Jaccard: {jaccard:.4f})"
                    )

        # Find unique parameters for each cluster
        for i, mask_i in cluster_masks.items():
            # Start with this cluster's mask
            unique_mask = mask_i.copy()

            # Remove parameters that appear in other clusters
            for j, mask_j in cluster_masks.items():
                if i != j:
                    # Remove parameters that also appear in cluster j
                    unique_mask = np.logical_and(unique_mask, np.logical_not(mask_j))

            unique_count = np.sum(unique_mask)
            analysis["unique_parameters"][i] = {
                "count": int(unique_count),
                "percentage": (
                    float(unique_count / np.sum(mask_i) * 100)
                    if np.sum(mask_i) > 0
                    else 0
                ),
            }

            self.log.info(
                f"Cluster {i} has {unique_count} unique important parameters "
                + f"({unique_count/np.sum(mask_i)*100:.2f}% of its important parameters)"
            )

        return combined_mask, cluster_masks, analysis

    def visualize_cluster_masks(self, cluster_masks, save_path=None):
        """
        Visualizes the importance masks for different clusters.

        Args:
            cluster_masks: Dictionary mapping cluster_idx to that cluster's mask
            save_path: Path to save the visualization. If None, will use a default path.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            import os

            if not cluster_masks:
                self.log.warn("No cluster masks to visualize")
                return

            # Determine the number of clusters and mask size
            num_clusters = len(cluster_masks)
            mask_size = len(next(iter(cluster_masks.values())))

            # If mask is too large, sample it
            max_display_size = 1000
            sample_rate = 1
            if mask_size > max_display_size:
                sample_rate = mask_size // max_display_size + 1
                self.log.info(
                    f"Mask is large ({mask_size} parameters), sampling every {sample_rate}th parameter"
                )

            # Create a figure with subplots for each cluster and one for the combined mask
            fig, axs = plt.subplots(
                num_clusters + 1, 1, figsize=(15, 3 * (num_clusters + 1))
            )

            # Define colors for each cluster
            colors = [
                "red",
                "blue",
                "green",
                "purple",
                "orange",
                "brown",
                "pink",
                "gray",
                "olive",
                "cyan",
            ]
            cluster_colors = colors[:num_clusters]

            # Create a custom colormap for the combined visualization
            # 0: white (unimportant), 1-N: cluster colors
            cmap_colors = ["white"] + cluster_colors
            combined_cmap = ListedColormap(cmap_colors)

            # Plot each cluster's mask
            for i, (cluster_idx, mask) in enumerate(sorted(cluster_masks.items())):
                # Sample the mask if needed
                sampled_mask = mask[::sample_rate]

                # Create a colormap for this cluster: white for 0, cluster color for 1
                cmap = ListedColormap(["white", cluster_colors[i]])

                # Reshape for better visualization (make it 2D)
                width = int(np.sqrt(len(sampled_mask)))
                height = len(sampled_mask) // width + (
                    1 if len(sampled_mask) % width > 0 else 0
                )
                padded_mask = np.zeros(width * height)
                padded_mask[: len(sampled_mask)] = sampled_mask
                reshaped_mask = padded_mask.reshape(height, width)

                # Plot
                im = axs[i].imshow(reshaped_mask, cmap=cmap, aspect="auto")
                axs[i].set_title(f"Cluster {cluster_idx} Importance Mask", pad=20)

                # Add axis labels
                axs[i].set_xlabel("Parameter Index (Sampled)", labelpad=10)
                axs[i].set_ylabel("Layer Group", labelpad=10)

                # Add grid lines for layers
                axs[i].grid(True, linestyle="--", alpha=0.3)
                axs[i].set_xticks(np.arange(0, width, width // 10))
                axs[i].set_yticks(np.arange(0, height, height // 5))

                # Add colorbar
                cbar = plt.colorbar(im, ax=axs[i])
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(["Unimportant", "Important"])

                # Add text showing percentage of important parameters
                important_pct = np.sum(mask) / len(mask) * 100
                axs[i].text(
                    0.02,
                    0.95,
                    f"{important_pct:.2f}% important",
                    transform=axs[i].transAxes,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.7),
                )

            # Create combined mask with cluster-specific values
            combined_mask = np.zeros(mask_size, dtype=np.int32)
            for cluster_idx, mask in cluster_masks.items():
                # Assign cluster-specific values (1-based for colormap)
                combined_mask[mask == 1] = cluster_idx + 1

            # Sample the combined mask
            sampled_combined = combined_mask[::sample_rate]
            width = int(np.sqrt(len(sampled_combined)))
            height = len(sampled_combined) // width + (
                1 if len(sampled_combined) % width > 0 else 0
            )
            padded_combined = np.zeros(width * height)
            padded_combined[: len(sampled_combined)] = sampled_combined
            reshaped_combined = padded_combined.reshape(height, width)

            # Plot the combined mask
            im = axs[-1].imshow(reshaped_combined, cmap=combined_cmap, aspect="auto")
            axs[-1].set_title("Combined Importance Mask Across All Clusters", pad=20)

            # Add axis labels for combined plot
            axs[-1].set_xlabel("Parameter Index (Sampled)", labelpad=10)
            axs[-1].set_ylabel("Layer Group", labelpad=10)

            # Add grid lines for layers in combined plot
            axs[-1].grid(True, linestyle="--", alpha=0.3)
            axs[-1].set_xticks(np.arange(0, width, width // 10))
            axs[-1].set_yticks(np.arange(0, height, height // 5))

            # Add legend for the combined plot
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black")
                for color in cluster_colors
            ]
            axs[-1].legend(
                legend_elements,
                [f"Cluster {i}" for i in range(num_clusters)],
                loc="upper right",
                bbox_to_anchor=(1.15, 1),
                title="Cluster Colors",
            )

            # Add text showing total percentage of important parameters
            total_important = np.sum(combined_mask > 0) / len(combined_mask) * 100
            axs[-1].text(
                0.02,
                0.95,
                f"{total_important:.2f}% total important",
                transform=axs[-1].transAxes,
                color="black",
                bbox=dict(facecolor="white", alpha=0.7),
            )

            # Add a main title to the figure
            fig.suptitle(
                "Parameter Importance Distribution Across Clusters\n"
                "White: Unimportant Parameters, Colored: Important Parameters",
                y=0.98,
                fontsize=14,
            )

            # Adjust layout to prevent title overlap
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Save the figure if a path is provided
            if save_path is None:
                save_path = f"cluster_masks_{len(cluster_masks)}_clusters.png"

            # Ensure directory exists
            os.makedirs(
                os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True,
            )
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self.log.info(f"Saved cluster mask visualization to {save_path}")

            plt.close(fig)

        except ImportError:
            self.log.warn("Matplotlib not available, skipping visualization")
        except Exception as e:
            self.log.warn(f"Error visualizing cluster masks: {str(e)}")
            import traceback

            self.log.warn(traceback.format_exc())
