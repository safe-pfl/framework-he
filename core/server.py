from constants.framework import GLOBAL_MODELS_SAVING_PATH, ENCRYPTION_HOMOMORPHIC_XMKCKKS
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

from utils.similarities.pairwise_coordinate_similarity import pairwise_coordinate_similarity
from utils.similarities.pairwise_cosine_similarity import pairwise_cosine_similarity
from utils.similarities.pairwise_euclidean_similarity import pairwise_euclidean_similarity
from validators.config_validator import ConfigValidator


class Server(FederatedBase):
    def __init__(self, model, config: 'ConfigValidator', log: 'Log'):
        super().__init__(model, config, log)

        self.model_cache = []
        self.distance_metric = self.config.DISTANCE_METRIC

        self.rlwe_vector_a_poly: Rq = self._generate_rlwe_vector()
        self.rlwe_vector_a_list: List[int] = self._get_vector_a_list()

    def _generate_rlwe_vector(self):
        if self.config.ENCRYPTION_METHOD is not None and self.config.ENCRYPTION_METHOD == ENCRYPTION_HOMOMORPHIC_XMKCKKS:
            self.log.info('generating RLWE vector')
            self.config.RUNTIME_COMFIG.rlwe.generate_vector_a()
            return self.config.RUNTIME_COMFIG.rlwe.get_vector_a()
        else:
            self.log.info('passing generating RLWE vector')
            return None

    def compute_pairwise_similarities(self, clients):
        self.log.info(f"Start compute pairwise similarities with metric: {self.distance_metric}")

        if self.distance_metric == DISTANCE_COSINE:
            return pairwise_cosine_similarity(clients, self.config.DISTANCE_METRIC_ON_PARAMETERS, self.log)
        elif self.distance_metric == DISTANCE_COORDINATE:
            return pairwise_coordinate_similarity(clients, self.config.REMOVE_COMMON_IDS, self.config, self.log)
        elif self.distance_metric == DISTANCE_EUCLIDEAN:
            return pairwise_euclidean_similarity(clients, self.config.DISTANCE_METRIC_ON_PARAMETERS, self.log)
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

    def aggregate(self, models):
        self.log.info(f"models to be aggregated count: {len(models)}")

        device = next(models[0].parameters()).device
        for model in models:
            model.to(device)
        avg_model = py_copy.deepcopy(models[0])

        with torch.no_grad():
            for param_name, param in avg_model.named_parameters():
                param.data.zero_()
                for model in models:
                    param.data.add_(model.state_dict()[param_name].data / len(models))

        return avg_model

    def aggregate_clusterwise(self, client_clusters):

        for cluster_idx, cluster in enumerate(client_clusters):
            if len(cluster) == 1:
                continue

            idcs = [client.id for client in cluster]
            self.log.info(f"Aggregating clients: {idcs}")

            cluster_models = [client.model for client in cluster]

            avg_model = self.aggregate(cluster_models)

            if self.config.SAVE_GLOBAL_MODELS:
                _path = f'{GLOBAL_MODELS_SAVING_PATH}/{self.config.MODEL_TYPE}'
                os.makedirs(_path, exist_ok=True)

                global_id = f"cluster{cluster_idx}_aggregated"
                model_path = f'{_path}/global_{global_id}.pth'
                torch.save(avg_model.state_dict(), model_path)
                self.log.info(f"Saved aggregated model for cluster {cluster_idx} to {model_path}")

            for client in cluster:
                client.model.load_state_dict(avg_model.state_dict())
                # client.optimizer = torch.optim.Adam(client.model.parameters(), lr=0.001,  amsgrad=True)
                # client.optimizer = torch.optim.SGD(client.model.parameters(),lr=0.001, momentum=0.9, weight_decay=1e-4)
                # client.optimizer = torch.optim.SGD(client.model.parameters(),lr=0.001, momentum=0.9)

    def _get_vector_a_list(self) -> List[int]:
        return self.rlwe_vector_a_poly.poly_to_list()