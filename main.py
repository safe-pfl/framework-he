import numpy as np
import torch
import math
from xmkckks import RLWE
from xmkckks.rlwe import discrete_uniform

from constants import distances_constants
from constants.framework import ENCRYPTION_HOMOMORPHIC_XMKCKKS
from core.client import Client
from core.clustering.data import (
    cluster_clients,
    compute_similarity_matrix,
    calculate_label_distribution,
)
from core.encryption.homomorphic import rlwe_generator
from core.sensitivity_percentage.find_optimal_pruning_rate import (
    calculate_optimal_sensitivity_percentage,
)
from core.server import Server
from data.data_driven_clustering import compute_data_driven_clustering
from data.load_and_prepare_data import load_and_prepare_data
from nets.network_factory import network_factory
from utils.check_train_test_class_mismatch import check_train_test_class_mismatch
from utils.client_ids_list import client_ids_list_generator
from utils.clustering.update_cluster import update_clusters_based_on_indexes
from utils.count_parameters import count_parameters
from utils.display_stats import display_stats, ExperimentLogger
from utils.encryption.next_prime import next_prime
from utils.framework_setup import FrameworkSetup
from utils.log import Log
from utils.log_path import log_path
from utils.variable_name import var_name
from utils.yaml_loader import load_objectified_yaml
from utils.checker import none_checker
import typer
from validators.config_validator import ConfigValidator
from validators.runtime_config import RuntimeConfig


def main(config_yaml_path: str = "./config.yaml"):
    config_yaml_path = none_checker(config_yaml_path, var_name(config_yaml_path))

    config_dict = load_objectified_yaml(config_yaml_path)

    config_dict = config_dict | {"desired_distribution": None}  # TODO: update

    config = ConfigValidator(**config_dict)

    log = Log(
        log_path(
            model_type=config.MODEL_TYPE,
            dataset_type=config.DATASET_TYPE,
            data_distribution=config.DATA_DISTRIBUTION,
            distance_metric=config.DISTANCE_METRIC,
            sensitivity_percentage=config.SENSITIVITY_PERCENTAGE,
            fed_avg=config.FED_AVG,
            dynamic_sensitivity_percentage=config.DYNAMIC_SENSITIVITY_PERCENTAGE,
            distance_metric_on_parameters=config.DISTANCE_METRIC_ON_PARAMETERS,
            pre_computed_data_driven_clustering=config.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING,
            remove_common_ids=config.REMOVE_COMMON_IDS,
        ),
        config.MODEL_TYPE,
        config.DISTANCE_METRIC,
    )

    MODEL_SAVING_PATH = f"models/{config.MODEL_TYPE}/{config.DISTANCE_METRIC}/"

    # table_data = [[key, value] for key, value in config.items()] # TODO: fix items function in config validator
    # log.info(tabulate(table_data, headers=["Config Key", "Value"], tablefmt="grid"))

    log.info(
        "----------    framework setup   --------------------------------------------------"
    )
    FrameworkSetup.path_setup(config)

    log.info(
        "----------    data distribution   --------------------------------------------------"
    )
    train_loaders, test_loaders = load_and_prepare_data(config, log)

    if config.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING:
        log.info("clients train loader label distribution")
        config = config | {
            "DATA_DRIVEN_CLUSTERING": compute_data_driven_clustering(
                train_loaders, config, log
            )
        }

    log.info(
        "----------    data driven clustering   --------------------------------------------------"
    )

    if config.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING:
        train_label_distributions = [
            calculate_label_distribution(loader, "train", config, log)
            for loader in train_loaders
        ]
        train_similarity_matrix = compute_similarity_matrix(train_label_distributions)
        OPTIMAL_TRAIN_CLUSTERING = cluster_clients(train_similarity_matrix)

    else:
        OPTIMAL_TRAIN_CLUSTERING = []

    log.info(
        "----------    runtime configurations  --------------------------------------------------"
    )
    clients_id_list = client_ids_list_generator(config.NUMBER_OF_CLIENTS, log=log)

    config.RUNTIME_CONFIG = RuntimeConfig(
        clients_id_list=clients_id_list, rlwe=None, q=None, log=log
    )

    log.info(
        "----------    model initialization --------------------------------------------------"
    )
    initial_model = network_factory(
        model_type=config.MODEL_TYPE,
        number_of_classes=config.NUMBER_OF_CLASSES,
        pretrained=config.PRETRAINED_MODELS,
    )

    if config.ENCRYPTION_METHOD is not None:
        log.info(
            "----------    encryption initialization --------------------------------------------------"
        )
        if config.ENCRYPTION_METHOD == ENCRYPTION_HOMOMORPHIC_XMKCKKS:
            config.RUNTIME_CONFIG.rlwe = rlwe_generator(
                model=initial_model, config=config, log=log
            )

    log.info(
        "----------    client initialization --------------------------------------------------"
    )

    client_list = [i for i in range(config.NUMBER_OF_CLIENTS)]
    assert len(client_list) == config.NUMBER_OF_CLIENTS

    clients = [
        Client(
            initial_model,
            lambda x: torch.optim.SGD(
                x, lr=config.LEARNING_RATE, momentum=0.9, weight_decay=1e-4
            ),
            i,
            train_loaders[i],
            test_loaders[i],
            config,
            log,
        )
        for i in client_list
    ]

    log.info(
        "----------    server initialization --------------------------------------------------"
    )
    server = Server(initial_model, config, log)

    if (
        config.ENCRYPTION_METHOD is not None
        and config.ENCRYPTION_METHOD == ENCRYPTION_HOMOMORPHIC_XMKCKKS
    ):
        log.info(
            "----------    generating sever vector a   --------------------------------------------------"
        )
        if server.rlwe.a is None:
            log.info(
                "the sever rlwe vector a must be already generated but here is None!"
            )

    log.info(
        "----------    Federated Learning initialization --------------------------------------------------"
    )
    cfl_stats = ExperimentLogger()
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = []
    CLUSTERING_LABELS = None
    STOP_CLUSTERING: bool = False

    # Track previous cluster composition to detect changes
    previous_cluster_composition = {}
    # Track which clients already have keys for which clusters
    client_cluster_keys = {client.id: set() for client in clients}

    for c_round in range(1, config.FEDERATED_LEARNING_ROUNDS + 1):
        if c_round == 1:
            for client in clients:
                client.synchronize_with_server(server)

        """
            Checking clustering conditions
        """
        TRIGGER_CLUSTERING = (
            not config.FED_AVG
            and not STOP_CLUSTERING
            and not config.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING
            and c_round % config.CLUSTERING_PERIOD == 0
        )
        """
            Participating clients training loop
        """
        for index, client in enumerate(clients):
            client.compute_weight_update(
                be_ready_for_clustering=TRIGGER_CLUSTERING,
                epochs=config.NUMBER_OF_EPOCHS,
            )
            # If we're using encryption and this is a clustering round, ensure importance masks are created
            if (
                config.ENCRYPTION_METHOD is not None
                and config.ENCRYPTION_METHOD == ENCRYPTION_HOMOMORPHIC_XMKCKKS
                and (TRIGGER_CLUSTERING or c_round == 1)
            ):
                # Create importance mask based on gradients if it doesn't exist yet
                if client.importance_mask is None and client.gradients:
                    client.importance_mask = client.create_importance_mask()
                    log.info(
                        f"Created importance mask for client {client.id} based on gradients"
                    )

        """
            Calculating the optimal sensitivity value (P)
        """
        if (
            c_round == 1
            and config.DISTANCE_METRIC == distances_constants.DISTANCE_COORDINATE
            and config.DYNAMIC_SENSITIVITY_PERCENTAGE
        ):
            config.SENSITIVITY_PERCENTAGE = calculate_optimal_sensitivity_percentage(
                clients[0].model, config, log
            )

            log.info(
                f"done calculating optimal sensitivity percentage with value of {config.SENSITIVITY_PERCENTAGE}"
            )

        # Flag to track if clustering has changed
        clustering_changed = False

        # Track current round in server for visualization purposes
        server.current_round = c_round

        if TRIGGER_CLUSTERING:
            full_similarities = server.compute_pairwise_similarities(clients=clients)
            log.warn(f"Global clustering triggered {c_round}")

            clustering = server.cluster_clients(full_similarities)

            # cleaning the memory up
            del full_similarities

            # Make sure importance masks are created based on the gradients before clearing them
            if (
                config.ENCRYPTION_METHOD is not None
                and config.ENCRYPTION_METHOD == ENCRYPTION_HOMOMORPHIC_XMKCKKS
            ):
                for client in clients:
                    if client.gradients and (
                        client.importance_mask is None or c_round > 1
                    ):
                        client.importance_mask = client.create_importance_mask()
                        log.info(
                            f"Created/updated importance mask for client {client.id} during clustering"
                        )

            # Now clear the gradients to save memory
            for client in clients:
                client.gradients = {}

            cluster_indices = []
            CLUSTERING_LABELS = clustering.labels_
            for label in np.unique(clustering.labels_):
                cluster_indices.append(
                    np.where(clustering.labels_ == label)[0].tolist()
                )

            client_clusters = update_clusters_based_on_indexes(
                clients=clients, cluster_indices=cluster_indices
            )

            # Check if clustering has changed
            current_cluster_composition = {
                i: sorted([client.id for client in cluster])
                for i, cluster in enumerate(client_clusters)
            }

            if previous_cluster_composition != current_cluster_composition:
                clustering_changed = True
                log.info(f"Clustering has changed in round {c_round}")
                previous_cluster_composition = current_cluster_composition
                # Reset client key tracking when clusters change
                client_cluster_keys = {client.id: set() for client in clients}

        elif (
            c_round % config.CLUSTERING_PERIOD == 0
            and config.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING
        ):
            cluster_indices = []
            for label in np.unique(OPTIMAL_TRAIN_CLUSTERING):
                cluster_indices.append(
                    np.where(OPTIMAL_TRAIN_CLUSTERING == label)[0].tolist()
                )

            log.info(
                f"clustering based on optimal clustering {cluster_indices} @ round number {c_round}"
            )

            client_clusters = update_clusters_based_on_indexes(
                clients=clients, cluster_indices=cluster_indices
            )

            # Check if clustering has changed
            current_cluster_composition = {
                i: sorted([client.id for client in cluster])
                for i, cluster in enumerate(client_clusters)
            }

            if previous_cluster_composition != current_cluster_composition:
                clustering_changed = True
                log.info(f"Clustering has changed in round {c_round}")
                previous_cluster_composition = current_cluster_composition
                # Reset client key tracking when clusters change
                client_cluster_keys = {client.id: set() for client in clients}

            if config.SAVE_BEFORE_AGGREGATION_MODELS:
                for client in clients:
                    torch.save(
                        client.model.state_dict(),
                        MODEL_SAVING_PATH + f"client_{client.id}_model.pt",
                    )
        if len(client_clusters) == 0:
            log.warn(
                f"all clients are going to aggregate to each others in FL round: {c_round}"
            )
            client_clusters = [clients]

            # First round is always considered a change in clustering
            if c_round == 1:
                clustering_changed = True
                previous_cluster_composition = {
                    0: sorted([client.id for client in clients])
                }

        if (
            config.ENCRYPTION_METHOD is not None
            and config.ENCRYPTION_METHOD == ENCRYPTION_HOMOMORPHIC_XMKCKKS
        ):
            # Only regenerate keys if clustering has changed or it's the first round
            if clustering_changed or c_round == 1:
                log.info(
                    "----------    aggregating public keys (clustering changed)    --------------------------------------------------"
                )
                for cluster_idx, cluster in enumerate(client_clusters):
                    # Generate cluster-specific RLWE vector
                    cluster_vector_a = server.get_cluster_rlwe_vector(cluster_idx)

                    # Generate and aggregate cluster-specific public keys
                    cluster_aggregated_public_key = None
                    for client in cluster:
                        # Update tracking of which client has keys for which cluster
                        client_cluster_keys[client.id].add(cluster_idx)

                        if cluster_aggregated_public_key is None:
                            cluster_aggregated_public_key = client.generate_pubkey(
                                vector_a=cluster_vector_a.poly_to_list(),
                                cluster_idx=cluster_idx,
                            )
                        else:
                            cluster_aggregated_public_key += client.generate_pubkey(
                                vector_a=cluster_vector_a.poly_to_list(),
                                cluster_idx=cluster_idx,
                            )

                    # Store and distribute cluster's aggregated key
                    server.store_cluster_aggregated_pubkey(
                        cluster_idx, cluster_aggregated_public_key
                    )
                    for client in cluster:
                        client.store_aggregated_pubkey(cluster_aggregated_public_key)
            else:
                log.info(
                    "----------    using existing public keys (clustering unchanged)    --------------------------------------------------"
                )

        log.info(
            f"triggering aggregation based on these indexes {cluster_indices} in FL round: {c_round}"
        )

        server.aggregate_clusterwise(
            client_clusters=client_clusters,
            use_encryption=config.ENCRYPTION_METHOD is not None
            and config.ENCRYPTION_METHOD == ENCRYPTION_HOMOMORPHIC_XMKCKKS,
        )

        acc_clients = [client.evaluate() for client in clients]

        if not STOP_CLUSTERING:
            acc_mean = np.mean(acc_clients)
            log.info(
                f"checking whether to stop clustering or not with STOP_AVG_ACCURACY value of {config.STOP_AVG_ACCURACY} and averaged accuracy of {acc_mean}"
            )
            if acc_mean >= config.STOP_AVG_ACCURACY and np.array_equal(
                CLUSTERING_LABELS, OPTIMAL_TRAIN_CLUSTERING
            ):
                log.info(f"clustering stop triggered at round {c_round}")
                STOP_CLUSTERING = True

        cfl_stats.log(
            {
                "acc_clients": acc_clients,
                "rounds": c_round,
                "clusters": cluster_indices,
            }
        )

        display_stats(
            cfl_stats,
            config.FEDERATED_LEARNING_ROUNDS,
            log,
        )


if __name__ == "__main__":
    typer.run(main)
