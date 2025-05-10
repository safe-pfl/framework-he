import numpy as np
import torch
from core.client import Client
from core.server import Server
from data.data_driven_clustering import compute_data_driven_clustering
from data.load_and_prepare_data import load_and_prepare_data
from nets.network_factory import network_factory
from utils.client_ids_list import client_ids_list_generator
from utils.display_stats import display_stats, ExperimentLogger
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

    config_dict = config_dict | {'desired_distribution': None} # TODO: update

    config = ConfigValidator(**config_dict)

    log = Log(log_path(
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
    ), config.MODEL_TYPE, config.DISTANCE_METRIC)

    # table_data = [[key, value] for key, value in config.items()] # TODO: fix items function in config validator
    # log.info(tabulate(table_data, headers=["Config Key", "Value"], tablefmt="grid"))


    log.info("----------    framework   setup   --------------------------------------------------")
    FrameworkSetup.path_setup(config)

    log.info("----------    data    distribution   --------------------------------------------------")
    train_loaders, test_loaders = load_and_prepare_data(config, log)

    if config.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING:
        log.info("clients train loader label distribution")
        config = config | {"DATA_DRIVEN_CLUSTERING": compute_data_driven_clustering(train_loaders, config, log)}


    log.info("----------    runtime configurations  --------------------------------------------------")
    clients_id_list = client_ids_list_generator(config.NUMBER_OF_CLIENTS, log=log)

    config.RUNTIME_COMFIG = RuntimeConfig(
        clients_id_list=clients_id_list,
        # train_loaders=train_loaders,
        # test_loaders=test_loaders,
        log=log
    )

    log.info("----------    model initialization --------------------------------------------------")
    initial_model = network_factory(model_type=config.MODEL_TYPE, number_of_classes=config.NUMBER_OF_CLASSES, pretrained=True)

    log.info("----------    client initialization --------------------------------------------------")

    client_list = [i for i in range(config.NUMBER_OF_CLIENTS)]
    assert len(client_list) == config.NUMBER_OF_CLIENTS

    clients = [
        Client(
            initial_model,
            lambda x: torch.optim.SGD(
                x, lr=0.001, momentum=0.9, weight_decay=1e-4
            ),
            i,
            train_loaders[i],
            test_loaders[i],
            config,
            log
        )
        for i in client_list
    ]


    log.info("----------    server initialization --------------------------------------------------")
    server = Server(initial_model, config, log)


    log.info("----------    Federated Learning initialization --------------------------------------------------")
    cfl_stats = ExperimentLogger()
    cluster_indices = [np.arange(len(clients)).astype("int")]
    global_clients_clustered = []
    CLUSTERING_LABELS = None
    STOP_CLUSTERING: bool = False

    for c_round in range(1, config.FEDERATED_LEARNING_ROUNDS + 1):
        if c_round == 1:
            for client in clients:
                client.synchronize_with_server(server)

        """
            Checking clustering conditions
        """
        TRIGGER_CLUSTERING = (
                not SAFE_PFL_CONFIG["FED_AVG"]
                and not STOP_CLUSTERING
                and not SAFE_PFL_CONFIG["PRE_COMPUTED_OPTIMAL_CLUSTERING"]
                and c_round % SAFE_PFL_CONFIG["CLUSTERING_PERIOD"] == 0
                or SAFE_PFL_CONFIG["CLUSTER_AT_FIRST"]
        )
        SAFE_PFL_CONFIG["CLUSTER_AT_FIRST"] = False
        """
            Participating clients training loop
        """
        for index, client in enumerate(clients):
            client.compute_weight_update(
                be_ready_for_clustering=TRIGGER_CLUSTERING,
                epochs=SAFE_PFL_CONFIG["ROUND_EPOCHS"],
            )

        """
            Calculating the optimal sensitivity value (P)
        """
        if (
                c_round == 1
                and SAFE_PFL_CONFIG["DISTANCE_METRIC"]
                == distances_constants.DISTANCE_COORDINATE
                and SAFE_PFL_CONFIG["DYNAMIC_SENSITIVITY_PERCENTAGE"]
        ):
            SAFE_PFL_CONFIG.update(
                {
                    "SENSITIVITY_PERCENTAGE": calculate_optimal_sensitivity_percentage(
                        clients[0].model
                    )
                }
            )
            log.info(
                f'done calculating optimal sensitivity percentage with value of {config.SENSITIVITY_PERCENTAGE}'
            )

        if TRIGGER_CLUSTERING:
            full_similarities = server.compute_pairwise_similarities(clients=clients)
            log.warn(f"Global clustering triggered {c_round}")

            clustering = server.cluster_clients(full_similarities)

            # cleaning the memory up
            del full_similarities
            for client in clients:
                client.gradients = {}

            cluster_indices = []
            CLUSTERING_LABELS = clustering.labels_
            for label in np.unique(clustering.labels_):
                cluster_indices.append(np.where(clustering.labels_ == label)[0].tolist())

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

            if config.SAVE_BEFORE_AGGREGATION_MODELS:
                for client in clients:
                    torch.save(
                        client.model.state_dict(),
                        MODEL_SAVING_PATH + f"client_{client.id}_model.pt",
                    )

        client_clusters = []
        for cluster in cluster_indices:
            new_orientation = []
            for index in cluster:
                new_orientation.append(clients[index])
            client_clusters.append(new_orientation)
        global_clients_clustered = client_clusters


        server.aggregate_clusterwise(global_clients_clustered)

        acc_clients = [client.evaluate() for client in clients]

        if not STOP_CLUSTERING:
            acc_mean = np.mean(acc_clients)
            log.info(
                f'checking whether to stop clustering or not with STOP_AVG_ACCURACY value of {config.STOP_AVG_ACCURACY} and averaged accuracy of {acc_mean}'
            )
            if acc_mean >= config.STOP_AVG_ACCURACY and (
                    np.array_equal(CLUSTERING_LABELS, OPTIMAL_TRAIN_CLUSTERING)
                    or np.array_equal(CLUSTERING_LABELS, OPTIMAL_TEST_CLUSTERING)
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

