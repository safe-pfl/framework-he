# PATHS
LOG_PATH: str = "./logs"
MODELS_SAVING_PATH: str = "./models"
GLOBAL_MODELS_SAVING_PATH: str = f"{MODELS_SAVING_PATH}/after_aggregation"
PLOT_PATH: str = './plots'

# RESOURCES
TOPOLOGY_MANAGER_CPU_RESOURCES: float | int = 0.5
SAFETY_EPSILON: float= 0.01

# Algorithm
SERVER_ID: str = "server"

# Communication
MODEL_UPDATE: str = "model_update"

MESSAGE_BODY_STATES = "state"

# Encryption Methods
ENCRYPTION_HOMOMORPHIC_XMKCKKS = "he_xmkckks"