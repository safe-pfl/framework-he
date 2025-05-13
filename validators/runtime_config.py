from dataclasses import dataclass
from typing import List

from xmkckks import RLWE

from utils.log import Log
from torch.utils.data import DataLoader


@dataclass
class RuntimeConfig:

    clients_id_list: List[str] | List[int]

    # TODO: HE related runtime configurations
    rlwe: RLWE

    # application Log instance
    log: Log

