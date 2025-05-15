from dataclasses import dataclass
from typing import List

from numpy import number
from xmkckks import RLWE
from utils.log import Log


@dataclass
class RuntimeConfig:

    clients_id_list: List[str] | List[int]

    # TODO: HE related runtime configurations
    rlwe: RLWE
    q: int

    # application Log instance
    log: Log

