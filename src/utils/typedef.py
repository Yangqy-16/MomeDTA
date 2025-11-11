from typing import TypeAlias
from torch import Tensor

SeqType: TypeAlias = str | Tensor
ValidType: TypeAlias = dict[str, Tensor] | Tensor