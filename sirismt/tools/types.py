import sys
sys.path.append('..')

from typing import *
from sirismt.tools import objects

__all__ = [
    'TacSeqData',
    'ProbeDict',
    'Strategy',
    'TacticSeq',
    'CandidateSeq',
    'TacticSeqGroup',
    'Optional'
]

TacSeqData: TypeAlias = tuple[str, list[str | objects.Tactic]] # str(tac_seq), tac_seq
ProbeDict: TypeAlias = dict[str, float]
Strategy: TypeAlias = Optional[objects.Tactic]

TacticSeq: TypeAlias = list[str]
CandidateSeq: TypeAlias = tuple[str, TacticSeq]

TacticSeqGroup: TypeAlias = dict[str, list[str]]