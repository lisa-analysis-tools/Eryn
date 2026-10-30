# -*- coding: utf-8 -*-


from .combine import CombineMove
from .de import DEMove, DESnookerMove
from .distgen import DistributionGenerate
from .distgenrj import DistributionGenerateRJ
from .eigenaxis import EigenAxisMove
from .flow import ConditionalFlowMove
from .flowgaussian import FlowGaussianMove
from .flownuts import FlowNUTSMove
from .gaussian import GaussianMove
from .group import GroupMove
from .groupde import GroupDEMove, GroupDESnookerMove
from .groupstretch import GroupStretchMove
from .independentmh import IndependentProposalMove
from .infomatrix import InfoMatrixMove
from .mh import MHMove
from .move import Move
# from .basicmodelswaprj import BasicSymmetricModelSwapRJMove
from .mtdistgen import MTDistGenMove
from .mtdistgenrj import MTDistGenMoveRJ
from .multipletry import MultipleTryMove
# from .walk import WalkMove
from .nuts import NUTSMove, NUTSSampler
from .red_blue import RedBlueMove
from .ridgegibbs import RidgeGibbsMove
from .rj import ReversibleJumpMove
from .stretch import StretchMove
# from .kde import KDEMove
# from .walk import WalkMove
from .tempering import TemperatureControl

__all__ = [
    "Move",
    "MHMove",
    "EigenAxisMove",
    "GaussianMove",
    "RedBlueMove",
    "StretchMove",
    "DEMove",
    "DESnookerMove",
    "InfoMatrixMove",
    "DistributionGenerateRJ",
    "DistributionGenerate",
    "MTDistGenMove",
    "MTDistGenMoveRJ",
    "TemperatureControl",
    "ReversibleJumpMove",
    "MultipleTryMove",
    "GroupMove",
    "GroupStretchMove",
    "GroupDEMove",
    "GroupDESnookerMove",
    "RidgeGibbsMove",
    "CombineMove",
    "NUTSMove",
    "NUTSSampler",
    "IndependentProposalMove",
    "ConditionalFlowMove",
    "FlowGaussianMove",
    "FlowNUTSMove",
]
