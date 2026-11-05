# -*- coding: utf-8 -*-

import numpy as np
import tqdm

from ..state import BranchSupplemental
from .move import Move

__all__ = ["CombineMove"]


class CombineMove(Move):
    """Move that combines specific moves in order.

    Args:
        moves (list): List of moves, similar to how ``moves`` is submitted
            to :class:`eryn.ensemble.EnsembleSampler`. If weights are provided,
            they will be ignored.
        *args (tuple, optional): args to be passed to :class:`Move`.
        share_temperature_control (bool, optional): If ``True``, share the ``temperature_control`` of the 
            combined move with its underlying constituent moves. Default is ``True``. 
        verbose (bool, optional): If ``True``, use ``tqdm`` to show progress throught steps.
            This can be very helpful when debugging.
        **kwargs (dict, optional): kwargs to be passed to :class:`Move`.


    """

    def __init__(self, moves, *args, verbose=False, share_temperature_control=True, **kwargs):
        # store moves
        self.moves = moves
        self.verbose = verbose
        self.share_temperature_control = share_temperature_control
        Move.__init__(self, *args, **kwargs)

    @property
    def accepted(self):
        """Accepted counts for each move."""
        if self._accepted is None:
            raise ValueError(
                "accepted must be inititalized with the init_accepted function if you want to use it."
            )
        # this retrieves the accepted arrays from the individual moves
        accepted_out = [move.accepted for move in self.moves]
        return accepted_out

    @accepted.setter
    def accepted(self, accepted):
        # set the accepted arrays for all moves
        assert isinstance(accepted, np.ndarray)
        for move in self.moves:
            move.accepted = accepted.copy()

    @property
    def acceptance_fraction(self):
        """get acceptance fraction averaged over all moves"""
        acceptance_fraction_out = np.mean(
            [move.acceptance_fraction for move in self.moves], axis=0
        )
        return acceptance_fraction_out

    @property
    def acceptance_fraction_separate(self):
        """get acceptance fraction from each move"""
        acceptance_fraction_out = [move.acceptance_fraction for move in self.moves]
        return acceptance_fraction_out

    @property
    def sub_moves(self):
        """Constituent moves, with ``(move, weight)`` tuples unwrapped."""
        return [move[0] if isinstance(move, tuple) else move for move in self.moves]

    @property
    def temperature_control(self):
        """temperature controller"""
        return self._temperature_control

    @temperature_control.setter
    def temperature_control(self, temperature_control):
        # when setting the temperature control object
        # need to apply it to each move
        for i, move in enumerate(self.moves):
            # if weights were provided with moves, remove move class
            if isinstance(move, tuple):
                move = move[0]
            # set temperature control for each move
            if temperature_control is not None and self.share_temperature_control:
                move.temperature_control = temperature_control

        # main temperature control here for reference
        self._temperature_control = temperature_control

    @property
    def periodic(self):
        """periodic parameter information"""
        return self._periodic

    @periodic.setter
    def periodic(self, periodic):
        # when setting the periodic parameters
        # need to apply it to each move
        for i, move in enumerate(self.moves):
            # if weights were provided with moves, remove move class
            if isinstance(move, tuple):
                move = move[0]
            move.periodic = periodic
        self._periodic = periodic

    def propose(self, model, state):
        """Propose a combined move.

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            tuple: (state, accepted)
                The first return is the state of the sampler after the move.
                The second return value is the accepted count array for each walker
                counting for all proposals.

        """

        # prepare for verbosity if needed
        iterator = enumerate(self.moves)
        if self.verbose:
            iterator = tqdm.tqdm(iterator)

        # we will set this inside the loop during the first iteration
        accepted_out = None
        for i, move in iterator:
            # get move out of tuple
            if isinstance(move, tuple):
                move = move[0]

            # run move
            state, accepted = move.propose(model, state)

            # set (first iteration) or add (after first iteration) to accepted_out
            if accepted_out is None:
                accepted_out = accepted.copy()
            else:
                accepted_out += accepted

        return state, accepted_out
