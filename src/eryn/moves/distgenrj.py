# -*- coding: utf-8 -*-

import numpy as np

from ..priors import ProbDistContainer
from .rj import ReversibleJumpMove

__all__ = ["DistributionGenerateRJ"]


class DistributionGenerateRJ(ReversibleJumpMove):
    """Generate Revesible-Jump proposals from a distribution.

    The prior can be entered as the ``generate_dist`` to generate proposals directly from the prior.

    Args:
        generate_dist (dict): Keys are branch names and values are :class:`ProbDistContainer` objects 
            that have ``logpdf`` and ``rvs`` methods. If you 
        *args (tuple, optional): Additional arguments to pass to parent classes.
        **kwargs (dict, optional): Keyword arguments passed to parent classes.

    """

    def __init__(self, generate_dist, *args, **kwargs):

        # make sure all inputs are distribution Containers
        for key in generate_dist:
            if not isinstance(generate_dist[key], ProbDistContainer):
                raise ValueError(
                    "Distributions need to be eryn.prior.ProbDistContiner object."
                )
        self.generate_dist = generate_dist
        super().__init__(*args, **kwargs)

    def get_model_change_proposal(self, inds, random, nleaves_min, nleaves_max):
        """Helper function for changing the model count by 1.
        
        This helper function works with nested models where you want to add or remove
        one leaf at a time. 

        Args:
            inds (np.ndarray): ``inds`` values for this specific branch with shape 
                ``(ntemps, nwalkers, nleaves_max)``.
            random (object): Current random state of the sampler.
            nleaves_min (int): Minimum allowable leaf count for this branch.
            nleaves_max (int): Maximum allowable leaf count for this branch.

        Returns:
            dict: Keys are ``"+1"`` and ``"-1"``. Values are indexing information.
                    ``"+1"`` and ``"-1"`` indicate if a source is being added or removed, respectively.
                    The indexing information is a 2D array with shape ``(number changing, 3)``.
                    The length 3 is the index into each of the ``(ntemps, nwalkers, nleaves_max)``.
        
        """

        ntemps, nwalkers, _ = inds.shape

        nleaves = inds.sum(axis=-1)

        # choose whether to add or remove
        if self.fix_change is None:
            change = random.choice([-1, +1], size=nleaves.shape)
        else:
            change = np.full(nleaves.shape, self.fix_change)

        # fix edge cases
        change = (
            change * ((nleaves != nleaves_min) & (nleaves != nleaves_max))
            + (+1) * (nleaves == nleaves_min)
            + (-1) * (nleaves == nleaves_max)
        )

        # setup storage for this information
        inds_for_change = {}
        num_increases = np.sum(change == +1)
        inds_for_change["+1"] = np.zeros((num_increases, 3), dtype=int)
        num_decreases = np.sum(change == -1)
        inds_for_change["-1"] = np.zeros((num_decreases, 3), dtype=int)

        # TODO: not loop ? Is it necessary?
        # TODO: might be able to subtract new inds from old inds type of thing
        # fill the inds_for_change
        increase_i = 0
        decrease_i = 0
        for t in range(ntemps):
            for w in range(nwalkers):
                # check if add or remove
                change_tw = change[t][w]
                # inds array from specific walker
                inds_tw = inds[t][w]

                # adding
                if change_tw == +1:
                    # find where leaves are not currently used
                    inds_false = np.where(inds_tw == False)[0]
                    # decide which spot to add
                    ind_change = random.choice(inds_false)
                    # put in the indexes into inds arrays
                    inds_for_change["+1"][increase_i] = np.array(
                        [t, w, ind_change], dtype=int
                    )
                    # count increases
                    increase_i += 1

                # removing
                elif change_tw == -1:
                    # change_tw == -1
                    # find which leavs are used
                    inds_true = np.where(inds_tw == True)[0]
                    # choose which to remove
                    ind_change = random.choice(inds_true)
                    # add indexes into inds
                    if inds_for_change["-1"].shape[0] > 0:
                        inds_for_change["-1"][decrease_i] = np.array(
                            [t, w, ind_change], dtype=int
                        )
                        decrease_i += 1
                    # do not care currently about what we do with discarded coords, they just sit in the state
                # model component number not changing
                else:
                    pass
        return inds_for_change

    def get_proposal(
        self, all_coords, all_inds, nleaves_min_all, nleaves_max_all, random, **kwargs
    ):
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            nleaves_min_all (dict): Minimum values of leaf ount for each model. Must have same order as ``all_cords``. 
            nleaves_max_all (dict): Maximum values of leaf ount for each model. Must have same order as ``all_cords``. 
            random (object): Current random state of the sampler.
            **kwargs (ignored): For modularity. 

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """
        # prepare the output dictionaries
        q = {}
        new_inds = {}
        all_inds_for_change = {}

        # loop over the models included here
        assert len(nleaves_min_all)
        assert len(all_coords.keys()) == len(nleaves_max_all.keys())
        for (name, inds) in all_inds.items():
            nleaves_max = nleaves_max_all[name]
            nleaves_min = nleaves_min_all[name]
            if nleaves_min == nleaves_max:
                continue
            elif nleaves_min > nleaves_max:
                raise ValueError("nleaves_min is greater than nleaves_max. Not allowed.")

            # get the inds adjustment information
            all_inds_for_change[name] = self.get_model_change_proposal(
                inds, random, nleaves_min, nleaves_max
            )

        # loop through branches and propose new points from the prio
        for i, (name, coords, inds) in enumerate(
            zip(all_coords.keys(), all_coords.values(), all_inds.values(),)
        ):
            # put in base information
            ntemps, nwalkers, nleaves_max, ndim = coords.shape
            new_inds[name] = inds.copy()
            q[name] = coords.copy()

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # if not included
            if name not in all_inds_for_change:
                continue

            # inds changing for this branch
            inds_for_change = all_inds_for_change[name]

            # adjust inds

            # adjust deaths from True -> False
            inds_here = tuple(inds_for_change["-1"].T)
            new_inds[name][inds_here] = False

            # factor is +log q()
            current_generate_dist = self.generate_dist[name]
            factors[inds_here[:2]] += +1 * current_generate_dist.logpdf(
                q[name][inds_here]
            )

            # adjust births from False -> True
            inds_here = tuple(inds_for_change["+1"].T)
            new_inds[name][inds_here] = True

            # add coordinates for new leaves
            current_generate_dist = self.generate_dist[name]
            inds_here = tuple(inds_for_change["+1"].T)
            num_inds_change = len(inds_here[0])

            q[name][inds_here] = current_generate_dist.rvs(size=num_inds_change)

            # factor is -log q()
            factors[inds_here[:2]] += -1 * current_generate_dist.logpdf(
                q[name][inds_here]
            )

        return q, new_inds, factors
