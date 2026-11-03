import numpy as np

from .mh import MHMove
from eryn.moves.multipletry import MultipleTryMove


class MTDistGenMove(MultipleTryMove, MHMove):
    """Perform a multiple try MH move based on a distribution.
    
    Distribution must be independent of the current point.
    
    This is effectively an example of the mutliple try class inheritance structure.

    Args:
        generate_dist (dict): Keys are branch names and values are :class:`ProbDistContainer` objects 
            that have ``logpdf`` and ``rvs`` methods. If you 
        *args (tuple, optional): Additional arguments to pass to parent classes.
        **kwargs (dict, optional): Keyword arguments passed to parent classes.
    
    """

    def __init__(self, generate_dist, **kwargs):

        MultipleTryMove.__init__(self, **kwargs)
        MHMove.__init__(self, **kwargs)

        self.generate_dist = generate_dist

    def special_generate_logpdf(self, generated_coords):
        """Get logpdf of generated coordinates.
        
        Args:
            generated_coords (np.ndarray): Current coordinates of walkers. 
            
        Returns:
            np.ndarray: logpdf of generated points.
            
        """
        return self.generate_dist.logpdf(generated_coords)

    def special_generate_func(
        self, coords, random, size=1, fill_tuple=None, fill_values=None, **kwargs
    ):
        """Generate samples and calculate the logpdf of their proposal function.
        
        Args:
            coords (np.ndarray): Current coordinates of walkers. 
            random (obj): Random generator.
            *args (tuple, optional): additional arguments passed by overwriting the 
                ``get_proposal`` function and passing ``args_generate`` keyword argument.
            size (int, optional): Number of tries to generate. 
            fill_tuple (tuple, optional): Length 2 tuple with the indexing of which values to fill
                when generating. Can be used for auxillary proposals or reverse RJ proposals. First index is the index into walkers and the second index is 
                the index into the number of tries. (default: ``None``)
            fill_values (np.ndarray): values to fill associated with ``fill_tuple``. Should 
                have size ``(len(fill_tuple[0]), ndim)``. (default: ``None``).
            **kwargs (tuple, optional): additional keyword arguments passed by overwriting the 
                ``get_proposal`` function and passing ``kwargs_generate`` keyword argument.

        Returns:
            tuple: (generated points, logpdf of generated points).
        
        """
        nwalkers = coords.shape[0]

        if not isinstance(size, int):
            raise ValueError("size must be an int.")

        # generate coordinates
        generated_coords = self.generate_dist.rvs(size=(nwalkers, size))

        # fill special coordinates
        if fill_values is not None:
            generated_coords[fill_tuple] = fill_values

        # get logpdf
        generated_logpdf = self.special_generate_logpdf(
            generated_coords.reshape(nwalkers * size, -1)
        ).reshape(nwalkers, size)

        return generated_coords, generated_logpdf

    def set_coords_and_inds(self, generated_coords):
        """Setup coordinates for prior and Likelihood
        
        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``. 

        Returns:
            dict: Coordinates for Likelihood and Prior.
        
        """
        ndim = self.current_state.branches[self.key_in].shape[-1]

        coords_in_dict = {}
        for key in self.current_state.branches.keys():
            if key == self.key_in:
                coords_in_dict[key] = generated_coords.reshape(-1, 1, ndim)[None, :]

            else:
                coords_in_dict[key] = self.current_state.branches[key].coords.reshape(
                    (1, -1) + self.current_state.branches[key].shape[-2:]
                )

        return coords_in_dict

    def special_like_func(self, generated_coords, **kwargs):
        """Calculate the Likelihood for sampled points.
        
        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``. 
            **kwargs (dict, optional): For compatibility. 

        Returns:
            np.ndarray: Likelihood values with shape ``(generated_coords.shape[0], num_try).``
        
        """
        coords_in = self.set_coords_and_inds(generated_coords)
        ll = self.current_model.compute_log_like_fn(coords_in)[0]
        ll = ll[0].reshape(-1, self.num_try)
        return ll

    def special_prior_func(self, generated_coords, **kwargs):
        """Calculate the Prior for sampled points.
        
        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``. 
            **kwargs (dict, optional): For compatibility. 
            
        Returns:
            np.ndarray: Prior values with shape ``(generated_coords.shape[0], num_try).``
        
        """
        coords_in = self.set_coords_and_inds(generated_coords)
        lp = self.current_model.compute_log_prior_fn(coords_in)
        return lp.reshape(-1, self.num_try)

