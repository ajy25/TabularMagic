import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from .base_model import BaseGenerativeModel


class GaussianBernoulliRBM_TorchModule(torch.nn.Module):
    """Computational support for GaussianBernoulliRBM class.
    """

    def __init__(self, n_vis: int, n_hid: int, var: float = None):
        """
        Constructs a Gaussian-Bernoulli Restricted Boltzmann Machine with 
        adversarial training on hidden units.

        Parameters
        ----------
        - n_vis : int. Number of visible nodes
        - n_hid : int. Number of hidden nodes
        - var : float | None. Set variance for each visible node;
            if None, we learn the variance on each visible node
        """
        super().__init__()
        self.rng = torch.Generator()
        self.reset_seed(42)
        self.adversary_memory = None
        self.reset_hyperparameters(n_vis=n_vis, n_hid=n_hid, var=var)


    def metadata(self):
        """
        Returns the metadata of the RBM object.

        Returns
        -------
        - dict
        """
        metadata = {
            "n_vis": self.n_vis,
            "n_hid": self.n_hid,
            "var": self.var
        }
        return metadata
    

    def reset_hyperparameters(self, n_vis: int = None, n_hid: int = None, 
                              var: float = None):
        if n_vis is not None:
            self.n_vis = n_vis
        if n_hid is not None:
            self.n_hid = n_hid
        self.var = var
        self.W = nn.Parameter(torch.Tensor(self.n_vis, self.n_hid))
        self.mu = nn.Parameter(torch.Tensor(self.n_vis))
        self.b = nn.Parameter(torch.Tensor(self.n_hid))
        if self.var is None:
            self.log_var = nn.Parameter(torch.Tensor(self.n_vis))
        else:
            self.log_var = torch.ones((self.n_vis)) * np.log(var)
        self.reset_parameters()


    def reset_parameters(self, seed: int = 42):
        """
        Resets trainable parameters of the Gaussian-Bernoulli RBM.
        """
        torch.manual_seed(seed)
        nn.init.xavier_normal_(self.W)
        nn.init.constant_(self.mu, 0)
        nn.init.constant_(self.b, 0)
        if self.var is None:
            nn.init.constant_(self.log_var, 0)


    def _energy(self, v: torch.Tensor, 
                h: torch.Tensor):
        """
        Equation (1, 2) in https://arxiv.org/pdf/2210.10318.

        Parameters
        ----------
        - v: torch.Tensor ~ (batch_size, n_vis)
        - h : torch.Tensor ~ (batch_size, n_hid)

        Returns
        -------
        - energy : float
        """
        var = self._variance()
        pos = torch.sum(0.5 * (v - self.mu) ** 2 / var, dim=1)
        neg = torch.sum(((v / var) @ (self.W)) * h, dim=1) + \
            torch.sum(h * self.b, dim=1)
        return (pos - neg) / v.shape[0]
    

    def _marginal_energy(self, v: torch.Tensor):
        """
        Equation (5) in https://arxiv.org/pdf/2210.10318.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)

        @returns
        - torch.Tensor
        """
        var = self._variance()
        pos = torch.sum(torch.square(0.5 * (v - self.mu) / var.sqrt()), dim=1)
        softmax_logit = ((v / var) @ self.W + self.b).clip(max=80)
        neg = torch.sum(torch.log(1 + torch.exp(softmax_logit)), dim=1)
        return pos - neg

    
    def reset_seed(self, seed: int):
        """
        Resets the rng seed to enforce reproducibility after training.

        @args
        - seed: int
        """
        self.rng.manual_seed(seed)

    @torch.no_grad()
    def _prob_h_given_v(self, v: torch.Tensor):
        """
        Computes sigmoid activation for p(h=1|v) according to equation (3) in
        https://arxiv.org/pdf/2210.10318; in other words, computes the
        parameters for hidden Bernoulli random variables given visible units.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)

        @returns
        - torch.Tensor ~ (batch_size, n_hid)
        """
        return torch.sigmoid((v / self._variance()) @ self.W + self.b)

    @torch.no_grad()
    def _prob_v_given_h(self, h: torch.Tensor):
        """
        Computes mu for p(v|h) according to equation (4) in
        https://arxiv.org/pdf/2210.10318.

        Parameters
        ----------
        - h : torch.Tensor ~ (batch_size, n_hid)

        Returns
        -------
        - torch.Tensor ~ (batch_size, n_vis)
        """
        return h @ self.W.t() + self.mu


    @torch.no_grad()
    def _block_gibbs_sample(self, v: torch.Tensor = None,
                            h: torch.Tensor = None, clamp: torch.Tensor = None,
                            n_gibbs = 1, add_noise = True, 
                            begin_from_randn_init = True):
        """
        Familiar block Gibbs sampling method of visible and hidden units.

        Parameters
        ----------
        - v : torch.Tensor ~ (batch_size, n_vis) | None. If None and h is
            not None, we begin the sampling process with the hidden units
        - h : torch.Tensor ~ (batch_size, n_hid) | None. If None and v is
            not None, we begin the sampling process with the visible units
        - clamp : torch.Tensor ~ (batch_size, n_vis) | None. Will only
            reconstruct elements marked False in the boolean mask
        - n_gibbs : int. Number of Gibbs sampling steps
        - add_noise : bool. Adds noise to the visible units
        - begin_from_randn_init : bool. If True, v_0 is drawn from randn
            with similar shape as v

        Returns
        -------
        - torch.Tensor ~ (batch_size, n_vis)
        - torch.Tensor ~ (batch_size, n_hid)
        """

        std = self._variance().sqrt()
        if clamp is not None:
            clamp = clamp.bool()
        if v is None and h is None:
            v_sample = torch.randn(size=(1, self.n_vis), generator=self.rng)
        elif v is None:
            v_sample = self._prob_v_given_h(h)
        else:
            if begin_from_randn_init and n_gibbs > 0:
                v_sample = torch.randn(size=(v.shape[0], v.shape[1]), 
                                       generator=self.rng)
            else:
                v_sample = v.clone()
        m = v_sample.shape[0]
        h_sample = torch.bernoulli(self._prob_h_given_v(v_sample),
                                    generator=self.rng)
        for _ in range(n_gibbs):
            if clamp is not None:
                old_v_sample = v_sample.clone()
            v_sample = self._prob_v_given_h(h_sample)
            if add_noise:
                v_sample += torch.randn(size=(m, self.n_vis),
                                generator=self.rng) * std
            if clamp is not None:
                v_sample[clamp] = old_v_sample[clamp]
            h_sample = torch.bernoulli(self._prob_h_given_v(v_sample),
                                        generator=self.rng)
        return v_sample, h_sample
    
    def _variance(self):
        """
        Returns the variance; we only attempt to train the log variance.

        @returns
        - torch.Tensor | float
        """
        return torch.exp(self.log_var)
    
    @torch.no_grad()
    def reconstruct(self, v: np.ndarray, clamp: np.ndarray = None,
                    randn_init = True, n_gibbs: int = 1, add_noise=True):
        """
        Reconstructs the visible units.

        @args
        - v: np.array ~ (batch_size, n_vis)
        - clamp: boolean np.array ~ (batch_size, n_vis) << will only
            reconstruct elements marked False in the boolean mask
        - randn_init: bool << if True, reset v via torch.randn
        - n_gibbs: int

        @returns
        - np.array ~ (batch_size, n_vis)
        """
        v = torch.Tensor(v).requires_grad_(False)
        if clamp is not None:
            v_sample, _ = self._block_gibbs_sample(v=v, n_gibbs=n_gibbs,
                clamp=torch.Tensor(clamp), add_noise=add_noise,
                begin_from_randn_init=randn_init)
        else:
            v_sample, _ = self._block_gibbs_sample(v=v, n_gibbs=n_gibbs,
                add_noise=add_noise, begin_from_randn_init=randn_init)
        return v_sample.numpy()
    
    def cd_loss(self, v: np.ndarray, n_gibbs: int = 1):
        """
        Computes the contrastive divergence loss with which parameters may be
        updated via an optimizer. Follows Algorithm 3 of
        https://arxiv.org/pdf/2210.10318. Contrastive divergence loss 
        may be combined with an adversarial loss as described in 
        https://arxiv.org/abs/1804.08682. 

        @args
        - v: np.array ~ (batch_size, n_vis)
        - n_gibbs: int

        @returns
        - torch.Tensor ~ (1) << contrastive divergence loss
        """
        v_data: torch.Tensor = torch.Tensor(v)
        _, h_data = self._block_gibbs_sample(v_data, n_gibbs=0)
        v_model, h_model = self._block_gibbs_sample(torch.rand_like(v_data), 
                                                    n_gibbs=n_gibbs)
        L = self._energy(v_data, h_data) - self._energy(v_model, h_model)
        return L.mean()
    
    @torch.no_grad()
    def _positive_grad(self, v: torch.Tensor):
        """
        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        """
        _, h = self._block_gibbs_sample(v=v, n_gibbs=0)
        grad = self._energy_grad_param(v, h)
        return grad
    
    @torch.no_grad()
    def _negative_grad(self, v: torch.Tensor, n_gibbs = 1):
        """
        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        - n_gibbs: int << number of Gibbs sampling steps to sample from model
        """
        v, h = self._block_gibbs_sample(v, n_gibbs=n_gibbs)
        grad = self._energy_grad_param(v, h)
        return grad

    @torch.no_grad()
    def _energy_grad_param(self, v: torch.Tensor, h: torch.Tensor):
        """
        Computes the gradient of energy with respect to parameter averaged 
        over the batch size. See the repository associated with the paper 
        https://arxiv.org/pdf/2210.10318:
        https://github.com/DSL-Lab/GRBM/blob/main/grbm.py.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        - h: torch.Tensor ~ (batch_size, n_hid)

        @returns
        - dict
        """
        var = self._variance()
        grad = {}
        grad["W"] = -torch.einsum("bi,bj->ij", v / var, h) / v.shape[0]
        grad["b"] = -h.mean(dim=0)
        grad["mu"] = ((self.mu - v) / var).mean(dim=0)
        grad["log_var"] = (-0.5 * (v - self.mu)**2 / var +
                            ((v / var) * h.mm(self.W.T))).mean(dim=0)
        return grad
    
    @torch.no_grad()
    def cd_grad(self, v: np.ndarray, n_gibbs = 1):
        """
        Updates gradients of the parameters. 

        @args
        - v: np.ndarray ~ (batch_size, n_vis)
        - n_gibbs: int << number of Gibbs sampling steps to sample from model
        """
        v_tensor = torch.Tensor(v)
        pos = self._positive_grad(v_tensor)
        neg = self._negative_grad(v_tensor, n_gibbs)
        for name, param in self.named_parameters():
            param.grad = pos[name] - neg[name]


    def fit(self, X: np.ndarray, n_gibbs: int = 1,
            lr: float = 0.1, n_epochs: int = 100, batch_size: int = 10,
            fail_tol: int = None,
            rng_seed: int = 0, verbose_interval: int = None, 
            reduce_lr_on_plateau = False):
        """
        Built-in, simple train method. Gradients are computed analytically. 
        Robust to NaNs in X. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_examples, n_vis)
        - n_gibbs : int. Number of Gibbs sampling steps (k in the CD-k loss), 
            literature recommends to keep at 1
        - lr : float. Learning rate
        - n_epochs: int. Number of epochs
        - batch_size: int
        """
    


        
    






class GaussianBernoulliRBM(BaseGenerativeModel):
    """Restricted Boltzmann machine with flexibly-defined 
    visible and hidden units."""

    def __init__(self, n_vars: int, hidden_size: int):
        """Constructs an RBM that is compatible with multimodal data. 
        """


    def fit(self, X: np.ndarray):
        """Fits the model.
        
        Parameters
        ----------
        - X : np.ndarray ~ (n_examples, n_features)
        """
        pass


    def sample(self, sample_size: int = None):
        """Samples from the learned distribution.

        Parameters
        ----------
        - sample_size : int. If None, returns a 1d array. 

        Returns
        -------
        - np.ndarray ~ (sample_size, n_features) or (n_features)
        """
        pass


    
        
        







