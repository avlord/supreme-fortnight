from jaxrl_m.typing import *
from jaxrl_m.networks import MLP, get_latent, default_init, ensemblize

import flax.linen as nn
import jax.numpy as jnp
from jax import jit, grad, lax, random
import jax

root_key = jax.random.PRNGKey(seed=0)


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x

class AutoEncoder(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
     
       
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x



class ICVFWithEncoder(nn.Module):
    encoder: nn.Module
    vf: nn.Module

    def get_encoder_latent(self, observations: jnp.ndarray) -> jnp.ndarray:     
        return get_latent(self.encoder, observations)
    
    def get_phi(self, observations: jnp.ndarray) -> jnp.ndarray:
        latent = get_latent(self.encoder, observations)
        return self.vf.get_phi(latent)

    def __call__(self, observations, outcomes, intents):
        latent_s = get_latent(self.encoder, observations)
        latent_g = get_latent(self.encoder, outcomes)
        latent_z = get_latent(self.encoder, intents)
        return self.vf(latent_s, latent_g, latent_z)
    
    def get_info(self, observations, outcomes, intents):
        latent_s = get_latent(self.encoder, observations)
        latent_g = get_latent(self.encoder, outcomes)
        latent_z = get_latent(self.encoder, intents)
        return self.vf.get_info(latent_s, latent_g, latent_z)

def create_icvf(icvf_cls_or_name, encoder=None, ensemble=True, **kwargs):    
    if isinstance(icvf_cls_or_name, str):
        icvf_cls = icvfs[icvf_cls_or_name]
    else:
        icvf_cls = icvf_cls_or_name

    if ensemble:
        vf = ensemblize(icvf_cls, 2, methods=['__call__', 'get_info', 'get_phi'])(**kwargs)
    else:
        vf = icvf_cls(**kwargs)
    
    if encoder is None:
        return vf

    return ICVFWithEncoder(encoder, vf)



##
#
# Actual ICVF definitions below
##

class ICVFTemplate(nn.Module):

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Returns useful metrics
        raise NotImplementedError
    
    def get_phi(self, observations):
        # Returns phi(s) for downstream use
        raise NotImplementedError
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        # Returns V(s, g, z)
        raise NotImplementedError

class MonolithicVF(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        network_cls = LayerNormMLP if self.use_layer_norm else MLP
        self.net = network_cls((*self.hidden_dims, 1), activate_final=False)

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        x = jnp.concatenate([observations, outcomes, z], axis=-1)
        v = self.net(x)
        return {
            'v': jnp.squeeze(v, -1),
            'psi': outcomes,
            'z': z,
            'phi': observations,
        }
    
    def get_phi(self, observations):
        print('Warning: StandardVF does not define a state representation phi(s). Returning phi(s) = s')
        return observations
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, outcomes, z], axis=-1)
        v = self.net(x)
        return jnp.squeeze(v, -1)

class MultilinearVF(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool = False

    def setup(self):
        
        network_cls = LayerNormMLP if self.use_layer_norm else MLP
        self.phi_net = network_cls(self.hidden_dims, activate_final=True, name='phi')
        self.psi_net = network_cls(self.hidden_dims, activate_final=True, name='psi')


        self.encoder = AutoEncoder((256,256), activate_final=True, name='encoder')
        self.decoder = AutoEncoder((12,29), activate_final=True, name='decoder') ###HERE IT SHOULD BE ORIGINAL intention space

        self.T_net =  network_cls(self.hidden_dims, activate_final=True, name='T')

        self.matrix_a = nn.Dense(self.hidden_dims[-1], name='matrix_a')
        self.matrix_b = nn.Dense(self.hidden_dims[-1], name='matrix_b')
        
    
    def __call__(self, observations: jnp.ndarray, outcomes: jnp.ndarray, intents: jnp.ndarray) -> jnp.ndarray:
        res = self.get_info(observations, outcomes, intents)
        return (res['v'], res['elbo'])
        
    def gaussian_sample(self,rng, mu, sigmasq):
        """Sample a diagonal Gaussian."""
         ###FIX ME###
        return mu + jnp.sqrt(sigmasq) * random.normal(rng, mu.shape)

    def get_phi(self, observations):
        return self.phi_net(observations)


    def encode(self,z):
        output  = self.encoder(z)
     
        return output
    
    def decode(self,z):
        z = self.decoder(z)
        return z

    # def encode(self,z):
    #     output, prng  = self.encoder(z)
     
    #     return (output[:,0], output[:,1])
    
    # def decode(self,z):
    #     z, prng = self.decoder(z)
    #     return z, prng

    # def bernoulli_logpdf(self,logits, x):
    #     """Bernoulli log pdf of data x given logits."""
    #     return -jnp.sum(jnp.logaddexp(0., jnp.where(x, -1., 1.) * logits))

    # def gaussian_kl(self,mu, sigmasq):
    #     """KL divergence from a diagonal Gaussian to the standard Gaussian."""
    #     return -0.5 * jnp.sum(1. + jnp.log(sigmasq) - mu**2. - sigmasq)

    def mean_squared_error(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """Calculate mean squared error between two tensors.

        Args:
                x1: variable tensor
                x2: variable tensor, must be of same shape as x1

        Returns:
                A scalar representing mean square error for the two input tensors.
        """
        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 must be of the same shape")

        x1 = jnp.reshape(x1, (x1.shape[0], -1))
        x2 = jnp.reshape(x2, (x2.shape[0], -1))

        return jnp.mean(jnp.square(x1 - x2), axis=-1)

    def kl_gaussian(self,mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
        r"""Calculate KL divergence between given and standard gaussian distributions.

        KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
                = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
                = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)

        Args:
            mean: mean vector of the first distribution
            var: diagonal vector of covariance matrix of the first distribution

        Returns:
            A scalar representing KL divergence of the two Gaussian distributions.
        """
        return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)

    def get_info(self, observations: jnp.ndarray, outcomes: jnp.ndarray, intents: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        phi = self.phi_net(observations)
        psi = self.encode(outcomes)
        
        #Before#
        z = self.encode(intents)
        elbo = 0
        Tz = self.T_net(z)


        # #AFTER
        # mu_z, sigmasq_z = self.encode(intents)
        # mu_z = mu_z.reshape(-1,1)
        # sigmasq_z = sigmasq_z.reshape(-1,1)
        # gaussian_sample = mu_z + sigmasq_z * random.normal(prng, mu_z.shape)
        # gaussian_sample = gaussian_sample.reshape(-1,1)
        # z,prng = self.decode(gaussian_sample)
        # z = z.reshape(-1,29) #256, 29
        # ll = self.mean_squared_error(z , intents)
        # gaussian_kl =  self.kl_gaussian(mu_z, sigmasq_z)
        # elbo = ll - gaussian_kl
        # Tz = self.T_net(z)
        # #################


        #AFTER AE
        # z = self.encode(intents)
        z = self.decode(z)
        # z = reconstructed.reshape(-1,29) #256, 29
        ll = self.mean_squared_error(z , intents)
        # elbo = ll
        # Tz = self.T_net(reconstructed)
        #################


        # T(z) should be a dxd matrix, but having a network output d^2 parameters is inefficient
        # So we'll make a low-rank approximation to T(z) = (diag(Tz) * A * B * diag(Tz))
        # where A and B are (fixed) dxd matrices and Tz is a d-dimensional parameter dependent on z

        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        v = (phi_z * psi_z).sum(axis=-1)

        return {
            'v': v,
            'phi': phi,
            'psi': psi,
            'Tz': Tz,
            'z': z,
            'phi_z': phi_z,
            'psi_z': psi_z,
            'elbo':elbo,
        }

icvfs = {
    'multilinear': MultilinearVF,
    'monolithic': MonolithicVF,
}