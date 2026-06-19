import torch

def noise_scheduler(beta_min = 1e-4, beta_max = 0.02, T = 500):

    print("Adopting a linear scheduler")

    
    ts = torch.arange(T, dtype=torch.float32)
    betas = beta_min +  ts/(T-1) * (beta_max - beta_min)

    
    alphas = 1.0 - betas
    alpha_hats = torch.cumprod(alphas, dim = 0)
    alpha_hats_sqrt = torch.sqrt(alpha_hats)
    one_minus_alpha_hats_sqrt = torch.sqrt((1.0 - alpha_hats))


    alpha_hats_prev = torch.cat([
        torch.ones(1, dtype=alpha_hats.dtype),
        alpha_hats[:-1]
    ])

    posterior_variance = (
        betas * (1.0 - alpha_hats_prev) / (1.0 - alpha_hats)
    )

    
    return {
    "betas": betas,
    "alphas": alphas,
    "alpha_hats": alpha_hats,
    "alpha_hats_sqrt": alpha_hats_sqrt,
    "one_minus_alpha_hats_sqrt": one_minus_alpha_hats_sqrt,
    "posterior_variance": posterior_variance,
    }


def add_noise_onestep(x_prev, t, noise_scheduler):
    added_noise = torch.randn_like(x_prev)
    x_current = noise_scheduler['alphas'][t].sqrt() * x_prev + noise_scheduler['betas'][t].sqrt() * added_noise

    return x_current



def add_noise_from_zero(x0, t, noise_scheduler):
    added_noise = torch.randn_like(x0)
    xt = x0 * noise_scheduler["alpha_hats_sqrt"][t].view(-1,1) + added_noise *  noise_scheduler["one_minus_alpha_hats_sqrt"][t].view(-1, 1)

    return xt, added_noise




