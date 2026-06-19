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


def add_noise_onestep(x_prev, t, scheduler_dict):
    added_noise = torch.randn_like(x_prev)
    x_current = scheduler_dict['alphas'][t].sqrt() * x_prev + scheduler_dict['betas'][t].sqrt() * added_noise

    return x_current





def add_noise_onestep_cum_sampler(x_0, t, scheduler_dict, no_samples = 1000):
    output = []
    for _ in range(no_samples):
        x = x_0
        for i in range(t):
            x = add_noise_onestep(x, i+1, scheduler_dict)

        output.append(x)

    return torch.cat(output, dim = 0)



def add_noise_from_zero(x0, t, noise_dict):
    added_noise = torch.randn_like(x0)
    xt = x0 * noise_dict["alpha_hats_sqrt"][t].view(-1,1) + added_noise *  scheduler_dict["one_minus_alpha_hats_sqrt"][t].view(-1, 1)

    return xt, added_noise




