import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation


def plot_batched_lines(lines, labels=None, x=None, max_lines=16, figsize=(8, 3)):
    """
    lines:  Tensor or array with shape [batch_size, line_length]
    labels: Tensor/list with shape [batch_size], optional
    x:      Tensor/array with shape [line_length], optional
    """

    if isinstance(lines, torch.Tensor):
        lines = lines.detach().cpu()

    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()

    n_lines = min(lines.shape[0], max_lines)

    if x is None:
        x = torch.arange(lines.shape[1])

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()

    plt.figure(figsize=figsize)

    for i in range(n_lines):
        if labels is None:
            label = f"sample {i}"
        else:
            label = f"sample {i}, label={labels[i].item():.2f}"

        plt.plot(x, lines[i], label=label)

    plt.xlabel("")
    plt.ylabel("")
    plt.legend()
    plt.tight_layout()
    plt.show()



from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt

def animate_forward_noise(
    image,
    add_noise_onestep,
    scheduler_dict,
    T=500,
    interval=50,
    step=5
):
    x = image.copy()

    # Handle batched or unbatched input
    if x.ndim == 1:
        original_y = x.detach().cpu().numpy()
    else:
        original_y = x.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 3))

    # Fixed original line
    
    # Updating noisy line
    noisy_line, = ax.plot(
        original_y,
        color="navy",
        alpha=0.8,
        linewidth = 1.0,
    )

    ax.plot(
        original_y,
        color="red",
        linewidth=1.5,
    )


    ax.set_ylim(-2, 2)
    ax.legend()
    ax.set_title("Forward diffusion step: t = 0")

    def update(frame):
        nonlocal x

        target_t = frame * step

        while update.current_t < target_t:
            x = add_noise_onestep(x, update.current_t, scheduler_dict)
            update.current_t += 1

        if x.ndim == 1:
            noisy_y = x.detach().cpu().numpy()
        else:
            noisy_y = x.detach().cpu().numpy()

        noisy_line.set_ydata(noisy_y)
        ax.set_title(f"Forward diffusion step: t = {target_t}")

        return noisy_line,

    update.current_t = 0

    anim = FuncAnimation(
        fig,
        update,
        frames=T // step + 1,
        interval=interval,
        blit=False
    )

    plt.close(fig)
    return anim