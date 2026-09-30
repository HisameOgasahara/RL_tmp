import matplotlib.pyplot as plt

def plot_state_trajectories(
    times,
    reference,
    prediction=None,
    labels=("DOP853", "prediction"),
    title="Cart-Pole trajectory",
):
    names = ["x", "theta", "x_dot", "theta_dot"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))

    for index, axis in enumerate(axes.ravel()):
        axis.plot(times[:len(reference)], reference[:, index], label=labels[0])
        if prediction is not None:
            axis.plot(
                times[:len(prediction)],
                prediction[:, index],
                "--",
                label=labels[1],
            )
        axis.set_xlabel("t")
        axis.set_ylabel(names[index])
        axis.grid(alpha=0.2)
        axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    plt.show()


def plot_training(history, title):
    plt.figure(figsize=(8, 4))

    for key, values in history.items():
        plt.plot(values, label=key)

    plt.xlabel("step/update")
    plt.ylabel("metric")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()
