import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(
        self, optimizer, d_model, n_warmup_steps, aneal_steps, aneal_rate, current_steps
    ):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = current_steps
        self.aneal_steps = aneal_steps
        self.aneal_rate = aneal_rate
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps,
            ]
        )
        for s in self.aneal_steps:
            if self.n_current_steps > s:
                lr = lr * self.aneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
