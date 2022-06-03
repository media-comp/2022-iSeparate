from torch.utils.tensorboard import SummaryWriter

from iSeparate.utils.plotting_utils import plot_spectrogram_to_numpy


class TBLogger(SummaryWriter):
    def __init__(self, logdir):
        super(TBLogger, self).__init__(logdir)

    def log_training(self, loss, learning_rate, iteration):
        for k, v in loss.items():
            if k == "iteration":
                continue
            self.add_scalar(f"training.{k}", v, iteration)
        # self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)

        self.flush()

    def log_validation(self, reduced_loss, model, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace(".", "/")
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        self.flush()

    def log_specs(self, iteration, x, y, y_hat):

        self.add_image("mixture_spec", plot_spectrogram_to_numpy(x), iteration, dataformats="HWC")
        self.add_image("target_spec", plot_spectrogram_to_numpy(y), iteration, dataformats="HWC")
        self.add_image(
            "prediction_spec", plot_spectrogram_to_numpy(y_hat), iteration, dataformats="HWC"
        )

    def log_audio(self, x, y, y_hat, targets):
        self.add_audio("mixture_wav", x)
        if len(y_hat.shape) == 3:
            for i in range(y_hat.shape[0]):
                self.add_audio("target_wav ({})".format(targets[i]), y[i])
                self.add_audio("predicted_wav ({})".format(targets[i]), y_hat[i])
        else:
            self.add_audio("predicted_wav ({})".format(targets[0]), y_hat)
