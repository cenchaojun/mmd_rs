from torch import nn

from ..functions.sigmoid_focal_loss_old import sigmoid_focal_loss_old


class SigmoidFocalLossOld(nn.Module):

    def __init__(self, gamma, alpha):
        super(SigmoidFocalLossOld, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        assert logits.is_cuda
        loss = sigmoid_focal_loss_old(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
