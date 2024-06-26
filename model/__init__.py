from model.Optimizers import build_Adam
from model.loss_function import build_CrossentropyLoss_ContrastiveLoss, build_BCELoss, build_CrossEntropyLoss, build_CrossEntropyLoss_weighted
import model.TRAR
from model.DyCR-Net import build_DyCR-Net
_models={
    "DyCR-Net":build_DyCR-Net
}

_optimizers={
    "Adam":build_Adam
}

_loss={
    "CrossEntropyLoss":build_CrossEntropyLoss,
    "BCELoss":build_BCELoss,
    "CrossentropyLoss_ContrastiveLoss": build_CrossentropyLoss_ContrastiveLoss,
    "Crossentropy_Loss_weighted": build_CrossEntropyLoss_weighted
}