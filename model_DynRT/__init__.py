from model_DynRT.Optimizers import build_Adam
from model_DynRT.loss_function import build_CrossentropyLoss_ContrastiveLoss, build_BCELoss, build_CrossEntropyLoss, \
    build_CrossEntropyLoss_weighted
import model_DynRT.TRAR
from model_DynRT.DynRT import build_DynRT

_models = {
    "DynRT": build_DynRT
}

_optimizers = {
    "Adam": build_Adam
}

_loss = {
    "CrossEntropyLoss": build_CrossEntropyLoss,
    "BCELoss": build_BCELoss,
    "CrossentropyLoss_ContrastiveLoss": build_CrossentropyLoss_ContrastiveLoss,
    "Crossentropy_Loss_weighted": build_CrossEntropyLoss_weighted
}
