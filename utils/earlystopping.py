import numpy as np
import torch

class EarlyStopping2Class:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.F1_1 = 0
        self.F1_2 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss,accs,F1_1,F1_2,model,path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1_1 = F1_1
            self.F1_2 = F1_2
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST LOSS:{:.4f}| Accuracy: {:.4f}|F1_1: {:.4f}|F1_2: {:.4f}"
                      .format(-self.best_score,self.accs,self.F1_1,self.F1_2))
        else:
            self.best_score = score
            self.accs = accs
            self.F1 = F1_1
            self.F2 = F1_2
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class EarlyStopping4Class:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.F1_1 = 0
        self.F1_2 = 0
        self.F1_3 = 0
        self.F1_4 = 0
        self.val_loss_min = np.Inf

    def __call__(self,val_loss,accs,F1_1,F1_2,F1_3,F1_4,model,path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1_1 = F1_1
            self.F1_2 = F1_2
            self.F1_3 = F1_3
            self.F1_4 = F1_4
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}"
                      .format(self.accs,self.F1_1,self.F1_2,self.F1_3,self.F1_4))
        else:
            self.best_score = score
            self.accs = accs
            self.F1_1 = F1_1
            self.F1_2 = F1_2
            self.F1_3 = F1_3
            self.F1_4 = F1_4
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss