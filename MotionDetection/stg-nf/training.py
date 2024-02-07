import os
import json
import numpy as np
import pandas as pd

import time
import torch
import torch.optim as optim
from tqdm import tqdm


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def compute_loss(nll, reduction="mean", mean=0):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "logsumexp":
        losses = {"nll": torch.logsumexp(nll, dim=0)}
    elif reduction == "exp":
        losses = {"nll": torch.exp(torch.mean(nll) - mean)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses

class Logger():
    def __init__(self, keys=['train_loss','val_acc'], exp_name='exp-101'):
        self.exp_name   = exp_name
        self.exp_dir    = f'logs/{exp_name}/'
        self.src_dir    = f'logs/{exp_name}/src/'
        self.model_dir  = f'logs/{exp_name}/model/'
        self.csv_path = f'{self.exp_dir}results.csv'
        self.arg_path = f'{self.exp_dir}args.json'
        self.keys     = keys
        self.log_df   = pd.DataFrame([],columns=keys)
        self.best_acc = -np.inf
        
        os.makedirs(self.exp_dir,exist_ok=True)
        os.makedirs(self.src_dir,exist_ok=True)
        os.makedirs(self.model_dir,exist_ok=True)
        
    def reset(self):
        self.temp = {key:[] for key in self.keys}
        
    def update(self,res):
        for key,val in res.items():
            if type(val) == torch.Tensor: 
                val = val.detach()
                if val.device.type != 'cpu' :
                    val = val.cpu().numpy()
                else:
                    val = val.numpy()
            self.temp[key].append(val)
            
    def accumulate(self,epoch,state,save_best_only=True):
        self.curr_epoch = epoch
        for key in self.keys:
            self.log_df.loc[epoch,key] = np.mean(self.temp[key])
        if self.log_df.loc[epoch,'val_acc'] > self.best_acc:
            self.best_acc   = self.log_df.loc[epoch,'val_acc']
            if save_best_only:
                os.system('rm -r {self.model_dir}*')
            self.save_model(state)
            
    def save_src(self):
        os.system(f"cp ./*py {self.src_dir}")
        os.system(f"cp ./*ipynb {self.src_dir}")
    
    def save_args(self,args):
        with open(self.arg_path,'w') as file:
             json.dump(args,file)
    
    def save_logs(self):
        self.log_df.to_csv(self.csv_path,index=False)
        self.is_best = False
        
    def save_model(self,state):
        torch.save(state,f'{self.model_dir}{self.curr_epoch}_{self.exp_name}_{round(self.best_acc,2)}.pth')
        
class Trainer:
    def __init__(self, model,
                 train_loader,
                 test_loader,
                 loss_func=None,
                 epochs=5,
                 lr=0.0001,
                 optimizer_f=None,
                 scheduler_f=None):
        self.model         = model
        self.lr            = lr
        self.optimizer     = 'adam'
        self.weight_decay  = 5e-5
        self.epochs        = epochs
        self.device        = 'cuda:0'
        self.model_confidence = False
        self.model_lr         = 5e-4
        self.model_lr_decay   = 0.99 
        self.train_loader     = train_loader
        self.test_loader      = test_loader
        self.loss_func        = loss_func
        # Loss, Optimizer and Scheduler
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.optimizer == 'adam':
            if self.lr:
                return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        elif self.optimizer == 'adamx':
            if self.lr:
                return optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            else:
                return optim.Adamax(self.model.parameters())
        return optim.SGD(self.model.parameters(), lr=self.lr)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.model_lr, self.model_lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        #state['args'] = self

        path_join = os.path.join(self.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = filename
        try:
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.set_actnorm_init()
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(filename, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.ckpt_dir))
    
        
    def train(self, epochs=1,lgr=None):
        #time_str = time.strftime("%b%d_%H%M_")
        self.model.train()
        self.model = self.model.to(self.device)
        lgr.reset()
        for epoch in range(epochs):
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                
                data   = data_arr[0].to(self.device, non_blocking=True)
                data   = data.permute((0,3,1,2))
                out    = self.model(data)
                losses = self.loss_func(*out)
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                self.optimizer.step()
                self.optimizer.zero_grad()
                pbar.set_description("Loss: {}".format(losses['loss']))
                lgr.update({'train_loss':losses['loss']}) 
            
            torch.cuda.empty_cache()
            state  = self.gen_checkpoint_state(epoch)
            new_lr = self.adjust_lr(epoch)
            
        return state

    def test(self,loader=None):
        loader =self.test_loader if loader==None else loader
        
        self.model.eval()
        self.model.to(self.device)
        pbar  = tqdm(loader)
        probs = torch.empty(0).to(self.device)
        mu    = torch.empty(0).to(self.device)
        std   = torch.empty(0).to(self.device)
        results = []
        print("Starting Test Eval")
        for itern, data_arr in enumerate(pbar):
            data_arr = [elem.to(self.device, non_blocking=True) for elem in data_arr]
            data     = data_arr[0].permute(0,3,1,2)
            scr      = data_arr[2].amin(dim=-1)
            with torch.no_grad():
                z = self.model(data[:,:2,:,:],label=torch.ones((data.shape[0])),score=scr)
                #scores = z[1]
                scores = torch.abs(z[0][:,:2] - z[1][:,:2]).mean(dim=[1,2,3])

            probs = torch.cat([probs, scores], dim=0)
            #mu  = torch.cat([mu, torch.abs(z[2])], dim=0)
            #std = torch.cat([std, torch.abs(z[3])], dim=0)
            results.append(z)
        
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        mu  = mu.cpu().detach().numpy()#.squeeze().copy(order='C')
        std = std.cpu().detach().numpy()#.squeeze().copy(order='C')
        
        return prob_mat_np, mu, std, results

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state
