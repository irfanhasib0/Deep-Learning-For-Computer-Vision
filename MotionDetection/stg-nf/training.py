import os
import time
import shutil
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


class Trainer:
    def __init__(self, model, train_loader, test_loader,loss_func=None,
                 optimizer_f=None, scheduler_f=None):
        self.model         = model
        self.lr            = 0.001
        self.optimizer     = 'adam'
        self.weight_decay  = 5e-5
        self.ckpt_dir = ''
        self.epochs        = 20
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

        state['args'] = self

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

    def train(self, log_writer=None, clip=100):
        time_str = time.strftime("%b%d_%H%M_")
        checkpoint_filename = time_str + '_checkpoint.pth.tar'
        start_epoch = 0
        num_epochs = self.epochs
        self.model.train()
        self.model = self.model.to(self.device)
        key_break = False
        for epoch in range(start_epoch, num_epochs):
            if key_break:
                break
            print("Starting Epoch {} / {}".format(epoch + 1, num_epochs))
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                
                data =  data_arr[0].to(self.device, non_blocking=True)
                data = data.permute((0,3,1,2))
                #score = data[-2].amin(dim=-1)
                #label = data[-1]
                #if self.model_confidence:
                #    samp = data[0]
                #else:
                samp = data[:, :2]
                z= self.model(samp.float())
                #if nll is None:
                #    continue
                #if self.model_confidence:
                #    nll = nll * score
                losses = self.loss_func(z)
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                pbar.set_description("Loss: {}".format(losses.item()))
                log_writer.add_scalar('NLL Loss', losses.item(), epoch * len(self.train_loader) + itern)

            self.save_checkpoint(epoch, filename=checkpoint_filename)
            new_lr = self.adjust_lr(epoch)
            print('Checkpoint Saved. New LR: {0:.3e}'.format(new_lr))

    def test(self):
        self.model.eval()
        self.model.to(self.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.device)
        print("Starting Test Eval")
        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.device, non_blocking=True) for data in data_arr]
            score = data[-2].amin(dim=-1)
            if self.model_confidence:
                samp = data[0]
            else:
                samp = data[0][:, :2]
            with torch.no_grad():
                z, nll = self.model(samp.float(), label=torch.ones(data[0].shape[0]), score=score)
            if self.model_confidence:
                nll = nll * score
            probs = torch.cat((probs, -1 * nll), dim=0)
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        return prob_mat_np

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state
