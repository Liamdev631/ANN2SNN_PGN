import numpy as np
import torch
from tqdm import tqdm
from utils import *
import torch.distributed as dist
import random
import os
from spikingjelly.activation_based import functional
from torch.utils.tensorboard import SummaryWriter   
import wandb
import torch.optim as optim

def seed_all(seed=42):
    print(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_ann(train_dataloader, test_dataloader, model, epochs, device, loss_fn, lr=0.1, lr_min=1e-5,
              wd=5e-4 , save=None, parallel=False, rank=0):
    # model.cuda(device)
    # writer = SummaryWriter('./runs/'+save)
    # mt=monitor.InputMonitor(model,SteppedReLU)
    # qcfs_vth={}
    # cnt=1
    # for name in mt.monitored_layers:
    #     qcfs=get_module_by_name(model,name)[1]
    #     #assert isinstance(qcfs,QCFS)
    #     qcfs_vth[str(cnt)+'+'+name]=qcfs.v_threshold
    #     #qcfs_p0[str(cnt)+'+'+name]=qcfs.p0
    #     cnt=cnt+1

    # mt.clear_recorded_data()
    # mt.remove_hooks()
    # wandb.login()
    # run = wandb.init(
    #     project="ANN2SNN_PNC",
    #     config={"lr": lr, "max_epochs": epochs, "weight_decay": wd},
    # )

    if parallel:
        wd=1e-4

    if rank==0:
        with open('./runs/'+save+'_log.txt','a') as log:
            log.write('lr={},epochs={},wd={}\n'.format(lr,epochs,wd))

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr, betas=[0.9, 0.999])

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=lr_min, T_max=epochs)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5, threshold=1e-3, min_lr=1e-5) # Accuracy
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)

    best_acc = eval_ann(test_dataloader, model, loss_fn, device, rank)[0]
    if parallel:
        dist.all_reduce(best_acc)
        best_acc /= dist.get_world_size()
    if rank == 0:
        print(best_acc)
    for epoch in tqdm(range(epochs)):
        model.train()
        if parallel:
            train_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0
        length = 0
        tot = torch.tensor(0.).cuda(device)
        model.train()
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            length += len(label)
            tot += (label==out.max(1)[1]).sum().data
        train_acc = tot / length

        val_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)
        if parallel:
            dist.all_reduce(val_acc)
            val_acc /= dist.get_world_size()
        if rank == 0 and save != None and val_acc >= best_acc:
            torch.save(model.state_dict(), './saved_models/' + save + '.pth')
        if rank == 0:
            train_loss = epoch_loss / length
            info='Epoch:{},Train_loss:{},Val_loss:{},Train_acc:{},Val_acc:{},lr:{}'.format(epoch, train_loss, val_loss, train_acc.item(), val_acc.item(), scheduler.get_last_lr()[0])
            with open('./runs/'+save+'_log.txt','a') as log:
                log.write(info+'\n')
            #run.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss, "val/acc": val_acc.item})
            
        best_acc = max(val_acc, best_acc)
        # print('Epoch:{},Train_loss:{},Val_loss:{},Acc:{}'.format(epoch, epoch_loss/length,val_loss, tmp_acc), flush=True)
        # print(f'lr={scheduler.get_last_lr()[0]}')
        # print('best_acc: ', best_acc)

        # writer.add_scalars('Acc',{'val_acc':tmp_acc,'best_acc':best_acc},epoch)
        # writer.add_scalars('Loss',{'train_loss':epoch_loss/length,'val_loss':val_loss},epoch)
        # writer.add_scalar('lr',scheduler.get_last_lr()[0],epoch)
        # writer.add_scalars('vth',qcfs_vth,epoch)
        scheduler.step(val_acc)
        #print(module)
    # writer.close()
    return best_acc, model

def eval_snn(test_dataloader, model, loss_fn, device, sim_len=8, rank=0) -> tuple[torch.Tensor, torch.Tensor]:
    tot = torch.zeros(sim_len).cuda()
    loss = torch.zeros(sim_len).cuda()
    length = 0
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, (img, label) in enumerate((test_dataloader)):
            spikes = 0
            length += len(label)
            img = img.cuda()
            label = label.cuda()
            for t in range(sim_len):
                out = model(img)
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum()
                loss[t] = loss_fn(spikes / t, label)
            functional.reset_net(model)
    return tot.detach().cpu().numpy() / length, loss.detach().cpu().numpy() / length

def eval_ann(test_dataloader, model, loss_fn, device, rank=0):
    epoch_loss = 0.0
    tot = torch.tensor(0.).cuda(device)
    model.eval()
    model.cuda(device)
    length = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            out = model(img)
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)    
            tot += (label==out.max(1)[1]).sum().data
    return tot.detach().cpu().numpy() / length, epoch_loss / length
