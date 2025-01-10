import os
import json
import argparse
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from dataloader.dataloader import CreateLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from models.nflows_models import create_nflows
from dataloader.create_data import hpDIRCCherenkovPhotons
from datetime import datetime
from models.freia_models import FreiaNet
from models.nflows_models import MAAF
from models.cnf import CNF
from models.cnf import CNFOdeFunc
from models.cnf import get_regularization

def run_warmup(net,loader,num_warmup_batches=10000):
    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, net.parameters())), lr=1e-4,weight_decay=0)

    kbar = pkbar.Kbar(target=num_warmup_batches, width=20, always_stateful=False)

    net.train()

    for i, data in enumerate(loader):
        input  = data[0].to('cuda').float()
        k = data[1].to('cuda').float()

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            loss = -net.log_prob(inputs=input,context=k).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5,error_if_nonfinite=True)
        optimizer.step()

        kbar.update(i, values=[("loss", loss.item())])

        if i == num_warmup_batches:
            break

    return net


def main(config,resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Create experiment name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    exp_name = exp_name[:-11]
    print(exp_name)

    # Create directory structure
    output_folder = config['output']['dir']
    os.mkdir(os.path.join(output_folder,exp_name))
    with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
        json.dump(config, outfile)


       # Load the dataset
    print('Creating Loaders.')
    stats = config['stats']
    
    train_dataset = hpDIRCCherenkovPhotons(kaon_path=config['dataset']['training']['smeared']['kaon_data_path'],
                    pion_path=config['dataset']['training']['smeared']['pion_data_path'],inference=False,mode=config['method'],stats=stats)
    # Evaluate on center of pixels
    val_dataset = hpDIRCCherenkovPhotons(kaon_path=config['dataset']['validation']['kaon_data_path'],
                    pion_path=config['dataset']['validation']['pion_data_path'],inference=True,mode=config['method'],stats=stats)


    history = {'train_loss':[],'val_loss':[],'lr':[]}

    reg_coeffs = [1.0]


    train_loader,val_loader = CreateLoaders(train_dataset,val_dataset,config)

    print("Training Size: {0}".format(len(train_loader.dataset)))
    print("Validation Size: {0}".format(len(val_loader.dataset)))

    # Create the model
    num_layers = int(config['model']['num_layers'])
    input_shape = int(config['model']['input_shape'])
    cond_shape = int(config['model']['cond_shape'])
    num_blocks = int(config['model']['num_blocks'])
    hidden_nodes = int(config['model']['hidden_nodes'])
    net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats)
    #net = CNF(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,train_T=True,T=1.0)
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    device = torch.device('cuda')
    net.to('cuda')

    # Kaons
    #dicte = torch.load("/sciclone/data10/jgiroux/Cherenkov/hpDIRC_Models/Kaon_hpDIRC_20L_8192_Batch_ExtraTracks_50Epoch_2LBlocks_128Node___Oct-26-2024/Kaon_hpDIRC_20L_8192_Batch_ExtraTracks_50Epoch_2LBlocks_128Node_epoch49_val_loss_-2.640291.pth")

    # Pions
    #dicte = torch.load("/sciclone/data10/jgiroux/Cherenkov/hpDIRC_Models/Pion_hpDIRC_20L_8192_Batch_ExtraTracks_50Epoch_2LBlocks_128Node___Oct-26-2024/Pion_hpDIRC_20L_8192_Batch_ExtraTracks_50Epoch_2LBlocks_128Node_epoch49_val_loss_-2.646162.pth")
    #net.load_state_dict(dicte['net_state_dict'])

    # Optimizer
    num_epochs=int(config['num_epochs'])
    lr = float(config['optimizer']['lr'])
    weight_decay = float(config['optimizer']['weight_decay'])
    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, net.parameters())), lr=lr,weight_decay=weight_decay)
    num_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
                                                           eta_min=0)


    startEpoch = 0
    global_step = 0

    warm_start = bool(config['warm_start'])

    save_itter = 1000
    alpha = config['optimizer']['alpha']

    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:',startEpoch)


    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      num_epochs:', num_epochs)
    print('')


    if warm_start:
        print("Running warmup.")
        net = run_warmup(net,train_loader)
        print("Finished warmup. Starting training.")
        print(" ")
    
    for epoch in range(startEpoch,num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            input  = data[0].to('cuda').float()
            k = data[1].to('cuda').float()

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                loss = -net.log_prob(inputs=input,context=k).mean()
                #logp,grad_log_p = net.loss_function(inputs=input,context=k)
                #loss = (-logp + alpha*grad_log_p).mean()


            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5,error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * input.shape[0]
            kbar.update(i, values=[("loss", loss.item())])#,("logp",-logp.mean().item()),("grad_logp",alpha*grad_log_p.mean().item())])
            global_step += 1


        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])


        ######################
        ## validation phase ##
        ######################
        if bool(config['run_val']):
            net.eval()
            val_loss = 0.0
            val_logp = 0.0
            val_grad_logp = 0.0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    input  = data[0].to('cuda').float()
                    k = data[1].to('cuda').float()
                    loss = -net.log_prob(inputs=input,context=k).mean()
                    #logp,grad_log_p = net.loss_function(inputs=input,context=k)

                    #loss = (-logp + alpha*grad_log_p).mean()

                    val_loss += loss
                    #val_logp += -logp.mean()
                    #val_grad_logp += alpha*grad_log_p.mean()

            val_loss = val_loss.cpu().numpy() / len(val_loader)
            #val_logp = val_logp.cpu().numpy() / len(val_loader)
            #val_grad_logp = val_grad_logp.cpu().numpy() / len(val_loader)


            history['val_loss'].append(val_loss)

            kbar.add(1, values=[("val_loss", val_loss.item())])#,("val_logp",val_logp.item()),("val_grad",val_grad_logp.item())])

            name_output_file = config['name']+'_epoch{:02d}_val_loss_{:.6f}.pth'.format(epoch, val_loss)

        else:
            kbar.add(1,values=[('val_loss',0.)])
            name_output_file = config['name']+'_epoch{:02d}_train_loss_{:.6f}.pth'.format(epoch, running_loss / len(train_loader.dataset))

        filename = os.path.join(output_folder , exp_name , name_output_file)

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

        print('')




if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.resume)
