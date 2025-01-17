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
from dataloader.create_data import hpDIRCCherenkovPhotons
from datetime import datetime
from models.OT_Flow.ot_flow import OT_Flow
from models.gsgm import GSGM 

warnings.filterwarnings("ignore", message=".*weights_only.*")

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

    train_loader,val_loader = CreateLoaders(train_dataset,val_dataset,config,model_type="CNF")

    print("Training Size: {0}".format(len(train_loader.dataset)))
    print("Validation Size: {0}".format(len(val_loader.dataset)))

    # Create the model
    dtype_ = torch.float32
    num_layers = int(config['model_CNF']['num_layers'])
    input_shape = int(config['model_CNF']['input_shape'])
    cond_shape = int(config['model_CNF']['cond_shape'])
    num_blocks = int(config['model_CNF']['num_blocks'])
    hidden_nodes = int(config['model_CNF']['hidden_nodes'])
    alph = config['model_CNF']['alph']
    net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=True,alph=alph)
    #net = GSGM(input_shape, cond_shape, device, num_layers, timesteps, num_embed, hidden_nodes, nonlinear_noise_schedule, learned_variance)
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    device = torch.device('cuda')
    net.to('cuda')
    net.to(dtype_)


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

    save_itter = 10000

    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']
        print("Utilizing grad clipping.")
        print('       ... Start at epoch:',startEpoch)


    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      num_epochs:', num_epochs)
    print('')
    loss_fn = nn.HuberLoss()

    clampMax = 1.5
    clampMin = -1.5

    for epoch in range(startEpoch,num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            input  = data[0].to('cuda', non_blocking=True).to(dtype_)
            k = data[1].to('cuda', non_blocking=True).to(dtype_)

            optimizer.zero_grad()

            for p in net.parameters():
                p.data = torch.clamp(p.data, clampMin, clampMax)

            with torch.set_grad_enabled(True):
                loss, costs = net.compute_loss(inputs=input,nt=6,context=k)

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0,error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()


            running_loss += loss.item() * input.shape[0]
            kbar.update(i, values=[("loss", loss.item()),("costC",costs[0].item()),("costL", costs[1].item()),("costR",costs[-1].item()),("T:",net.end_time.item())])
            global_step += 1

            if i % save_itter == 0 and i > 0:
                name_output_file = config['name']+'_epoch{:02d}_save_iter_{:02d}.pth'.format(epoch, i)
                filename = os.path.join(output_folder , exp_name , name_output_file)
                checkpoint={}
                checkpoint['net_state_dict'] = net.state_dict()
                checkpoint['optimizer'] = optimizer.state_dict()
                checkpoint['scheduler'] = scheduler.state_dict()
                checkpoint['epoch'] = epoch
                checkpoint['history'] = history
                checkpoint['global_step'] = global_step

                torch.save(checkpoint,filename)


        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])


        ######################
        ## validation phase ##
        ######################
        if bool(config['run_val']):
            net.eval()
            val_loss = 0.0
            val_costC = 0.0
            val_costL = 0.0
            val_costR = 0.0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    input  = data[0].to('cuda', non_blocking=True).to(dtype_)
                    k = data[1].to('cuda', non_blocking=True).to(dtype_)
                    loss,costs = net.compute_loss(inputs=input,nt=6,context=k)
                    val_costC += costs[0]
                    val_costL += costs[1]
                    val_costR += costs[-1]
                    val_loss += loss

            val_loss = val_loss.cpu().numpy() / len(val_loader)
            val_costC = val_costC.cpu().numpy() / len(val_loader)
            val_costL = val_costL.cpu().numpy() / len(val_loader)
            val_costR = val_costR.cpu().numpy() / len(val_loader)

            history['val_loss'].append(val_loss)

            kbar.add(1, values=[("val_loss", val_loss.item()),("valC",val_costC.item()),("valL",val_costL.item()),("valR",val_costR.item())])

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

    if not os.path.exists("Trained_Models"):
        print("Creating trained models folder. This will only occur once.")
        os.makedirs("Trained_Models")

    main(config,args.resume)
