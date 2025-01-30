import os
import copy
import numpy as np
import torch
import torch.nn as nn
import random
import argparse

from torch import optim
from datetime import datetime

import json 
import pkbar
import shutil

from models.Diffusion.gsgm import GSGM

from dataloader.create_data import hpDIRCCherenkovPhotons
from dataloader.dataloader import CreateLoaders


def train(config, resume, overwrite = False):
    # Setup
    # Setup random seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Wait till you have access to a GPU.")
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Create experiment name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    exp_name = exp_name[:-11]
    MODEL_NAME = "GSGM"
    print("Training",MODEL_NAME,"model.")
    print(exp_name)

    # Create directory structure
    output_folder = config['output']['dir']
    if resume:
        exp_path = os.path.dirname(resume)
    else:
        exp_path = os.path.join(output_folder,exp_name)
    version_num = 0
    
    if not resume:
        if os.path.exists(exp_path):
            if overwrite:
                print(f"Overwriting existing directory: {exp_path}")
                shutil.rmtree(exp_path)
                os.mkdir(exp_path)
            else:
                # Create a new versioned folder if overwrite is not allowed
                version_num = 1
                new_loc = f"{exp_path}_v{version_num}"
                while os.path.exists(new_loc):
                    version_num += 1
                    new_loc = f"{exp_path}_v{version_num}"
                print(f"Existing directory found, creating new version: {new_loc}")
                os.mkdir(new_loc)
                exp_path = new_loc
        else:
            os.mkdir(exp_path)

    # os.mkdir(os.path.join(output_folder,exp_name))
    # with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
    #     json.dump(config, outfile)


       # Load the dataset
    stats = config['stats']

    
    train_dataset = hpDIRCCherenkovPhotons(kaon_path=config['dataset']['training']['smeared']['kaon_data_path'],
                    pion_path=config['dataset']['training']['smeared']['pion_data_path'],inference=False,mode=config['method'],stats=stats)
    # Evaluate on center of pixels
    val_dataset = hpDIRCCherenkovPhotons(kaon_path=config['dataset']['validation']['kaon_data_path'],
                    pion_path=config['dataset']['validation']['pion_data_path'],inference=True,mode=config['method'],stats=stats)

    
    history = {'train_loss':[],'val_loss':[],'lr':[], 'logsnr':[]}
    print("Creating loaders")
    train_loader,val_loader = CreateLoaders(train_dataset,val_dataset,config, model_type=MODEL_NAME)

    num_epochs=int(config['num_epochs'])
    lr = float(config['optimizer']['lr'])
    lr_noise = float(config['optimizer']['lr_noise'])

    input_shape = int(config['model_GSGM']['input_shape'])
    cond_shape = int(config['model_GSGM']['cond_shape'])
    num_layers = int(config['model_GSGM']['num_layers'])
    timesteps = int(config['model_GSGM']['num_steps'])
    num_embed = int(config['model_GSGM']['num_embed'])
    hidden_nodes = int(config['model_GSGM']['hidden_nodes'])
    nonlinear_noise_schedule = bool(config['model_GSGM']['nonlinear_noise_schedule'])
    learned_variance = bool(config['model_GSGM']['learned_variance'])

    startEpoch = 0
    global_step = 0

    save_itter = 50000

    #setup_logging(args.run_name)
    device = torch.device('cuda')

    diffusion = GSGM(num_input=input_shape, 
               num_conds=cond_shape, 
               device=device, 
               stats=stats, # for hpDIRC
               num_layers=num_layers, 
               num_steps=timesteps, 
               num_embed=num_embed, 
               mlp_dim=hidden_nodes, 
               nonlinear_noise_schedule=nonlinear_noise_schedule, 
               learnedvar=learned_variance)

    #optimizer = optim.Adamax(model.parameters(), lr=lr)
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr)

    diffusion.to(device)

    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume, map_location=device)
        diffusion.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        # scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']
        # print("Utilizing grad clipping.")
        print('       ... Start at epoch:',startEpoch)
    
    mse = nn.MSELoss()

    t_params = sum(p.numel() for p in diffusion.parameters())
    print("Score-based Network Parameters: ",t_params)
    if learned_variance:
        noise_net_params = sum(p.numel() for p in diffusion.noise_schedule_net.parameters())
        print("Noise Scheduler Network Parameters: ",noise_net_params)

    #logger = SummaryWriter(os.path.join("runs", args.run_name))

    l = len(train_loader)
    # ema = EMA(0.995)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    diffusion.train()
    running_loss = 0.0
    running_logsnr = 0.0

    for epoch in range(startEpoch,num_epochs):
        # logging.info(f"Starting epoch {epoch}:")

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        for i, data in enumerate(train_loader):
            #inputs = data.to(device).float()
            input  = data[0].float().to(device)
            k = data[1].float().to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                loss, logsnr = diffusion.train_step(input, k, optimizer_noise=None)

            loss.backward()
            optimizer.step()
            
            running_logsnr += logsnr.mean().item()

            running_loss += loss.item() * input.shape[0]
            kbar.update(i, values=[("loss", loss.detach().cpu().numpy().item())])
            #print('\n',logsnr.mean().item())
            global_step += 1

            if i % save_itter == 0 and i > 0:
                name_output_file = config['name']+'_epoch{:02d}_save_iter_{:02d}.pth'.format(epoch, i)
                filename = os.path.join(exp_path , name_output_file)
                checkpoint={}
                checkpoint['net_state_dict'] = diffusion.state_dict()
                checkpoint['optimizer'] = optimizer.state_dict()
                #checkpoint['scheduler'] = scheduler.state_dict()
                checkpoint['epoch'] = epoch
                checkpoint['history'] = history
                checkpoint['global_step'] = global_step

                torch.save(checkpoint,filename)

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(lr)
        history['logsnr'].append(running_logsnr)


        ######################
        ## validation phase ##
        ######################
        if bool(config['run_val']):
            diffusion.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    input  = data[0].to('cuda').float()
                    k = data[1].to('cuda').float()
                    loss, logsnr = diffusion.test_step(input, k)

                    val_loss += loss

            val_loss = val_loss.cpu().numpy() / len(val_loader)

            history['val_loss'].append(val_loss)

            kbar.add(1, values=[("val_loss", val_loss.item())])

            name_output_file = config['name']+'_epoch{:02d}_val_loss_{:.6f}.pth'.format(epoch, val_loss)
        else:
            kbar.add(1,values=[('val_loss',0.)])
            name_output_file = config['name']+'_epoch{:02d}_train_loss_{:.6f}.pth'.format(epoch, running_loss / len(train_loader.dataset))
                
        # Save the output file
        
        filename = os.path.join(exp_path , name_output_file)

        checkpoint={}
        checkpoint['net_state_dict'] = diffusion.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = None
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step
        checkpoint['config'] = config

        torch.save(checkpoint,filename)

        print('')



def launch():
    parser = argparse.ArgumentParser(description='SGM Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the existing experiment directory if it exists.')
    # parser.add_argument('-r', '--resume', default=None, type=str,
    #                     help='Path to the .pth model checkpoint to resume training')
    args = parser.parse_args()

    config = json.load(open(args.config))

    train(config, args.resume, args.overwrite)


if __name__ == '__main__':
    launch()