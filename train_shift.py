import os
import numpy as np
import torch
import torch.nn as nn
import random
import argparse
from torch import optim
from datetime import datetime
import logging
import json 
import pkbar
import shutil

from models.Diffusion.resnet import ResNet, EMA
from models.Diffusion.gaussian_diffusion_shift import ShiftGaussianDiffusion
from models.Diffusion.shift_predictor import TabShiftPredictor

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
    config_name = config['name']
    MODEL_NAME = 'Shift'
    print("Training",MODEL_NAME,"model.")
    print(exp_name)

    # Create directory structure
    output_folder = config['output']['dir']
    if resume:
        exp_path = os.path.dirname(resume)
        config_name = exp_name.split('___')[0]
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

    # Load the dataset
    print('Creating Loaders.')
    stats = config['stats']

    history = {'train_loss':[],'val_loss':[],'lr':[]}
    
    train_dataset = hpDIRCCherenkovPhotons(kaon_path=config['dataset']['training']['smeared']['kaon_data_path'],
                    pion_path=config['dataset']['training']['smeared']['pion_data_path'],inference=False,mode=config['method'],stats=stats)
    # Evaluate on center of pixels
    val_dataset = hpDIRCCherenkovPhotons(kaon_path=config['dataset']['validation']['kaon_data_path'],
                    pion_path=config['dataset']['validation']['pion_data_path'],inference=True,mode=config['method'],stats=stats)

    train_loader,val_loader = CreateLoaders(train_dataset,val_dataset,config, model_type=MODEL_NAME)
    

    num_epochs=int(config['num_epochs'])
    lr = float(config['optimizer']['lr_Shift'])

    # ResNet/MLP parameters
    num_layers = int(config['model_Shift']['num_layers'])
    input_shape = int(config['model_Shift']['input_shape'])
    cond_shape = int(config['model_Shift']['cond_shape'])
    hidden_dim = int(config['model_Shift']['hidden_nodes']) 
    num_steps = int(config['model_Shift']['num_steps'])
    noise_schedule = config['model_Shift']['noise_schedule']
    shift_type = config['model_Shift']['shift_type']

    startEpoch = 0
    global_step = 0


    #setup_logging(args.run_name)
    device = torch.device('cuda')

    model = ResNet(input_dim=input_shape, 
                   end_dim=input_shape, 
                   cond_dim=cond_shape, 
                   mlp_dim=hidden_dim, 
                   num_layer=num_layers
                   )

    shift_predictor = TabShiftPredictor(device=device,
                                        input_dim=input_shape, 
                                        cond_dim=cond_shape, 
                                        hidden_dim=hidden_dim)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    diffusion = ShiftGaussianDiffusion(
                                device=device,
                                denoise_fn=model,
                                shift_predictor=shift_predictor,
                                stats=stats,
                                timesteps=num_steps, 
                                noise_schedule=noise_schedule,
                                shift_type=shift_type)

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

    t_params = sum(p.numel() for p in model.parameters())
    print("Diffusion Network Parameters: ",t_params)

    #logger = SummaryWriter(os.path.join("runs", args.run_name))

    l = len(train_loader)
    ema = EMA(0.995)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(startEpoch,num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        
        model.train()
        shift_predictor.train()

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            inp  = data[0].to(device).float()
            conds = data[1].to(device).float()

            loss = diffusion.shift_train_one_batch(
                x_0=inp,
                condition=conds
                )['prediction_loss']
            #backwards(loss, optimizer)
            loss.backward()

            running_loss += loss.item() * inp.shape[0]

            optimizer.step()

            # ema.step_ema(ema_model, model)

            kbar.update(i, values=[("loss", loss.item())])

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(lr)


        ######################
        ## validation phase ##
        ######################
        if bool(config['run_val']):
            model.eval()
            # ema_model.eval()
            running_val_loss = 0.0

            val_kbar = pkbar.Kbar(target=len(val_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inp  = data[0].to('cuda').float()
                    conds = data[1].to('cuda').float()

                    val_loss = diffusion(inp, conds)

                    running_val_loss += val_loss.item() * inp.shape[0]

                    val_kbar.update(i, values=[("val_loss", val_loss.item())])

            epoch_val_loss = running_val_loss / len(val_loader.dataset)

            history['val_loss'].append(epoch_val_loss)

            val_kbar.add(1, values=[("val_loss", epoch_val_loss)])

            name_output_file = config_name+'_epoch{:02d}_val_loss_{:.6f}.pth'.format(epoch, epoch_val_loss)
        else:
            #kbar.add(1,values=[('val_loss',0.)])
            name_output_file = config_name+'_epoch{:02d}_train_loss_{:.6f}.pth'.format(epoch, running_loss / len(train_loader.dataset))

                
        # Save the output file

        # kbar.add(1,values=[('val_loss',0.)])
        # name_output_file = config_name+'_epoch{:02d}_train_loss_{:.6f}.pth'.format(epoch, running_loss / len(train_loader.dataset))
        
        filename = os.path.join(exp_path, name_output_file)

        checkpoint={}
        checkpoint['net_state_dict'] = diffusion.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = None
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

        # if (epoch+1) % 20 == 0: 
        #     print("trying out a sample at epoch {0}...".format(epoch+1))
        #     with torch.no_grad():
        #         sample(config, diffusion, input_shape, epoch+1)

        print('')

def launch():
    parser = argparse.ArgumentParser(description='Diffusion Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the existing experiment directory if it exists.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    train(config, args.resume, args.overwrite)


if __name__ == '__main__':
    launch()