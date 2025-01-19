import pandas as pd
from dataloader.create_data import hpDIRCCherenkovPhotons
from dataloader.dataloader import CreateLoaders
import json

pion_path = '/sciclone/data10/jgiroux/Cherenkov/hpDIRC/FullPhaseSpace/Training_Pions_hpDIRC_Smeared.feather'
kaon_path = "/sciclone/data10/jgiroux/Cherenkov/hpDIRC/FullPhaseSpace/Training_Kaons_hpDIRC_Smeared.feather"
stats = {"x_max": 350.0,"x_min":2.0,"y_max":230.1,"y_min":2.0,"time_max":157.00,"time_min":0.0,"P_max":10.0 ,"P_min":0.5 ,"theta_max": 160.0,"theta_min": 25.0}
config_path = '/sciclone/home/mcmartinez/Deep_hpDIRC/config/hpDIRC_config_gulf.json'
config = json.load(open(config_path))

try:
    train_dataset = hpDIRCCherenkovPhotons(kaon_path=kaon_path,
                    pion_path=pion_path,inference=False,mode='Pion',stats=stats)
    # Evaluate on center of pixels
    val_dataset = hpDIRCCherenkovPhotons(kaon_path=kaon_path,
                    pion_path=pion_path,inference=True,mode='Pion',stats=stats)
    
    print("Creating loaders")
    train_loader,val_loader = CreateLoaders(train_dataset,val_dataset,config)
    print("Successfully made them")
except Exception as e:
    print(f"Error reading Feather file: {e}")