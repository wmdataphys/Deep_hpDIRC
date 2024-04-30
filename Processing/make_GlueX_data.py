import json 
import numpy as np
import argparse
import pickle
import os
import pkbar

def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)

def extract_json(file_path):
    #data_dict = {'EventID':None, 'PDG':None, 'NHits':None, 'BarID':None, 'P':None, 'Theta':None, 'Phi':None, 'X':None, 'Y':None, 'Z':None, 'pmtID':None, 'pixelID':None, 'channel':None, 'leadTime':None}
    data_dict = {}
    EventID = []
    PDG = []
    NHits = []
    BarID = []
    P = []
    Theta = []
    Phi = []
    X = []
    Y = []
    Z = []
    LikelihoodElectron = []
    LikelihoodPion = []
    LikelihoodKaon = []
    LikelihoodProton = []
    pmtID = []
    pixelID = []
    channel = []
    leadTime = []
    invMass = []
    i = 0
    n_events = count_lines(file_path)
    kbar = pkbar.Kbar(target=n_events, width=20, always_stateful=False)
    print("Extracting data from json file.")
    with open(file_path) as f:
        for line in f:
            #data.append(json.loads(line))
            data = json.loads(line)
            EventID.append(data['EventID'])
            PDG.append(data['PDG'])
            NHits.append(data['NHits'])
            BarID.append(data['BarID'])
            P.append(data['P'])
            Theta.append(data['Theta'])
            Phi.append(data['Phi'])
            X.append(data['X'])
            Y.append(data['Y'])
            Z.append(data['Z'])
            LikelihoodElectron.append(data['LikelihoodElectron'])
            LikelihoodPion.append(data['LikelihoodPion'])
            LikelihoodKaon.append(data['LikelihoodKaon'])
            LikelihoodProton.append(data['LikelihoodProton'])
            invMass.append(data['invMass'])
            pmtID.append(data['pmtID'])
            pixelID.append(data['pixelID'])
            channel.append(data['channel'])
            leadTime.append(data['leadTime'])
            kbar.update(i)
            i+= 1
            
            #if i == 10:
            #    break
    
    print(" ")
    print('Converting to dictionary.')
    data_dict['EventID']  =      np.array(EventID)
    data_dict['PDG']  =        np.array(PDG)
    data_dict['NHits']  =        np.array(NHits)
    data_dict['BarID']  =        np.array(BarID)
    data_dict['P']  =        np.array(P)
    data_dict['Theta']  =        np.array(Theta)
    data_dict['Phi']  =        np.array(Phi)
    data_dict['X']  =        np.array(X)
    data_dict['Y']  =        np.array(Y)
    data_dict['Z']  =        np.array(Z)
    data_dict['invMass'] = np.array(invMass)
    data_dict['LikelihoodElectron'] = np.array(LikelihoodElectron)
    data_dict['LikelihoodPion'] = np.array(LikelihoodPion)
    data_dict['LikelihoodKaon'] = np.array(LikelihoodKaon)
    data_dict['LikelihoodProton'] = np.array(LikelihoodProton)
    data_dict['pmtID']  =        pmtID
    data_dict['pixelID']  =        pixelID
    data_dict['channel']  =        channel
    data_dict['leadTime']  =        leadTime      

    print(" Done " ,len(data_dict))
    return data_dict

def parse_data(data_dict):
    #clean_events = [] # For GlueX data we need to consider both cases under one file. Simulation use notebook.
    #PIDS = [-321,321] # You need to change this for Phi (K+,K-,321,-321) or Rho (pi+,pi-, 211,-22) decays
    print(" ")
    print("Parsing dictionary.")
    pions = []
    kaons = []
    PIDS = [-321,321,211,-211] 
    event_with_other_PID = 0
    conditional_maxes = np.array([8.5,11.63,175.5])
    conditional_mins = np.array([0.95,0.90,-176.])
    kbar = pkbar.Kbar(target=len(np.unique(data_dict['EventID'])), width=20, always_stateful=False)
    l = 0
    # For actual data
    # rho_mass_upper = 0.9
    # rho_mass_lower = 0.6
    # phi_mass_lower = 1.0
    # phi_mass_upper = 1.04
    # For particle gun -> Inv mass has no meaning here
    rho_mass_upper = np.inf
    rho_mass_lower = -np.inf
    phi_mass_lower = -np.inf
    phi_mass_upper = np.inf
    for j in np.unique(data_dict['EventID']):
        idx = np.where(data_dict['EventID'] == j)[0]
        ys = data_dict['Y'][idx]
        #print('PDG',data_dict['PDG'][idx])
        #print(j)
        if len(idx) > 2:
            print('Skipping event.')
            kbar.update(l)
            l+=1
            continue
            
        if len(idx) > 1:
            if (ys[0] > 0) and (ys[1] > 0):
                kbar.update(l)
                l+=1 
                continue
            elif (ys[0] < 0) and (ys[1] < 0):
                kbar.update(l)
                l+= 1
                continue
            else:
                temp_oboxes = np.array(data_dict['pmtID'][idx[0]])//108
                #print(temp_oboxes)
                hits_in_obox0 = np.where(temp_oboxes == 0)[0]
                hits_in_obox1 = np.where(temp_oboxes == 1)[0]
                obox_0_idx = np.where(data_dict['Y'][idx] < 0)[0][0]
                obox_1_idx = np.where(data_dict['Y'][idx] > 0)[0][0]
                
                pmt_obox_0 = np.array(data_dict['pmtID'][idx[0]])[hits_in_obox0]
                pmt_obox_1 = np.array(data_dict['pmtID'][idx[0]])[hits_in_obox1]
                pixel_obox_0 = np.array(data_dict['pixelID'][idx[0]])[hits_in_obox0]
                pixel_obox_1 = np.array(data_dict['pixelID'][idx[0]])[hits_in_obox1]
                leadTime_obox_0 = np.array(data_dict['leadTime'][idx[0]])[hits_in_obox0]
                leadTime_obox_1 = np.array(data_dict['leadTime'][idx[0]])[hits_in_obox1]
                channel_obox_0 = np.array(data_dict['channel'][idx[0]])[hits_in_obox0]
                channel_obox_1 = np.array(data_dict['channel'][idx[0]])[hits_in_obox1]
                
                #print(pmt_obox_0)
                
                
                event0 = {'EventID':idx[0],'invMass':data_dict['invMass'][idx][obox_0_idx], 'PDG':data_dict['PDG'][idx][obox_0_idx], 'NHits':len(pmt_obox_0), 'BarID':data_dict['BarID'][idx][obox_0_idx], 'P':data_dict['P'][idx][obox_0_idx], 'Theta':data_dict['Theta'][idx][obox_0_idx], 'Phi':data_dict['Phi'][idx][obox_0_idx], 'X':data_dict['X'][idx][obox_0_idx], 'Y':data_dict['Y'][idx][obox_0_idx], 'Z':data_dict['Z'][idx][obox_0_idx],
                        'pmtID':pmt_obox_0, 'pixelID':pixel_obox_0, 'channel':channel_obox_0, 'leadTime':leadTime_obox_0,'LikelihoodElectron':data_dict['LikelihoodElectron'][idx][obox_0_idx],'LikelihoodPion':data_dict['LikelihoodPion'][idx][obox_0_idx],'LikelihoodKaon':data_dict['LikelihoodKaon'][idx][obox_0_idx],'LikelihoodProton':data_dict['LikelihoodProton'][idx][obox_0_idx]}
                event1 = {'EventID':idx[0],'invMass':data_dict['invMass'][idx][obox_1_idx], 'PDG':data_dict['PDG'][idx][obox_1_idx], 'NHits':len(pmt_obox_1), 'BarID':data_dict['BarID'][idx][obox_1_idx], 'P':data_dict['P'][idx][obox_1_idx], 'Theta':data_dict['Theta'][idx][obox_1_idx], 'Phi':data_dict['Phi'][idx][obox_1_idx], 'X':data_dict['X'][idx][obox_1_idx], 'Y':data_dict['Y'][idx][obox_1_idx], 'Z':data_dict['Z'][idx][obox_1_idx],
                        'pmtID':pmt_obox_1, 'pixelID':pixel_obox_1, 'channel':channel_obox_1, 'leadTime':leadTime_obox_1,'LikelihoodElectron':data_dict['LikelihoodElectron'][idx][obox_1_idx],'LikelihoodPion':data_dict['LikelihoodPion'][idx][obox_1_idx],'LikelihoodKaon':data_dict['LikelihoodKaon'][idx][obox_1_idx],'LikelihoodProton':data_dict['LikelihoodProton'][idx][obox_1_idx]}
                
            if (event0['PDG'] in PIDS) and (len(pmt_obox_0) < 300) and (event0['P'] < conditional_maxes[0]) and (event0['P'] > conditional_mins[0]) and (event0['Theta'] < conditional_maxes[1]) and (event0['Theta'] > conditional_mins[1]) and (event0['Phi'] < conditional_maxes[2]) and (event0['Phi'] > conditional_mins[2]):
                #clean_events.append(event0)
                if (abs(event0['PDG']) == 321) and (event0['invMass'] > phi_mass_lower) and (event0['invMass'] < phi_mass_upper):
                    kaons.append(event0)
                elif (abs(event0['PDG']) == 211) and (event0['invMass'] > rho_mass_lower) and (event0['invMass'] < rho_mass_upper):
                    pions.append(event0) # Else its a pion
                else:
                    #print('Skipped')
                    continue
            if (event1['PDG'] in PIDS) and (len(pmt_obox_1) < 300)  and (event1['P'] < conditional_maxes[0]) and (event1['P'] > conditional_mins[0]) and (event1['Theta'] < conditional_maxes[1]) and (event1['Theta'] > conditional_mins[1]) and (event1['Phi'] < conditional_maxes[2]) and (event1['Phi'] > conditional_mins[2]):
                #clean_events.append(event1)
                if (abs(event1['PDG']) == 321) and (event1['invMass'] > phi_mass_lower) and (event1['invMass'] < phi_mass_upper):
                    kaons.append(event1)
                elif (abs(event1['PDG']) == 211) and (event1['invMass'] > rho_mass_lower) and (event1['invMass'] < rho_mass_upper):
                    pions.append(event1) # Else its a pion
                else:
                    #print("skipped")
                    continue

            if event1['PDG'] not in PIDS or event0['PDG'] not in PIDS:
                event_with_other_PID +=1
            
        else:
            if (data_dict['PDG'][idx] in PIDS) and (data_dict['P'][idx] < conditional_maxes[0]) and (data_dict['P'][idx] > conditional_mins[0]) and (data_dict['Theta'][idx] < conditional_maxes[1]) and (data_dict['Theta'][idx] > conditional_mins[1]) and (data_dict['Phi'][idx] < conditional_maxes[2]) and (data_dict['Phi'][idx] > conditional_mins[2]) :
                idx = idx[0]
                temp_oboxes = np.array(data_dict['pmtID'][idx])//108
                if len(np.unique(temp_oboxes)) > 1:
                    y = data_dict['Y'][idx]
                    if y < 0:
                        hits_in_obox = np.where(temp_oboxes == 0)[0]
                    elif y > 0:
                        hits_in_obox = np.where(temp_oboxes == 1)[0]
                        
                    pmt_obox = np.array(data_dict['pmtID'][idx])[hits_in_obox]
                    pixel_obox = np.array(data_dict['pixelID'][idx])[hits_in_obox]
                    leadTime_obox = np.array(data_dict['leadTime'][idx])[hits_in_obox]
                    channel_obox = np.array(data_dict['channel'][idx])[hits_in_obox]
             
                    event = {'EventID':idx,'invMass':data_dict['invMass'][idx], 'PDG':data_dict['PDG'][idx], 'NHits':len(pmt_obox), 'BarID':data_dict['BarID'][idx], 'P':data_dict['P'][idx], 'Theta':data_dict['Theta'][idx], 'Phi':data_dict['Phi'][idx], 'X':data_dict['X'][idx], 'Y':data_dict['Y'][idx], 'Z':data_dict['Z'][idx],
                        'pmtID':pmt_obox, 'pixelID':pixel_obox, 'channel':channel_obox, 'leadTime':leadTime_obox,'LikelihoodElectron':data_dict['LikelihoodElectron'][idx],'LikelihoodPion':data_dict['LikelihoodPion'][idx],'LikelihoodKaon':data_dict['LikelihoodKaon'][idx],'LikelihoodProton':data_dict['LikelihoodProton'][idx]}
                        
                else:  
                    event = {'EventID':idx,'invMass':data_dict['invMass'][idx], 'PDG':data_dict['PDG'][idx], 'NHits':data_dict['NHits'][idx], 'BarID':data_dict['BarID'][idx], 'P':data_dict['P'][idx], 'Theta':data_dict['Theta'][idx], 'Phi':data_dict['Phi'][idx], 'X':data_dict['X'][idx], 'Y':data_dict['Y'][idx], 'Z':data_dict['Z'][idx],
                            'pmtID':data_dict['pmtID'][idx], 'pixelID':data_dict['pixelID'][idx], 'channel':data_dict['channel'][idx], 'leadTime':data_dict['leadTime'][idx],'LikelihoodElectron':data_dict['LikelihoodElectron'][idx],'LikelihoodPion':data_dict['LikelihoodPion'][idx],'LikelihoodKaon':data_dict['LikelihoodKaon'][idx],'LikelihoodProton':data_dict['LikelihoodProton'][idx]}
                
                if (abs(event['PDG']) == 321) and (event['invMass'] > phi_mass_lower) and (event['invMass'] < phi_mass_upper):
                    kaons.append(event)
                elif (abs(event['PDG']) == 211) and (event['invMass'] > rho_mass_lower) and (event['invMass'] < rho_mass_upper):
                    pions.append(event) # Else its a pion
                else:
                    #print('Skipped single track')
                    continue
                #clean_events.append(event)
        kbar.update(l)
        l+=1

    print('Done.')
    return pions,kaons


def main(args):
    file_path = os.path.join(args.base_dir,args.file)
    print('Processing ' + str(file_path))
    data_dict = extract_json(file_path)
    pions,kaons = parse_data(data_dict)
    file = str(args.file).split("/")[-1]
    out_dir = os.path.join(args.base_dir,"processed")

    if not os.path.exists(out_dir):
        print("Creating output directory: ",out_dir)
        os.mkdir(out_dir)

    print(out_dir)

    kaon_file = file[:-5] + "_Kaons.pkl"
    out_path_kaons = os.path.join(out_dir,str(kaon_file))
    print('Writing kaons to ' + str(out_path_kaons))
    with open(out_path_kaons,"wb") as k_file:
        pickle.dump(kaons,k_file)


    pion_file = file[:-5] + "_Pions.pkl"
    out_path_pions = os.path.join(out_dir,str(pion_file))
    print('Writing pions to ' + str(out_path_pions))
    with open(out_path_pions,"wb") as p_file:
        pickle.dump(pions,p_file)

    
if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('-f', '--file', default="hd_root_071725.json", type=str,
                        help='Path to the .json data folder.')
    parser.add_argument('-d', '--base_dir', default="/sciclone/data10/jgiroux/Cherenkov/Real_Data/json", type=str,
                        help='.json File name.')
    args = parser.parse_args()

    main(args)


    
