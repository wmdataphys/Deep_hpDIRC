import numpy as np
import random
import pickle
import os
import argparse



def main(args):

    list_of_files = os.listdir(args.base_dir)
    method = str(args.method)
    data = []
    print("Creating datasets for ",method)
    i = 0
    #kaon_length = 1400683
    kaon_length = 270498
    n_pions = int(3 * kaon_length)
    for file in list_of_files:
        path_ = os.path.join(args.base_dir,file)

        if method in file:
            data += np.load(path_,allow_pickle=True)
            i += 1
        
            if i % 100 == 0:
                print("File {0}:".format(i)," ",len(data))

        if method == "Pions":
            if i > 300:
                break


    

    print("Total {0}".format(method),len(data))
    #random.shuffle(data)

    with open(os.path.join(args.base_dir,"{0}_RealData.pkl".format(method)),"wb") as file:
        pickle.dump(data,file)

    # Nt = int(0.7 * len(data))
    # print("Training: ",Nt)
    # train = data[:Nt]
    # test_val = data[Nt:]

    # Nv = int(0.5 * len(test_val))
    # val = test_val[:Nv]
    # test = test_val[Nv:]
    # print("Validation: ",len(val))
    # print("Testing: ",len(test))

    # with open(os.path.join(args.base_dir,"Training_{0}_RealData.pkl".format(method)),"wb") as file:
    #     pickle.dump(train,file)

    # with open(os.path.join(args.base_dir,"Validation_{0}_RealData.pkl".format(method)),"wb") as file:
    #     pickle.dump(val,file)

    # with open(os.path.join(args.base_dir,"Testing_{0}_RealData.pkl".format(method)),"wb") as file:
    #     pickle.dump(test,file)

    
if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Dataset creation')
    parser.add_argument('-f', '--method', default="Kaons", type=str,
                        help='Path to the .json data folder.')
    parser.add_argument('-d', '--base_dir', default="/sciclone/data10/jgiroux/Cherenkov/Real_Data/json", type=str,
                        help='.json File name.')
    args = parser.parse_args()

    main(args)