import re
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import argparse
import json

def convert_indices(pmtID,pixelID):
    gapx =  1.89216111455965 + 4.
    gapy = 1.3571428571428572 + 4.
    pixel_width = 3.3125
    pixel_height = 3.3125
    
    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    
    return x,y#,row,col

def make_plots_fastsim(file_path,label,momentum,theta,outpath,filename,log_norm=True):
    data = np.load(file_path,allow_pickle=True)
    xs = []
    ys = []
    time = []
    
    #true_pixel = []
    #true_pmt = []
    true_xs = []
    true_ys = []
    true_time = []
    gapx =  1.89216111455965 + 4.
    gapy = 1.3571428571428572 + 4.
    pixel_width = 3.3125
    pixel_height = 3.3125

    bins_x = [0, 3.65625 - pixel_width/2,  3.65625 + pixel_width/2 ,   6.96875 + pixel_width/2   ,  10.28125 + pixel_width/2  ,  13.59375 +  pixel_width/2   ,
        16.90625+  pixel_width/2  ,  20.21875 + pixel_width/2  ,  23.53125 + pixel_width/2  ,  26.84375 + pixel_width/2  ,
        30.15625 + pixel_width/2  ,  33.46875  + pixel_width/2 ,  36.78125 + pixel_width/2  ,  40.09375 + pixel_width/2  ,
        43.40625 + pixel_width/2  ,  46.71875  + pixel_width/2 ,  50.03125+ pixel_width/2   ,  53.34375 + pixel_width/2  , 62.54841111 - pixel_width/2,
        62.54841111+ pixel_width/2,  65.86091111+ pixel_width/2,  69.17341111+ pixel_width/2,  72.48591111+ pixel_width/2,
        75.79841111+ pixel_width/2,  79.11091111+ pixel_width/2,  82.42341111+ pixel_width/2,  85.73591111+ pixel_width/2,
        89.04841111+ pixel_width/2,  92.36091111+ pixel_width/2,  95.67341111+ pixel_width/2,  98.98591111+ pixel_width/2,
       102.29841111+ pixel_width/2, 105.61091111+ pixel_width/2, 108.92341111+ pixel_width/2, 112.23591111+ pixel_width/2, 121.44057223 - pixel_width/2,
       121.44057223+ pixel_width/2, 124.75307223+ pixel_width/2, 128.06557223+ pixel_width/2, 131.37807223+ pixel_width/2,
       134.69057223+ pixel_width/2, 138.00307223+ pixel_width/2, 141.31557223+ pixel_width/2, 144.62807223+ pixel_width/2,
       147.94057223+ pixel_width/2, 151.25307223+ pixel_width/2, 154.56557223+ pixel_width/2, 157.87807223+ pixel_width/2,
       161.19057223+ pixel_width/2, 164.50307223+ pixel_width/2, 167.81557223+ pixel_width/2, 171.12807223+ pixel_width/2, 180.33273334 - pixel_width/2,
       180.33273334+ pixel_width/2, 183.64523334+ pixel_width/2, 186.95773334+ pixel_width/2, 190.27023334+ pixel_width/2,
       193.58273334+ pixel_width/2, 196.89523334+ pixel_width/2, 200.20773334+ pixel_width/2, 203.52023334+ pixel_width/2,
       206.83273334+ pixel_width/2, 210.14523334+ pixel_width/2, 213.45773334+ pixel_width/2, 216.77023334+ pixel_width/2,
       220.08273334+ pixel_width/2, 223.39523334+ pixel_width/2, 226.70773334+ pixel_width/2, 230.02023334+ pixel_width/2, 239.22489446 - pixel_width/2,
       239.22489446+ pixel_width/2, 242.53739446+ pixel_width/2, 245.84989446+ pixel_width/2, 249.16239446+ pixel_width/2,
       252.47489446+ pixel_width/2, 255.78739446+ pixel_width/2, 259.09989446+ pixel_width/2, 262.41239446+ pixel_width/2,
       265.72489446+ pixel_width/2, 269.03739446+ pixel_width/2, 272.34989446+ pixel_width/2, 275.66239446+ pixel_width/2,
       278.97489446+ pixel_width/2, 282.28739446+ pixel_width/2, 285.59989446+ pixel_width/2, 288.91239446+ pixel_width/2, 298.11705557 - pixel_width/2,
      298.11705557+ pixel_width/2, 301.42955557+ pixel_width/2, 304.74205557+ pixel_width/2, 308.05455557+ pixel_width/2,
       311.36705557+ pixel_width/2, 314.67955557+ pixel_width/2, 317.99205557+ pixel_width/2, 321.30455557+ pixel_width/2,
       324.61705557+ pixel_width/2, 327.92955557+ pixel_width/2, 331.24205557+ pixel_width/2, 334.55455557+ pixel_width/2,
       337.86705557+ pixel_width/2, 341.17955557+ pixel_width/2, 344.49205557+ pixel_width/2, 347.80455557+ pixel_width/2, 347.80455557 + pixel_width/2 + 2]
    
    bins_y = [0, 3.65625 - pixel_height/2,  3.65625 + pixel_height/2, 6.96875+ pixel_height/2   ,  10.28125+ pixel_height/2   ,  13.59375+ pixel_height/2   ,
        16.90625+ pixel_height/2   ,  20.21875+ pixel_height/2   ,  23.53125+ pixel_height/2   ,  26.84375+ pixel_height/2   ,
        30.15625+ pixel_height/2   ,  33.46875+ pixel_height/2   ,  36.78125+ pixel_height/2   ,  40.09375+ pixel_height/2   ,
        43.40625+ pixel_height/2   ,  46.71875+ pixel_height/2   ,  50.03125+ pixel_height/2   ,  53.34375+ pixel_height/2   , 62.01339286 - pixel_height/2.,
        62.01339286+ pixel_height/2,  65.32589286+ pixel_height/2,  68.63839286+ pixel_height/2,  71.95089286+ pixel_height/2,
        75.26339286+ pixel_height/2,  78.57589286+ pixel_height/2,  81.88839286+ pixel_height/2,  85.20089286+ pixel_height/2,
        88.51339286+ pixel_height/2,  91.82589286+ pixel_height/2,  95.13839286+ pixel_height/2,  98.45089286+ pixel_height/2,
       101.76339286+ pixel_height/2, 105.07589286+ pixel_height/2, 108.38839286+ pixel_height/2, 111.70089286+ pixel_height/2, 120.37053571 - pixel_height/2.,
       120.37053571+ pixel_height/2, 123.68303571+ pixel_height/2, 126.99553571+ pixel_height/2, 130.30803571+ pixel_height/2,
       133.62053571+ pixel_height/2, 136.93303571+ pixel_height/2, 140.24553571+ pixel_height/2, 143.55803571+ pixel_height/2,
       146.87053571+ pixel_height/2, 150.18303571+ pixel_height/2, 153.49553571+ pixel_height/2, 156.80803571+ pixel_height/2,
       160.12053571+ pixel_height/2, 163.43303571+ pixel_height/2, 166.74553571+ pixel_height/2, 170.05803571+ pixel_height/2, 178.72767857 - pixel_height/2.,
       178.72767857+ pixel_height/2, 182.04017857+ pixel_height/2, 185.35267857+ pixel_height/2, 188.66517857+ pixel_height/2,
       191.97767857+ pixel_height/2, 195.29017857+ pixel_height/2, 198.60267857+ pixel_height/2, 201.91517857+ pixel_height/2,
       205.22767857+ pixel_height/2, 208.54017857+ pixel_height/2, 211.85267857+ pixel_height/2, 215.16517857+ pixel_height/2,
       218.47767857+ pixel_height/2, 221.79017857+ pixel_height/2, 225.10267857+ pixel_height/2, 228.41517857+ pixel_height/2, 228.41517857 + pixel_height/2 + 2 ]
    
    for i in range(len(data['fast_sim'])):
        xs.append(data['fast_sim'][i]['x'])
        ys.append(data['fast_sim'][i]['y'])
        time.append(data['fast_sim'][i]['leadTime'])
        x_,y_ = convert_indices(data['truth'][i]['pmtID'],data['truth'][i]['pixelID'])
        true_xs.append(x_)
        true_ys.append(y_)
        true_time.append(data['truth'][i]['leadTime'])
    
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    time = np.concatenate(time)
    true_xs = np.concatenate(true_xs)
    true_ys = np.concatenate(true_ys)
    true_time = np.concatenate(true_time)

    fig, ax = plt.subplots(2,2,figsize=(18,12))
    ax = ax.ravel()
    if log_norm:
        norm = LogNorm()
    else:
        norm = None
        
    ax[0].hist2d(xs,ys,bins=[bins_x,bins_y],norm=norm,density=True)
    ax[0].set_title(label + " Fast Simulated Hit Pattern",fontsize=20)
    ax[0].tick_params(axis="both", labelsize=18) 

    t_bins = np.arange(0,100,0.5)
    ax[1].hist(time,bins=t_bins,density=True,histtype='step',color='k')
    ax[1].set_xlabel("Fast Simulated Time (ns)",fontsize=20)
    ax[1].tick_params(axis="both", labelsize=18) 
    ax[1].set_ylabel("A.U.",fontsize=20)
    ax[1].set_title(r"$|\vec{p}|$ ="+r" {0} GeV/c,".format(momentum) + r" $\theta = {0}^o$".format(theta),fontsize=20)
    
    ax[2].hist2d(true_xs,true_ys,bins=[bins_x,bins_y],norm=norm,density=True)
    ax[2].set_title(label + " Geant4 Hit Pattern",fontsize=20)
    ax[2].tick_params(axis="both", labelsize=18) 

    t_bins = np.arange(0,100,0.5)
    ax[3].hist(true_time,bins=t_bins,density=True,histtype='step',color='k')
    ax[3].set_xlabel("Geant4 Time (ns)",fontsize=20)
    ax[3].tick_params(axis="both", labelsize=18) 
    ax[3].set_ylabel("A.U.",fontsize=20)
    ax[3].set_title(r"$|\vec{p}|$ ="+r" {0} GeV/c,".format(momentum) + r" $\theta = {0}^o$".format(theta),fontsize=20)
    
    plt.subplots_adjust(wspace=0.2,hspace=0.3)
    save_path = os.path.join(outpath,filename[:-3]+".png")
    plt.savefig(save_path,bbox_inches="tight")
    plt.close()


def combine_images_to_pdf(image_folder, output_pdf, images_per_page=(2, 2), figure_size=(8, 6)):
    # Create a PdfPages object to save the PDF
    with PdfPages(output_pdf) as pdf:
        images = [f for f in os.listdir(image_folder) if f.endswith('.png')]

        # Sort images by theta (extracted from the file names)
        images.sort(key=lambda x: float(re.search(r'theta_(\d+\.\d+)', x).group(1)))

        # Calculate the number of pages needed
        num_images = len(images)
        images_per_page_total = images_per_page[0] * images_per_page[1]
        num_pages = (num_images + images_per_page_total - 1) // images_per_page_total

        for page in range(num_pages):
            fig, axes = plt.subplots(images_per_page[0], images_per_page[1], figsize=(images_per_page[1] * figure_size[0], images_per_page[0] * figure_size[1]))
            axes = axes.ravel()  # Flatten the 2D grid of subplots into a 1D array

            for i, ax in enumerate(axes):
                img_index = page * images_per_page_total + i
                if img_index >= num_images:
                    ax.axis('off')  # Hide empty subplots
                else:
                    img_path = os.path.join(image_folder, images[img_index])
                    img = Image.open(img_path)
                    ax.imshow(img)
                    #ax.set_title(images[img_index], fontsize=10)
                    ax.axis('off')  # Hide axes ticks

            plt.tight_layout()
            pdf.savefig(fig)  # Save current figure to the PDF
            plt.close(fig)


def main(config,args):
    print("Making plots for generations.")
    print("The folder used: ",str(config['Inference']['fixed_point_dir']))
    print(" ")

    file_folder = os.path.join("Generations",config['Inference']['fixed_point_dir'])
    file_list = os.listdir(file_folder)
    outpath = os.path.join(file_folder,"2DPlots")
    os.makedirs(outpath,exist_ok=True)

    for file in file_list:
        if ".pkl" in file:
            if "Kaon" in file:
                label = 'Kaon'
            elif "Pion" in file:
                label = 'Pion'
                
            match = re.search(r'theta_(\d+\.\d+)', file)
            if match:
                theta_value = match.group(1)
                file_path = os.path.join(file_folder,file)
                make_plots_fastsim(file_path=file_path,label=label,momentum=args.momentum,theta=theta_value,outpath=outpath,filename=file)
                print("Made plot for ", label, " at theta=",theta_value," momentum=",args.momentum)
                
            else:
                print('Cant find theta')
        else:
            continue

    print("Combining images into a single PDF.")

    pdf_output = os.path.join(outpath,"Combined_FastSim_Plots.pdf")
    combine_images_to_pdf(outpath,pdf_output,images_per_page=(2,3),figure_size=(8,8))



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Plotting.')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-p', '--momentum', default=6.0,type=float,help='Particle Momentum.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args)