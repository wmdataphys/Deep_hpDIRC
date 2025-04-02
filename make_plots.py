import re
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import argparse
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pdf2image import convert_from_path 
from utils.hpDIRC import bins_x,bins_y,gapx,gapy,pixel_width,pixel_height

t_bins = np.arange(9.0,157.0,0.5)

def label_mom(rho):
    # works when rho is a range or if rho is just a float. 
    try:
        fvalue = float(rho)
        if fvalue.is_integer():
            return str(int(fvalue))
        else:
            return str(fvalue)
    except ValueError:
        return str(rho)

def convert_indices(pmtID,pixelID): 
    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    
    return x,y

def make_plots_fastsim(file_path,label,momentum,theta,outpath,filename,log_norm=True):
    data = np.load(file_path,allow_pickle=True)
    xs = []
    ys = []
    time = []
    true_xs = []
    true_ys = []
    true_time = []
     
    for i in range(len(data['fast_sim'])):
        xs.append(data['fast_sim'][i]['x'])
        ys.append(data['fast_sim'][i]['y'])
        time.append(data['fast_sim'][i]['leadTime'])
        x_,y_ = convert_indices(data['truth'][i]['pmtID'],data['truth'][i]['pixelID'])
        true_xs.append(x_)
        true_ys.append(y_)
        true_time.append(data['truth'][i]['leadTime'])
    
    xs = np.concatenate(xs).astype('float32')
    ys = np.concatenate(ys).astype('float32')
    time = np.concatenate(time)
    true_xs = np.concatenate(true_xs).astype('float32')
    true_ys = np.concatenate(true_ys).astype('float32')
    true_time = np.concatenate(true_time)

    if log_norm:
        norm = LogNorm()
    else:
        norm = None

    gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 0.5, 1])

    text_momentum = label_mom(momentum)

    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right
    ax3 = fig.add_subplot(gs[2, :])  # Bottom image, spans both columns

    ax3.set_position([
        ax1.get_position().x0 - 0.05 + (ax2.get_position().x0 - ax1.get_position().x0) / 2,  # Center horizontally
        ax3.get_position().y0 - 0.02,  # Keep original y position
        ax1.get_position().width * 1.2,  # Keep same width as top images
        ax1.get_position().height  # Keep same height
    ])

    # FastSim 2D Hit Pattern
    h_fs, xedges, yedges, im1 = ax1.hist2d(xs, ys, bins=[bins_x, bins_y], norm=norm, density=True)
    ax1.set_title(label + r" Fast Simulated Hit Pattern", fontsize=30)
    ax1.tick_params(axis="both", labelsize=28)
    ax1.set_xlabel("X (mm)",fontsize=30,labelpad=15)
    ax1.set_ylabel("Y (mm)",fontsize=30,labelpad=15)
    # Geant4 2D Hit Pattern
    h_g, xedges, yedges, im2 = ax2.hist2d(true_xs, true_ys, bins=[bins_x, bins_y], norm=norm, density=True)
    ax2.set_title(label + r" Geant4 Hit Pattern", fontsize=30)
    ax2.tick_params(axis="both", labelsize=28)
    ax2.set_xlabel("X (mm)",fontsize=30,labelpad=15)   

    # Bottom row: Combined Time Distribution
    ax3.hist(true_time, bins=t_bins, density=True, histtype='step', color='k', label="Geant4",linewidth=2)
    ax3.hist(time, bins=t_bins, density=True, histtype='step', color='r', label="FastSim.",linewidth=2)
    
    ax3.set_xlabel("Time (ns)", fontsize=30)
    ax3.tick_params(axis="both", labelsize=28)
    ax3.set_ylabel("A.U.", fontsize=30)
    ax3.set_yscale('log')
    ax3.set_ylim(1e-5, 10e-1)
    ax3.text(108, 0.015, r"$|\vec{p}|$" + f" = {text_momentum} GeV/c" "\n" r"$\theta =$"+ f"{int(theta)}" +r"$^\circ$".format(momentum, theta), fontsize=24,
    verticalalignment='top',  # Align text at the top
    bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.3'))
    legend_lines = [
        Line2D([0], [0], color='k', linewidth=2, label="Geant4"),
        Line2D([0], [0], color='r', linewidth=2, label="FastSim.")
    ]
    
    ax3.legend(handles=legend_lines, fontsize=24,loc='upper right')
    save_path = os.path.join(outpath,filename[:-3]+"pdf")
    plt.savefig(save_path,bbox_inches="tight")
    plt.close()

    # Make new folder for radius plots in same directory
    out_path_radius = os.path.join(outpath,"RadiusPlots")
    os.makedirs(out_path_radius,exist_ok=True)

    # Radius vs Time.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax = axes[0]
    hist1 = ax.hist2d(np.sqrt(true_xs**2 + true_ys**2), true_time, bins=200, range=[[0, 417], [0, 100]], norm=LogNorm(), cmap='magma')
    ax.set_xlabel("Radius (mm)", fontsize=25,labelpad=10)
    ax.set_ylabel("Time (ns)", fontsize=25)
    ax.set_xticks(np.linspace(0, 400, 5))
    ax.tick_params(axis='x', labelsize=22, rotation=0)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_title("Geant4", fontsize=30)

    if label == "Pion":
        ax.text(0.03, 0.965, r"$\pi^{+-}$" "\n" rf"{text_momentum} GeV/c " "\n" rf"${int(float(theta))}^\degree$" , 
                transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left',  
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    elif label == "Kaon":
         ax.text(0.03, 0.965, r"$\mathcal{K}^{+-}$" "\n" rf"{text_momentum} GeV/c " "\n" rf"${int(float(theta))}^\degree$" , 
                transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left',  
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))   
    else:
        pass    

    ax = axes[1]
    hist2 = ax.hist2d(np.sqrt(xs**2 + ys**2), time, bins=200, range=[[0, 417], [0, 100]], norm=LogNorm(), cmap='magma')
    ax.set_xlabel("Radius (mm)", fontsize=25,labelpad=10)
    ax.tick_params(axis='x', labelsize=22, rotation=0)
    ax.set_title("Fast Simulated", fontsize=30)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(hist2[3], cax=cax, orientation='vertical', pad=0.02)
    cbar.set_label('Counts', fontsize=25)
    cbar.ax.tick_params(labelsize=22)

    plt.tight_layout()
    save_path = os.path.join(out_path_radius,filename[:-3]+"_RadiusVsTime.pdf")
    plt.savefig(save_path,bbox_inches='tight')
    plt.close()

def combine_images_to_pdf(image_folder, output_pdf, images_per_page=(2, 2), figure_size=(8, 6)):
    images = [f for f in os.listdir(image_folder) if f.endswith('.pdf') and "theta" in f]
    
    if not images:
        print("No matching PDF files found.")
        return

    images.sort(key=lambda x: float(re.search(r'theta_(\d+\.\d+)', x).group(1)))

    num_images = len(images)
    images_per_page_total = images_per_page[0] * images_per_page[1]
    num_pages = (num_images + images_per_page_total - 1) // images_per_page_total

    with PdfPages(output_pdf) as pdf:
        for page in range(num_pages):
            fig, axes = plt.subplots(images_per_page[0], images_per_page[1], 
                                     figsize=(images_per_page[1] * figure_size[0], 
                                              images_per_page[0] * figure_size[1]))
            axes = axes.ravel()

            for i, ax in enumerate(axes):
                img_index = page * images_per_page_total + i
                if img_index >= num_images:
                    ax.axis('off')
                else:
                    img_path = os.path.join(image_folder, images[img_index])

                    images_from_pdf = convert_from_path(img_path, first_page=1, last_page=1)
                    if images_from_pdf:
                        ax.imshow(images_from_pdf[0])  
                    ax.axis('off')

            plt.tight_layout()
            pdf.savefig(fig)  
            plt.close(fig)


def make_ratios(path_,label,momentum,outpath):
    file_counter = 0
    x = []
    y = []
    t = []
    x_true = []
    y_true = []
    t_true = []
    for file in os.listdir(path_):
        if label not in file or ".pkl" not in file:
            continue

        file_counter += 1

        file_path = os.path.join(path_,file)
        data = np.load(file_path,allow_pickle=True)
 
        
        for i in range(len(data['fast_sim'])):
            x.append(data['fast_sim'][i]['x'])
            y.append(data['fast_sim'][i]['y'])
            t.append(data['fast_sim'][i]['leadTime'])
            x_,y_ = convert_indices(data['truth'][i]['pmtID'],data['truth'][i]['pixelID'])
            x_true.append(x_)
            y_true.append(y_)
            t_true.append(data['truth'][i]['leadTime'])
    
    if file_counter == 0:
        print("No files found for ",label, ". Exiting.")
        return
    else:
        pass

    x = np.concatenate(x)
    y = np.concatenate(y)
    t = np.concatenate(t)
    x_true = np.concatenate(x_true)
    y_true = np.concatenate(y_true)
    t_true = np.concatenate(t_true)

    
    fig, ax = plt.subplots(2, 3, figsize=(18, 8), gridspec_kw={'height_ratios': [4, 1]}, sharex='col',sharey='row')
    ax = ax.ravel()

    text_momentum = label_mom(momentum)

    # Time PDF
    n_true_t, _ = np.histogram(t_true, bins=t_bins, density=True)
    n_gen_t, _ = np.histogram(t, bins=t_bins, density=True)
    alea_t,_ = np.histogram(t,bins=t_bins,density=False)
    alea_t = np.sqrt(alea_t) / (alea_t.sum() * (t_bins[1] - t_bins[0]))
    ratio_err_t = 3*alea_t / (n_true_t + 1e-50)
    ax[0].hist(t_true, density=True, color='k', label='Truth', bins=t_bins, histtype='step', lw=2)
    ax[0].hist(t, density=True, color='red', label='Generated', bins=t_bins, histtype='step', linestyle='dashed', lw=2)
    ax[0].set_xlabel("Hit Time (ns)", fontsize=20, labelpad=10)
    ax[0].set_ylabel("Density", fontsize=20, labelpad=10)

    # X PDF
    n_true_x, _ = np.histogram(x_true, bins=bins_x, density=True)
    n_gen_x, _ = np.histogram(x, bins=bins_x, density=True)
    alea_x,_ = np.histogram(x,bins=bins_x,density=False)
    alea_x = np.sqrt(alea_x) / (alea_x.sum() * (bins_x[1] - bins_x[0]))
    ratio_err_x = 3*alea_x / (n_true_x + 1e-10)
    ax[1].hist(x_true, density=True, color='k', label='Truth', bins=bins_x, histtype='step', lw=2)
    ax[1].hist(x, density=True, color='red', label='Generated', bins=bins_x, histtype='step', linestyle='dashed', lw=2)
    ax[1].set_xlabel("X (mm)", fontsize=20, labelpad=10)
    ax[1].set_title(str(label)+r" - $|\vec{p}|$ ="+r" {0} GeV/c".format(text_momentum),fontsize=25)

    # Y PDF
    n_true_y, _ = np.histogram(y_true, bins=bins_y, density=True)
    alea_y, _ = np.histogram(y,bins=bins_y,density=False)
    alea_y = np.sqrt(alea_y) / (alea_y.sum() * (bins_y[1] - bins_y[0]))
    ratio_err_y = 3*alea_y / (n_true_y + 1e-10)
    n_gen_y, _ = np.histogram(y, bins=bins_y, density=True)
    ax[2].hist(y_true, density=True, color='k', label='Truth', bins=bins_y, histtype='step', lw=2)
    ax[2].hist(y, density=True, color='red', label='Generated', bins=bins_y, histtype='step', linestyle='dashed', lw=2)
    ax[2].set_xlabel("Y (mm)", fontsize=20, labelpad=10)

    # Ratio plot for Time PDF
    ratio_t = n_gen_t / (n_true_t + 1e-50)
    ratio_t[np.where(n_gen_t == n_true_t)[0]] = 1.0
    ratio_t_upper = ratio_t + ratio_err_t
    ratio_t_lower = ratio_t - ratio_err_t
    ax[3].step((t_bins[:-1] + t_bins[1:]) /2, ratio_t, color='red',linestyle='-',linewidth=1)
    ax[3].fill_between((t_bins[:-1] + t_bins[1:])/2 , ratio_t_lower, ratio_t_upper, color='red', alpha=0.3, step='pre')
    ax[3].set_ylabel('Ratio', fontsize=15)
    ax[3].set_ylim([0, 2])
    ax[3].set_yticks([0.5, 1, 1.5])
    ax[3].axhline(1.0, color='black', linestyle='--', lw=1)

    # Ratio plot for X PDF
    bin_centers_x = [(bins_x[i] + bins_x[i + 1]) / 2 for i in range(len(bins_x) - 1)]
    ratio_x = n_gen_x / (n_true_x + 1e-50)
    ratio_x[np.where(n_gen_x == n_true_x)[0]] = 1.0
    ratio_x_upper = ratio_x + ratio_err_x
    ratio_x_lower = ratio_x - ratio_err_x
    ax[4].step(bin_centers_x, ratio_x,color='red',linestyle='-',linewidth=1)
    ax[4].fill_between(bin_centers_x, ratio_x_lower, ratio_x_upper, color='red', alpha=0.3, step='pre')
    ax[4].set_ylim([0, 2])
    ax[4].set_yticks([0.5, 1, 1.5])
    ax[4].axhline(1.0, color='black', linestyle='--', lw=1)

    # Ratio plot for Y PDF
    bin_centers_y = [(bins_y[i] + bins_y[i + 1]) / 2 for i in range(len(bins_y) - 1)]
    ratio_y = n_gen_y / (n_true_y + 1e-50)
    ratio_y[np.where(n_gen_y == n_true_y)[0]] = 1.0
    ratio_y_upper = ratio_y + ratio_err_y
    ratio_y_lower = ratio_y - ratio_err_y
    ax[5].step(bin_centers_y, ratio_y, color='red',linestyle='-',linewidth=1)
    ax[5].fill_between(bin_centers_y, ratio_y_lower, ratio_y_upper, color='red', alpha=0.3, step='pre')
    ax[5].set_ylim([0, 2])
    ax[5].set_yticks([0.5, 1, 1.5])
    ax[5].axhline(1.0, color='black', linestyle='--', lw=1)

    # Format
    legend_lines = [
        Line2D([0], [0], color='k', linewidth=2, label="Geant4",linestyle='-'),
        Line2D([0], [0], color='r', linewidth=2, label="FastSim.",linestyle='--')
    ]
    
    ax[1].legend(handles=legend_lines,loc="upper center", fontsize=18.5,)
    for i in range(3):
        ax[i].tick_params(axis='both', which='major', labelsize=18)
        ax[i].set_yscale('log')
        
    ax[3].set_xlabel("Time (ns)",fontsize=20)
    ax[4].set_xlabel("X (mm)",fontsize=20)
    ax[5].set_xlabel("Y (mm)",fontsize=20)

    for i in range(3, 6):
        ax[i].tick_params(axis='both', which='major', labelsize=15)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    print(label," ratios.")
    print("X: ",np.average(ratio_x,weights=n_true_x))
    print("Y: ", np.average(ratio_y,weights=n_true_y))
    print("Time: ",np.average(ratio_t,weights=n_true_t))

    print("RMS X: ",np.sqrt(np.average((ratio_x -1)**2,weights=n_true_x))/np.sqrt(len(ratio_x)))
    print("RMS Y: ", np.sqrt(np.average((ratio_y - 1)**2,weights=n_true_y))/np.sqrt(len(ratio_y)))
    print("RMS Time: ",np.sqrt(np.average((ratio_t - 1)**2,weights=n_true_t)) / np.sqrt(len(ratio_t)))
    print(" ")
        


def main(config,args):
    print("Making plots for generations.")
    print("The folder used: ",str(config['Inference']['fixed_point_dir']))
    print(" ")

    file_folder = os.path.join("Generations",config['Inference']['fixed_point_dir'])
    file_list = os.listdir(file_folder)
    outpath = os.path.join(file_folder,"Plots")
    os.makedirs(outpath,exist_ok=True)

    plot_files = os.listdir(outpath)

    for file in file_list:
        if ".pkl" in file:
            if "Kaon" in file:
                label = 'Kaon'
            elif "Pion" in file:
                label = 'Pion'
                
            match = re.search(r'theta_(\d+\.\d+)', file)
            if match:
                theta_value = float(match.group(1))

                already_processed = False
                for out_file in plot_files:
                    out_match = re.search(r'theta_(\d+\.\d+)', out_file)
                    if out_match and float(out_match.group(1)) == theta_value:
                        already_processed = True
                        break
                
                if already_processed:
                    print(f"File with theta {theta_value} already exists in {outpath}. Skipping {file}.")
                    continue

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
    print(" ")

    # pdf_output = os.path.join(outpath,"Combined_FastSim_Plots_Radius.pdf")
    # combine_images_to_pdf(os.path.join(outpath,"RadiusPlots"),pdf_output,images_per_page=(3,2),figure_size=(8,4))
    
    print("Making ratio plots at ",args.momentum," integrated over theta for Pions.")
    make_ratios(path_=file_folder,label="Pion",momentum=args.momentum,outpath=os.path.join(outpath,"Ratios_Pion.pdf"))

    print("Making ratio plots at ",args.momentum," integrated over theta for Kaons.")
    make_ratios(path_=file_folder,label="Kaon",momentum=args.momentum,outpath=os.path.join(outpath,"Ratios_Kaon.pdf"))
    


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Plotting.')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-p', '--momentum', default=6.0,type=float,help='Particle Momentum.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args)