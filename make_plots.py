import re
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import argparse
import json

import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

from matplotlib.font_manager import FontProperties


####################################### hpDIRC Discretization #################################
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

t_bins = np.arange(0.0,157.0,0.5)
####################################### hpDIRC Discretization #################################


def convert_indices(pmtID,pixelID): 
    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    
    return x,y

def make_plots_fastsim(file_path,label,momentum,theta,outpath,filename,log_norm=True, time_cutoff=1.5e-4):
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

    gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 0.5, 1])

    fig = plt.figure(figsize=(18, 12),dpi=300)

    ax1 = fig.add_subplot(gs[0, 0])  # Top-left
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right
    ax3 = fig.add_subplot(gs[2, :])  # Bottom image, spans both columns

    ax3.set_position([
        ax1.get_position().x0 + (ax2.get_position().x0 - ax1.get_position().x0) / 2,  # Center horizontally
        ax3.get_position().y0,  # Keep original y position
        ax1.get_position().width * 1.2,  # Keep same width as top images
        ax1.get_position().height  # Keep same height
    ])

    if log_norm:
        norm = LogNorm()
    else:
        norm = None
    
    title_font = FontProperties(family="Verdana", size=24)

    ax1.hist2d(xs,ys,bins=[bins_x,bins_y],norm=norm,density=True)
    ax1.set_title(label + " Fast Simulated Hit Patterns",fontproperties=title_font)
    ax1.tick_params(axis="both", labelsize=18) 
    ax1.set_xlabel("X (mm)",fontsize=20)
    ax1.set_ylabel("Y (mm)",fontsize=20)

    ax2.hist2d(true_xs,true_ys,bins=[bins_x,bins_y],norm=norm,density=True)
    ax2.set_title(label + " Geant4 Hit Pattern",fontproperties=title_font)
    ax2.tick_params(axis="both", labelsize=18) 
    ax2.set_xlabel("X (mm)",fontsize=20)
    ax2.set_ylabel("Y (mm)",fontsize=20)

    counts, bin_edges, _ = ax3.hist(time,bins=t_bins,label = 'FastSim', density=True,histtype='step',color='r',linewidth=1.5)
    # Dynamic binning
    for i in range(len(counts)-1, -1, -1):
        if counts[i] >= time_cutoff:
            bins_max = bin_edges[i+1]
            break
    
    tcounts, tbin_edges, _ = ax3.hist(true_time,bins=t_bins,label = 'Geant4',density=True,histtype='step',color='k',linewidth=1.5)
    for i in range(len(tcounts)-1, -1, -1):
        if tcounts[i] >= time_cutoff:
            tbins_max = tbin_edges[i+1]
            break

    ax3.set_xlim(0, max(tbins_max,bins_max))
    ax3.set_xlabel("Hit Time (ns)",fontproperties=title_font)
    ax3.tick_params(axis="both", labelsize=18) 
    ax3.set_ylabel("A.U.",fontsize=20)
    # ax3.legend(["Fast Simulated","Geant4"],fontsize=20,ncol=2,frameon=True, framealpha=1)
    handles, labels = ax3.get_legend_handles_labels()
    line_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    legend = ax3.legend(line_handles, labels, fontsize=20, ncol=2, frameon=True, framealpha=1)

    ax3.set_title("Fast Simulated Time vs. Geant4 Time",fontproperties=title_font)
    
    # Setting text box for momentum and theta
    fig.canvas.draw() # updates the figure before accessing 
    bbox = legend.get_window_extent().transformed(ax3.transAxes.inverted())

    infostr = r"$|\vec{\rho}|$ ="+r" {0} GeV/c,".format(momentum) + r" $\theta = {0}^o$".format(theta)
    props = dict(boxstyle='round', edgecolor = ax3.get_legend().get_frame().get_edgecolor(), facecolor='white',alpha=1)

    inital_text = ax3.text(bbox.x1, bbox.y0-0.05, infostr, transform=ax3.transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
    fig.canvas.draw()

    text_width = inital_text.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax3.transAxes.inverted()).width
    
    x_pos = bbox.x1 - text_width - 0.05
    inital_text.set_x(x_pos)

    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
    
    save_path = os.path.join(outpath,filename[:-3]+".svg")
    plt.savefig(save_path,bbox_inches="tight")
    plt.close()


def combine_images_to_pdf(svg_folder, output_pdf, images_per_page=(2, 2), figure_size=(8, 6), row_gap = 0.05):
                        
    svg_files = [f for f in os.listdir(svg_folder) if f.lower().endswith('.svg')]
    
    svg_files.sort(key=lambda x: float(re.search(r'theta_(\d+\.\d+)', x).group(1)))
    
    total_images = len(svg_files)
    images_per_page_total = images_per_page[0] * images_per_page[1]
    num_pages = (total_images + images_per_page_total - 1) // images_per_page_total

    cell_width = figure_size[0] * inch * 0.9
    cell_height = figure_size[1] * inch * 0.7

    rows, cols = images_per_page

    page_width = cols * cell_width 
    page_height = rows * cell_height

    c = canvas.Canvas(output_pdf, pagesize=(page_width, page_height))
    
    for page in range(num_pages):
        for i in range(images_per_page_total):
            img_index = page * images_per_page_total + i
            if img_index >= total_images:
                break

            row = i // cols  
            col = i % cols

            if row == 0:
                gap_above = 0
            else:
                gap_above = row_gap * inch

            # In ReportLab the origin is bottom-left, so compute cell_y from the top:
            cell_x = col * cell_width
            cell_y = page_height - (row + 1) * cell_height - gap_above  # bottom of the cell

            svg_path = os.path.join(svg_folder, svg_files[img_index])
            drawing = svg2rlg(svg_path)

            if drawing.width > 0 and drawing.height > 0:
                scale = min(cell_width / drawing.width, cell_height / drawing.height)
            else:
                scale = 1

            drawing.scale(scale, scale)
            new_width = drawing.width * scale
            new_height = drawing.height * scale

            offset_x = (cell_width - new_width) / 2
            offset_y = (cell_height - new_height) / 2

            final_x = cell_x + offset_x
            final_y = cell_y + offset_y

            renderPDF.draw(drawing, c, final_x, final_y)
        c.showPage()
    c.save()

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
    
    #lower_limit = 1e-5  # Define your lower limit 
    #upper_limit = 5e-1   # Define an upper limit

    #for i in range(3):
    #    ax[i].set_ylim([lower_limit, upper_limit])

    # Time PDF
    n_true_t, _ = np.histogram(t_true, bins=t_bins, density=True)
    n_gen_t, _ = np.histogram(t, bins=t_bins, density=True)
    alea_t,_ = np.histogram(t,bins=t_bins,density=False)
    alea_t = np.sqrt(alea_t) / (alea_t.sum() * (t_bins[1] - t_bins[0]))
    ratio_err_t = alea_t / (n_true_t + 1e-50)
    ax[0].hist(t_true, density=True, color='k', label='Truth', bins=t_bins, histtype='step', lw=2)
    ax[0].hist(t, density=True, color='red', label='Generated', bins=t_bins, histtype='step', linestyle='dashed', lw=2)
    ax[0].set_xlabel("Hit Time (ns)", fontsize=20, labelpad=10)
    ax[0].set_ylabel("Density", fontsize=20, labelpad=10)

    # X PDF
    n_true_x, _ = np.histogram(x_true, bins=bins_x, density=True)
    n_gen_x, _ = np.histogram(x, bins=bins_x, density=True)
    alea_x,_ = np.histogram(x,bins=bins_x,density=False)
    alea_x = np.sqrt(alea_x) / (alea_x.sum() * (bins_x[1] - bins_x[0]))
    ratio_err_x = alea_x / (n_true_x + 1e-10)
    ax[1].hist(x_true, density=True, color='k', label='Truth', bins=bins_x, histtype='step', lw=2)
    ax[1].hist(x, density=True, color='red', label='Generated', bins=bins_x, histtype='step', linestyle='dashed', lw=2)
    ax[1].set_xlabel("X (mm)", fontsize=20, labelpad=10)
    ax[1].set_title(str(label)+r" - $|\vec{p}|$ ="+r" {0} GeV/c".format(momentum),fontsize=25)

    # Y PDF
    n_true_y, _ = np.histogram(y_true, bins=bins_y, density=True)
    alea_y, _ = np.histogram(y,bins=bins_y,density=False)
    alea_y = np.sqrt(alea_y) / (alea_y.sum() * (bins_y[1] - bins_y[0]))
    ratio_err_y = alea_y / (n_true_y + 1e-10)
    n_gen_y, _ = np.histogram(y, bins=bins_y, density=True)
    ax[2].hist(y_true, density=True, color='k', label='Truth', bins=bins_y, histtype='step', lw=2)
    ax[2].hist(y, density=True, color='red', label='Generated', bins=bins_y, histtype='step', linestyle='dashed', lw=2)
    ax[2].set_xlabel("Y (mm)", fontsize=20, labelpad=10)

    # Ratio plot for Time PDF
    ratio_t = n_gen_t / (n_true_t + 1e-50)
    ratio_t[np.where(n_gen_t == n_true_t)[0]] = 1.0
    ratio_t_upper = ratio_t + ratio_err_t
    ratio_t_lower = ratio_t - ratio_err_t
    #ax[3].errorbar(t_bins[:-1], ratio_t, yerr=ratio_err_t, color='red', ls='none', capsize=3, marker='.', ms=8)
    ax[3].fill_between((t_bins[:-1] + t_bins[1:]) / 2, ratio_t_lower, ratio_t_upper, color='red', alpha=0.7, step='mid')
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
    #ax[4].errorbar(bins_x[:-1], ratio_x, yerr=ratio_err_x, color='red', ls='none', capsize=3, marker='.', ms=8)
    ax[4].fill_between(bin_centers_x, ratio_x_lower, ratio_x_upper, color='red', alpha=0.7, step='mid')
    ax[4].set_ylim([0, 2])
    ax[4].set_yticks([0.5, 1, 1.5])
    ax[4].axhline(1.0, color='black', linestyle='--', lw=1)

    # Ratio plot for Y PDF
    bin_centers_y = [(bins_y[i] + bins_y[i + 1]) / 2 for i in range(len(bins_y) - 1)]
    ratio_y = n_gen_y / (n_true_y + 1e-50)
    ratio_y[np.where(n_gen_y == n_true_y)[0]] = 1.0
    ratio_y_upper = ratio_y + ratio_err_y
    ratio_y_lower = ratio_y - ratio_err_y
    #ax[5].errorbar(bins_y[:-1], ratio_y, yerr=ratio_err_y, color='red', ls='none', capsize=3, marker='.', ms=8)
    ax[5].fill_between(bin_centers_y, ratio_y_lower, ratio_y_upper, color='red', alpha=0.7, step='mid')
    ax[5].set_ylim([0, 2])
    ax[5].set_yticks([0.5, 1, 1.5])
    ax[5].axhline(1.0, color='black', linestyle='--', lw=1)

    # Format
    ax[1].legend(loc="best", fontsize=18.5, ncol=2, handlelength=2, handleheight=-.5)
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

    print("RMS X: ",np.sqrt(np.average((ratio_x -1)**2,weights=n_true_x)))
    print("RMS Y: ", np.sqrt(np.average((ratio_y - 1)**2,weights=n_true_y)))
    print("RMS Time: ",np.sqrt(np.average((ratio_t - 1)**2,weights=n_true_t)))
    print(" ")
        


def main(config,args):
    print("Making plots for generations.")
    print("The folder used: ",str(config['Inference']['fixed_point_dir']))
    print(" ")

    file_folder = os.path.join("Generations",config['Inference']['fixed_point_dir'])
    file_list = os.listdir(file_folder)
    outpath = os.path.join(file_folder,"Plots")
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
    print(" ")
    
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