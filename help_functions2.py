import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os

def draw_bar(labels,quants,name):
    width = 0.4
    ind = np.linspace(0.5,9.5,7)
    # make a square figure
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    # Bar Plot
    ax.bar(ind-width/2,quants,width,color='green')
    # Set the ticks on x-axis
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    # labels
    ax.set_xlabel('error mode')
    ax.set_ylabel('total no. of segments')
    # title
    ax.set_title('Total no. of segments for each mode', bbox={'facecolor':'0.8', 'pad':5})
    plt.grid(True)
    plt.show()
    plt.savefig(name)
    plt.close()

#the overall 10 videos dataset
mode_0=37+38+22+30+22+36+26+26+32+22
mode_2=1+0+0+3+0+1
mode_6=3+4
mode_7=24+4+7+12+6+5+2+19+5+7
mode_8=4+1+1+2+1+1+1+2
mode_9=18+3+6+15+10+11+5+38+7+5
mode_10=3+2+1+1+12
print(mode_0)
labels = ['error_mode_0','error_mode_2','error_mode_6','error_mode_7','error_mode_8','error_mode_9','error_mode_10']
quants = [mode_0,mode_2,mode_6,mode_7,mode_8,mode_9,mode_10]
#draw_bar(labels,quants,"bar_total.jpg")

#for calot triangle dissection
data_path = 'D:/UCF101/ucf101_jpegs_256/ClippingCutting/'    # define UCF-101 RGB data path
test_data_path = 'D:/UCF101/ucf101_jpegs_256_test/ClippingCutting/'
actions = []
fnames = os.listdir(data_path)
all_names = []
mode_0 = []
mode_2 = []
mode_6 = []
mode_7 = []
mode_8 = []
mode_9 = []
mode_10 = []
fnames_test = os.listdir(test_data_path)
for f in fnames:
    loc1 = f.find('_e')
    loc2=f.find('_s')
    loc3=f.find('_n')
           
    if f[(loc1 + 1):loc2] == "error_mode_0":  
        mode_0.append(f) 
    if f[(loc1 + 1):loc2] == "error_mode_7":  
        mode_7.append(f) 
    if f[(loc1 + 1):loc2] == "error_mode_8":  
        mode_8.append(f) 
    if f[(loc1 + 1):loc2] == "error_mode_9":  
        mode_9.append(f)   
    if f[(loc1 + 1):loc2] == "error_mode_2":  
        mode_2.append(f) 
    if f[(loc1 + 1):loc2] == "error_mode_6":  
        mode_6.append(f) 
    if f[(loc1 + 1):loc2] == "error_mode_10":  
        mode_10.append(f) 

    if f[(loc1 + 1):loc3] == "error_mode_0":  
        mode_0.append(f) 
    if f[(loc1 + 1):loc3] == "error_mode_7":  
        mode_7.append(f) 
    if f[(loc1 + 1):loc3] == "error_mode_8":  
        mode_8.append(f) 
    if f[(loc1 + 1):loc3] == "error_mode_9":  
        mode_9.append(f)   
    if f[(loc1 + 1):loc3] == "error_mode_2":  
        mode_2.append(f) 
    if f[(loc1 + 1):loc3] == "error_mode_6":  
        mode_6.append(f) 
    if f[(loc1 + 1):loc3] == "error_mode_10":  
        mode_10.append(f)   
for f in fnames_test:
    loc1 = f.find('_e')
    loc2=f.find('_s')
    loc3=f.find('_n')
           
    if f[(loc1 + 1):loc2] == "error_mode_0":  
        mode_0.append(f) 
    if f[(loc1 + 1):loc2] == "error_mode_7":  
        mode_7.append(f) 
    if f[(loc1 + 1):loc2] == "error_mode_8":  
        mode_8.append(f) 
    if f[(loc1 + 1):loc2] == "error_mode_9":  
        mode_9.append(f)   
    if f[(loc1 + 1):loc2] == "error_mode_2":  
        mode_2.append(f) 
    if f[(loc1 + 1):loc2] == "error_mode_6":  
        mode_6.append(f) 
    if f[(loc1 + 1):loc2] == "error_mode_10":  
        mode_10.append(f) 

    if f[(loc1 + 1):loc3] == "error_mode_0":  
        mode_0.append(f) 
    if f[(loc1 + 1):loc3] == "error_mode_7":  
        mode_7.append(f) 
    if f[(loc1 + 1):loc3] == "error_mode_8":  
        mode_8.append(f) 
    if f[(loc1 + 1):loc3] == "error_mode_9":  
        mode_9.append(f)   
    if f[(loc1 + 1):loc3] == "error_mode_2":  
        mode_2.append(f) 
    if f[(loc1 + 1):loc3] == "error_mode_6":  
        mode_6.append(f) 
    if f[(loc1 + 1):loc3] == "error_mode_10":  
        mode_10.append(f)  

labels = ['error_mode_0','error_mode_2','error_mode_6','error_mode_7','error_mode_8','error_mode_9','error_mode_10']
quants = [len(mode_0),len(mode_2),len(mode_6),len(mode_7),len(mode_8),len(mode_9),len(mode_10)]
draw_bar(labels,quants,"bar_calot.jpg")



 