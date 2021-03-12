# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 22:15:21 2020

@author: chujie zhang
"""

import numpy as np
import os,shutil
import torch
import pickle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


'''
all_names = []
for f in fnames:
    loc1 = f.find('_e')
    loc2=f.find('_s')

    if f[(loc1 + 1):loc2] == "error_mode_0":  
        actions.append("error_mode_0")
        all_names.append(f)
    if f[(loc1 + 1):loc2] == "error_mode_7":  
        actions.append("error_mode_7")  
        all_names.append(f)
    if f[(loc1 + 1):loc2] == "error_mode_8":  
        actions.append("error_mode_8") 
        all_names.append(f)
    if f[(loc1 + 1):loc2] == "error_mode_9":  
        actions.append("error_mode_9") 
        all_names.append(f)
    frames=os.listdir(data_path+f)  
    n=0
    for i in frames:
        n+=1
    print(f)
    print(n)
    #for i in frames:
        
'''

action_name_path = './two_classes.pkl'
action_names=['error','no_error']
#action_names=['error_mode_0','error_mode_9','error_mode_7','error_mode_8']
with open (action_name_path,'wb') as fo:
    pickle.dump(action_names,fo)
    
with open(action_name_path,'rb') as f:
    action_names=pickle.load(f)
print(action_names)

datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
#path='D:/UCF101/ucf101_jpegs_256/CalotTriangleDissection/'
#path='D:/UCF101/ucf101_jpegs_256/ClippingCutting/'
#path='D:/UCF101/ucf101_jpegs_256/GallbladderDissection/'
#fnames = os.listdir(path)
'''
#-------------data augmentation-----------------------#
i=1
for f in fnames:
    loc1 = f.find('_e')
    loc2=f.find('_s')
    video_name=f[0:loc1]
    error_name=f[(loc1 + 1):loc2]
    newfilename=video_name+'_'+error_name+'_'+'new_DA_'+str(i)
    folder = os.path.exists(path+newfilename)
    if not folder:
        os.makedirs(path+newfilename)
        print('build new folder successfully')
    i+=1
    new_path=os.listdir(path+f)
    for pig in new_path:
        loc3=pig.find('.p')
        img = load_img(path+f+'/'+pig)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        j = 0
        for batch in datagen.flow(x,
                                  batch_size=1,
                                  save_to_dir=path+newfilename,#生成后的图像保存路径
                                  save_prefix=pig[0:loc3],
                                  save_format='png'):
            j += 1
            if j > 0: #这个20指出要扩增多少个数据
                break  # otherwise the generator would loop indefinitely
'''     
           
'''
#---------copy file and make sure each sample has at least 24 frames--------#
fnames = os.listdir(path)
for f in fnames:
    new_path=os.listdir(path+f)
    number=0
    for pig in new_path:
        number+=1
    if number<24:
        print(f+':'+str(number))
        os.chdir(path+f)
        for i in range(number+1,25):
            shutil.copyfile('frame'+str(number)+'.png','frame'+str(i)+'.png')
''' 
'''
for f in fnames: 

    if f.find('new') != -1:
        print('true',f)
        new_path=os.listdir(path+f)
        os.chdir(path+f)
        for old_name in new_path:
            loc=old_name.find('_0')
            new_name=old_name[0:loc]
            #print('old_name',old_name)
            #print('new_name',new_name)
            os.rename(old_name,new_name+'.png')
        
    else:
        print('false',f)
'''

'''
#-----------------------transfer files---------------#
ppath='F:/ucl/Cholec80_OCHRA_err_annotations_GH/sophia/'
newfile='D:/UCF101/ucf101_jpegs_256_test/'
for video_id in ['video65','video74']:
#for video_id in ['video04','video20','video31','video33','video42','video55','video64']:
    for phase in ['CalotTriangleDissection','ClippingCutting','GallbladderDissection']: 
    #for phase in ['CalotTriangleDissection']: 
        for error_mode in ['error_mode_0','error_mode_2','error_mode_6','error_mode_7','error_mode_8','error_mode_9','error_mode_10']:
        #for error_mode in ['error_mode_0']: 
            new_path=ppath+video_id+'/'+phase+'/'+error_mode
            fnames=os.listdir(new_path)
            #os.chdir(new_path)
            for f in fnames:
                loc1=f.find('_s')
                first=f[0:loc1]
                final=f[loc1:]
                new_name=first+'_'+phase+'_'+error_mode+final
                #os.rename(f,new_name)
                newfile2=newfile+phase
                shutil.move(new_path+'/'+f,newfile2)
                
'''     
'''       
#------change name to frame(i).png-------#
newfile='D:/UCF101/ucf101_jpegs_256_test/'       
for phase in ['CalotTriangleDissection','ClippingCutting','GallbladderDissection']: 
        new_path=newfile+phase
        fnames=os.listdir(new_path)
        for f in fnames:
            new_path2=new_path+'/'+f
            fnames2=os.listdir(new_path2)
            num=0
            for f2 in fnames2:
                num+=1
            print(f,num)
            number=0
            os.chdir(new_path2)
            for f2 in fnames2:
                number+=1
                os.rename(f2,'frame'+str(number)+'.png')
'''      
'''          
path2='D:/UCF101/ucf101_jpegs_256/CalotTriangleDissection/video03_CalotTriangleDissection_error_mode_7_seq1'   
fnames=os.listdir(path2)
num=0  
for f in fnames:
    num+=1
os.chdir(path2)

for i in range(num+1,25):
    print(i)
    shutil.copyfile('frame'+str(num)+'.png','frame'+str(i)+'.png')
       
'''
'''
print(torch.__version__)

print(torch.version.cuda)
print(torch.backends.cudnn.version())
test='C:/Users/chujie zhang/Desktop/test/'
fnames = os.listdir(test)
for f in fnames:
    new_path2=test+f
    fnames2=os.listdir(new_path2)
    num=0
    for f2 in fnames2:
        num+=1
    print(f,num)
    sequence=1
    os.chdir(test)
    while num>24:
        new_file_name=f+'_subset'+str(sequence)
        os.mkdir(new_file_name)
        
        for f3 in fnames2:
            repeat=0
            shutil.move(new_path2+'/'+f3,test+new_file_name)
            repeat+=1
        
            
'''