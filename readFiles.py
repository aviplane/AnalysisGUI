# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:47:42 2016

@author: Stanford University
"""
import numpy as np
from PIL import Image as IM
import struct
from os import listdir,path
from re import compile
import h5py
import csv
from astropy.io import fits


#filepath includes path and filename  i.e. "C:\\Users\\admin\\Desktop\\gui\\testpics\\beam.npy"



def open_all_HDF5_in_dir(directory_name):
    list_of_files=listdir(directory_name)
    re_sis=compile('\.h5')
    file_path_array=[]
    file_name_array=[]
    for item in list_of_files:
        if re_sis.search(item,1):
            file_path_array.append(path.join(directory_name,item))
            file_name_array.append(item)
    return file_path_array, file_name_array

def get_all_fitsfiles(directory_name):
    list_of_files=listdir(directory_name)
    re_sis=compile('\.fits')
    file_path_array=[]
    file_name_array=[]
    for item in list_of_files:
        if re_sis.search(item,1):
            file_path_array.append(path.join(directory_name,item))
            file_name_array.append(item)
    return file_path_array, file_name_array

#
def read_fits(filename):
    fitsdat=fits.open(filename)
    image=fitsdat[0].data
    return image[0]



def getdata(path, dataname):
#path=complete path including filename.h5, dataname=variable name where data is stored, like "MOT3D_Fluorescence" or "flat"
    def find_foo(name):
        if dataname in name:
            return name
    with h5py.File(path,'r') as hf:
        loc=hf.visit(find_foo)
        dat=hf.get(loc)
        return np.array(dat)

def getexactdata(path, dataname):
#path=complete path including filename.h5, dataname=variable name where data is stored, like "MOT3D_Fluorescence" or "flat"
    def find_foo(name):
        if dataname == name[-len(dataname):]:
            return name
    with h5py.File(path,'r') as hf:
        loc=hf.visit(find_foo)
        dat=np.array(hf.get(loc))
        return dat

def getTrace(path):
    with h5py.File(path,'r') as hf:
        arr = None
        gp1 = hf.get('data/traces')
        arr = np.array(gp1.get('MOT3D_Fluorescence'))
        return arr

def getIxonImage(path):
    with h5py.File(path,'r') as hf:
        try:
            img = hf.get('images/ixon/ixonatoms').value
        except AttributeError:
            img = hf.get('images/top/IxonAtoms/IxonAtoms').value
        return img

def getManta145Image(path):
    with h5py.File(path,'r') as hf:
    #        img = hf.get('images/top/IxonAtoms/IxonAtoms').value
        img = hf.get('images/manta145/manta145atoms').value
        return img

def getImage(path):
    with h5py.File(path,'r') as hf:
        flatarray = None
        fluoroarray=None
        gp1 = hf.get('images/top/AtomCloud')
        flatarray = np.array(gp1.get('flat'), dtype='int32')
        fluoroarray = np.array(gp1.get('fluoro'), dtype='int32')
        image=fluoroarray-flatarray
        image=posify(image) #should we cap the min val to zero?
        return image

def getxval(path, global_group, global_name):
#path to HDF5 including filename.h5, global_group specifies the global variable we are iterating with, e.g., 'MOT', global_name specifies the global variable we are iterating with, e.g., 'MOT_LoadTime'
    with h5py.File(path,'r') as hf:
        globalgroup=hf.get('globals/'+global_group)
        xval=globalgroup.attrs[global_name]
        return xval

def getxval2(path, global_name):
#path to HDF5 including filename.h5, global_name specifies the global variable we are iterating with, e.g., 'MOT_LoadTime'
    with h5py.File(path,'r') as hf:
        globalgroup=hf.get('globals')
        xval=globalgroup.attrs[global_name]
        return xval

def getxvals(path, global_name_list):
    with h5py.File(path, 'r') as hf:
        globalgroup=hf.get('globals')
        xvals = []
        for i in global_name_list:
            try:
                xvals.append(str(globalgroup.attrs[i]))
            except:
                print("%s not found." % i)
        xvals = np.array(xvals)
        return xvals

def getCavityParams(path):
    with h5py.File(path, 'r') as hf:
        group = hf.get("CavityParams/cavityParams").value
    return group

def getRunNumber(path):
    with h5py.File(path, 'r') as hf:
        n_runs = hf.attrs['n_runs']
        run_number =  hf.attrs['run number']
    return run_number, n_runs
#def open_all_type_in_dir(directory_name, filetype):
#    #example filetype: '\.sis'
#    list_of_files=listdir(directory_name)
#    re_sis=compile(filetype)
#    file_path_array=[]
#    file_name_array=[]
#    for item in list_of_files:
#        if re_sis.search(item,1):
#            file_path_array.append(path.join(directory_name,item))
#            file_name_array.append(item)
#    return file_path_array, file_name_array
#
#
#def open_all_sis_in_dir(directory_name):
#    list_of_files=listdir(directory_name)
#    re_sis=compile('\.sis')
#    file_path_array=[]
#    file_name_array=[]
#    for item in list_of_files:
#        if re_sis.search(item,1):
#            file_path_array.append(path.join(directory_name,item))
#            file_name_array.append(item)
#    return file_path_array, file_name_array
#
#
#def read_numpy_file(filepath):
#    file_handle=open(filepath,'rb')
#    image=np.load(file_handle)
#
#    return image
#
#
#def read_all_numpy(fparray, log=0):
#    numpyims=[]
#    for item in fparray:
#        im=read_numpy_file(item)
#        if log==1:
#            pic=-np.log(np.double(im))
#        else:
#            pic=im
#        numpyims.append(pic)
#    return numpyims
#
#
#def read_tif_file(filepath):
#    tifimage=IM.open(filepath)
#    tifarray=np.array(tifimage)
#
#    return tifarray
#
#def read_sis_file(filepath, log=0): #opens sis file and converts it to log 16 bit
#    image_file=open(filepath,'rb')
#    #check the image dimensions
#    image_file.seek(4)
#    (d1,)=struct.unpack('H', image_file.read(2)) #x
#    image_file.seek(8)
#    (d2,)=struct.unpack('H', image_file.read(2)) #y
#    #imagearray=np.ndarray([d1,d2],'H') #was d1, d2
#    image_file.seek(256) #seek past header info to read data
##
#    #print 'd1 x: ', d1
#    #print 'd2 y: ', d2
#
#    raw_data=image_file.read()
#    imagearray=np.fromstring(raw_data,dtype='H').reshape(d1,d2)
#    imagearray=imagearray.transpose()
#    imagearray=posify(imagearray)
#    if log==1:
#
#        scaledlogpic=-np.log(np.double(imagearray+(imagearray==0))/(2*2**14))
#        #logpic=-np.log(np.double(imagearray)/2**16)
#        #scaledlogpic=logpic*10000
#        #rows, cols=np.where(scaledlogpic>50000)
#        #scaledlogpic[rows, cols]=1
#        #scaledlogpic=scaledlogpic.astype(np.uint16)
#        #scaledlogpic=scaledlogpic<<3
#        #norm=np.double(scaledlogpic)/scaledlogpic.flatten().max()
#        #image=(65535*norm).astype(np.uint16)
#        #image=np.rot90(image,2)
#        #image=image>>1
#
#        imagearray=scaledlogpic
#        image_file.close()
#    return imagearray
#
#def read_all_sis(arrayofpaths, log=0): #takes in array of sis files and converts them to 16 bit log numpy images
#    imagearray=[]
#    for item in range(len(arrayofpaths)):
#        if log==1:
#            numpyimage=read_sis_file(arrayofpaths[item], log=1)
#        else:
#            numpyimage=read_sis_file(arrayofpaths[item], log=0)
#        imagearray.append(numpyimage)
#    return imagearray
#
#
#
#
#def rescale(arrayofims): #takes an array of images and converts them to uint8
#    scaledarray=[]
#    for item in range(len(arrayofims)):
#        norm=np.double(arrayofims[item])/arrayofims[item].flatten().max()
#        image=(255*norm).astype(np.uint8)
#        image=np.rot90(image,0)
#        scaledarray.append(image)
#    #imshow(image)
#    return scaledarray
#
def posify(array):
        rows, cols=np.where(array<=0)
        array[rows, cols]=1
        return array
#
def posify2(array):
        rows, cols=np.where(array<=0)
        array[rows, cols]=0
        return array
#
#def posify2_all(array):
#    posified=[]
#    for item in array:
#        posdata=posify2(item)
#        posified.append(posdata)
#    return posified
#
#def posify_all(array):
#    posified=[]
#    for item in array:
#        posdata=posify(item)
#        posified.append(posdata)
#    return posified


#def getxarray(filename, stringstart, stringend):
#    starti=filename.find(stringstart)+len(stringstart)
#    endi=filename.find(stringend)
#
#    return float(filename[starti:endi])

#def getxarray(filename, stringstart):
#    starti=filename.find(stringstart)+len(stringstart)
#    for n in range(starti+1, len(filename)+1):
#        try:
#            x = float(filename[starti:n])
#        except:
#            break
#
#    return x
