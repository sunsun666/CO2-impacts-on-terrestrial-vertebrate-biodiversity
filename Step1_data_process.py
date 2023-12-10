###############          dowmload the cmip data       #################
#https://github.com/darothen/cmip5_download/
import netCDF4 as nc
import scipy.io as scio
import pickle
from CEDA_download import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

group_models = [
    ("IPSL", "IPSL-CM5A-LR"),
    ("MOHC", "HadGEM2-ES"),
    ("NOAA-GFDL", "GFDL-ESM2M"),
]
experiments = ["1pctCO2","esmControl", "esmFdbk2","esmFixClim2","historical"]
freqs = ["mon", ]
realms=['atmos']
cmor_tables=["Amon"]
ensembles = ["r1i1p1",]
variables=["evspsbl","hurs","pr","rsds","sfcWind","tas","tasmax","tasmin",]
SAVE_PATH="F:/CO2_SENERIO/{experiment}/{model}"
datasets= get_datasets(group_models,experiments,freqs,realms,cmor_tables,ensembles,variables)
download_batch(datasets,SAVE_PATH,username='sunchuanlian001',password='54chuanlian',overwrite= False)


freqs = ["mon", ]
realms=['land']
cmor_tables=["Lmon"]
ensembles = ["r1i1p1",]
variables=["gpp","lai","mrso","treeFrac"]
SAVE_PATH="H:/CO2_SENERIO/{experiment}/{model}"
datasets= get_datasets(group_models,experiments,freqs,realms,cmor_tables,ensembles,variables)
download_batch(datasets,SAVE_PATH,username='sunchuanlian001',password='54chuanlian',overwrite= False)



#compile the cmip data

"""
files=os.listdir("H:/CO2_SENERIO/esmControl/GFDL-ESM2M/")
for file in files:
    n1=file.split("_")
    start=int(n1[-1][0:6])+184900
    end = int(n1[-1][7:13]) + 184900
    n1[-1]=str(start)+"-"+str(end)+".nc"
    newname="_".join(n1)
    os.rename(os.path.join("H:/CO2_SENERIO/esmControl/GFDL-ESM2M/",file) ,os.path.join("H:/CO2_SENERIO/esmControl/GFDL-ESM2M/",newname))
    print(n1[-1])
"""


def z(year,month):
    result=12*(year-1850)+month#需要修改
    return result

def slice_time(startyr,startmon,endyr,endmon,year_range):
    total_start_z=z(year_range['start'],1)
    total_end_z=z(year_range['end'],12)
    start_z=z(startyr,startmon)
    slice_start=max(start_z,total_start_z)-start_z+1
    end_z=z(endyr,endmon)
    slice_end=min(end_z,total_end_z)-start_z +1
    return [slice_start,slice_end,start_z,end_z]

def filterfile(experiment,model,basepath,yearrange):
    paths=basepath+"/"+ experiment+'/'+model
    datafiles=os.listdir(paths)
    for datafile in datafiles:
        pathsplit=datafile.split("_")
        pathsplit.insert(0, datafile)
        datafiles[datafiles.index(datafile)]=pathsplit

    pdData=pd.DataFrame(datafiles,columns=['pathdir','variable', 'cmor_table', 'model','experiment','ensemble','yearrange'])

    pdData['startyr']=pdData['yearrange'].str[0:4].astype(int)
    pdData['startmon']=pdData['yearrange'].str[4:6].astype(int)
    pdData['endyr']=pdData['yearrange'].str[7:11].astype(int)
    pdData['endmon']=pdData['yearrange'].str[11:13].astype(int)

    redata=pdData[pdData.endyr>=yearrange['start']]
    redata2 = redata[redata.startyr <= yearrange['end']]

    redata2['start_slice'] = pd.NA
    redata2['end_slice'] = pd.NA
    redata2['z_start']=pd.NA
    redata2['z_end']=pd.NA

    for i in range(len(redata2.pathdir)):
        result=slice_time(startyr=redata2.iloc[i,7], startmon=redata2.iloc[i,8],
                          endyr=redata2.iloc[i,9], endmon=redata2.iloc[i,10], year_range=yearrange)

        redata2.iloc[i,13]=result[2]
        redata2.iloc[i,14]=result[3]
        redata2.iloc[i,11]=result[0]
        redata2.iloc[i,12]=result[1]
    return redata2

#some files can not be proccessed using this function, because of different name rules
def compile(filenames,experiment,model,vars,basepath,yearrange):
    dataall={}
    for var in vars:
        try:
            sub=filenames[filenames.variable==var]
            test=nc.Dataset(basepath+"/"+ experiment+'/'+model+'/'+sub.iloc[0,0])[var][:]

            time=(yearrange["end"]-yearrange["start"]+1)*12

            nrow=test.shape[1]
            ncol=test.shape[2]

            da=np.full([time,nrow,ncol],np.nan)
            for d in range(len(sub.pathdir)):
                path=basepath+"/"+ experiment+'/'+model+'/'+sub.iloc[d,0]

                data1=nc.Dataset(path)[var][:]
                data1=np.array(data1)
                t1=int(sub.iloc[d,11])-1
                t2=int(sub.iloc[d,12])

                data2=data1[t1:t2,:,:]
                z_start=sub.iloc[d,-2]-1
                if z_start<0:
                    z_start=0
                z_end=sub.iloc[d,-1]
                if z_end>1812:
                    z_end=1812
                da[z_start:z_end]=data2
            dataall[var]=da
        except:
            print(var+'失败')
    return dataall



#experiments = ["esmControl", "esmFdbk2","esmFixClim2","historical"]#1pctCO2 scenario is treated separately
experiments=["1pctCO2"]
basepath='H:/CO2_SENERIO'
yearrange={'start':1984,'end':2014}

models=[
    ("IPSL", "IPSL-CM5A-LR"),
    ("MOHC", "HadGEM2-ES"),
    ("NOAA-GFDL", "GFDL-ESM2M"),
]


variables=["evspsbl","hurs","pr","rsds","sfcWind","tas","tasmax","tasmin","gpp","lai","mrso","treeFrac"]

for experiment in experiments:
    for model1 in models:
        model=model1[1]

        filedata=filterfile(experiment=experiment,model=model,basepath='H:/CO2_SENERIO',yearrange=yearrange)
        print(experiment+' '+model)

        datadict=compile(filenames=filedata,experiment=experiment,model=model,vars=variables,basepath='H:/CO2_SENERIO',
                         yearrange=yearrange)
        savedir='H:/compile'+"/"+experiment
        if os.path.exists(savedir)==False:
            os.makedirs(savedir)
        savepath=savedir+'/'+experiment+'_'+model+'.npy'
        np.save(savepath, datadict)


#calculate the baseline(1984-2014) and future(2070-2100) variables
def fanzhuan(data):
    for i in range(data.shape[0]):
        transfer=data[i,:,:]
        ftransfer = np.flipud(transfer)
        nrow=transfer.shape[0]
        ncol=transfer.shape[1]
        empty = np.full([nrow,ncol],np.nan)

        empty[:,0:int(ncol/2)]=ftransfer[:,int(ncol/2):ncol]
        empty[:,int(ncol/2):ncol]=ftransfer[:,0:int(ncol/2)]
        data[i,:,:]=empty

def monthly(data):
    nyr=int(data.shape[0]/12)
    nrow=data.shape[1]
    ncol=data.shape[2]
    redata=np.full([12,nrow,ncol],np.nan)
    for i in range(12):
        mondata=np.full([nyr,nrow,ncol],np.nan)
        for j in range(nyr):
            mondata[j]=data[j*12+i]
        mon_mean=np.nanmean(mondata,axis=0)
        redata[i]=mon_mean
    return redata

for experiment in experiments:
    for model1 in models:
        model=model1[1]
        print(experiment+"_"+model)
        data=np.load("H:/compile/"+experiment+"/"+experiment+"_"+model+".npy",allow_pickle=True).item()
        for var in variables:
            vdata=data[var]
            fanzhuan(vdata)
            var_1984_2014=vdata[19*12:33*12+12]#var_1984_2014=vdata[408:780]
            var_2070_2100=vdata[56*12:70*12+12]#var_2070_2100=vdata[1440:1812]
            var_mont_1984_2014 = monthly(var_1984_2014)
            var_mont_2070_2100 = monthly(var_2070_2100)
            np.save("H:/compile/result/"+experiment+"_"+model+"_"+var+"_"+"current"+".npy",var_mont_1984_2014 )
            np.save("H:/compile/result/" + experiment + "_" + model + "_" + var + "_" + "future" + ".npy",
                    var_mont_2070_2100)

#

###########     计算相对变化比率（气温和treeFrac用绝对）     #########
###填补缺失

from numba import jit

@jit
def fill_data(data,data_re):
    for i in range(1,nrow-1,1):
        for j in range(1,ncol-1,1):
            data_re[i,j]=np.nanmean(data[(i-1):(i+2),j-1:(j+2)])

experiments=["1pctCO2"]
basepath='H:/CO2_SENERIO'

models=[
    ("IPSL", "IPSL-CM5A-LR"),
    ("MOHC", "HadGEM2-ES"),
    ("NOAA-GFDL", "GFDL-ESM2M"),
]

land_variables=["gpp","lai","mrso"]
for experiment in experiments:
    for model1 in models:
        model = model1[1]
        fill_mrso=np.load("H:/compile/result/"+experiment+"_"+model+"_mrso_"+"current"+".npy")
        fill_gpp = np.load("H:/compile/result/" + experiment + "_" + model + "_gpp_" + "current" + ".npy")
        fill_lai = np.load("H:/compile/result/" + experiment + "_" + model + "_lai_" + "current" + ".npy")
        fill_tree=np.load("H:/compile/result/" + experiment + "_" + model + "_treeFrac_" + "current" + ".npy")
        NA = fill_mrso[0, 0, 0]
        fill_mrso[fill_mrso==NA]=np.nan
        fill_gpp[np.isnan(fill_mrso)] = np.nan
        fill_lai[np.isnan(fill_mrso)] = np.nan
        fill_tree[np.isnan(fill_mrso)] = np.nan

        nrow = fill_mrso.shape[1]
        ncol = fill_mrso.shape[2]
        for i in range(fill_mrso.shape[0]):
            for j in range(200):
                data=fill_mrso[i]
                data_re=fill_mrso[i].copy()
                fill_data(data, data_re)
                data[np.isnan(data)]=data_re[np.isnan(data)]
                fill_mrso[i]=data
        fill_mrso[:,:,0][np.isnan(fill_mrso[:,:,0])]=fill_mrso[:,:,1][np.isnan(fill_mrso[:,:,0])]
        fill_mrso[:, :, -1][np.isnan(fill_mrso[:, :, -1])] = fill_mrso[:, :, -2][np.isnan(fill_mrso[:, :, -1])]
        fill_mrso[:, 0, :][np.isnan(fill_mrso[:, 0, :])] = fill_mrso[:, 1, :][np.isnan(fill_mrso[:, 0, :])]
        fill_mrso[:, -1, :][np.isnan(fill_mrso[:, -1, :])] = fill_mrso[:, -2, :][np.isnan(fill_mrso[:, -1, :])]

        for i in range(fill_gpp.shape[0]):
            for j in range(200):
                data = fill_gpp[i]
                data_re = fill_gpp[i].copy()
                fill_data(data, data_re)
                data[np.isnan(data)] = data_re[np.isnan(data)]
                fill_gpp[i] = data
        fill_gpp[:, :, 0][np.isnan(fill_gpp[:, :, 0])] = fill_gpp[:, :, 1][np.isnan(fill_gpp[:, :, 0])]
        fill_gpp[:, :, -1][np.isnan(fill_gpp[:, :, -1])] = fill_gpp[:, :, -2][np.isnan(fill_gpp[:, :, -1])]
        fill_gpp[:, 0, :][np.isnan(fill_gpp[:, 0, :])] = fill_gpp[:, 1, :][np.isnan(fill_gpp[:, 0, :])]
        fill_gpp[:, -1, :][np.isnan(fill_gpp[:, -1, :])] = fill_gpp[:, -2, :][np.isnan(fill_gpp[:, -1, :])]

        for i in range(fill_lai.shape[0]):
            for j in range(200):
                data = fill_lai[i]
                data_re = fill_lai[i].copy()
                fill_data(data, data_re)
                data[np.isnan(data)] = data_re[np.isnan(data)]
                fill_lai[i] = data
        fill_lai[:, :, 0][np.isnan(fill_lai[:, :, 0])] = fill_lai[:, :, 1][np.isnan(fill_lai[:, :, 0])]
        fill_lai[:, :, -1][np.isnan(fill_lai[:, :, -1])] = fill_lai[:, :, -2][np.isnan(fill_lai[:, :, -1])]
        fill_lai[:, 0, :][np.isnan(fill_lai[:, 0, :])] = fill_lai[:, 1, :][np.isnan(fill_lai[:, 0, :])]
        fill_lai[:, -1, :][np.isnan(fill_lai[:, -1, :])] = fill_lai[:, -2, :][np.isnan(fill_lai[:, -1, :])]

        for i in range(fill_tree.shape[0]):
            for j in range(200):
                data = fill_tree[i]
                data_re = fill_tree[i].copy()
                fill_data(data, data_re)
                data[np.isnan(data)] = data_re[np.isnan(data)]
                fill_tree[i] = data
        fill_tree[:, :, 0][np.isnan(fill_tree[:, :, 0])] = fill_tree[:, :, 1][np.isnan(fill_tree[:, :, 0])]
        fill_tree[:, :, -1][np.isnan(fill_tree[:, :, -1])] = fill_tree[:, :, -2][np.isnan(fill_tree[:, :, -1])]
        fill_tree[:, 0, :][np.isnan(fill_tree[:, 0, :])] = fill_tree[:, 1, :][np.isnan(fill_tree[:, 0, :])]
        fill_tree[:, -1, :][np.isnan(fill_tree[:, -1, :])] = fill_tree[:, -2, :][np.isnan(fill_tree[:, -1, :])]

        np.save("H:/compile/result/"+experiment+"_"+model+"_mrso_"+"current_filledNA"+".npy",fill_mrso)
        np.save("H:/compile/result/" + experiment + "_" + model + "_gpp_" + "current_filledNA"+ ".npy",fill_gpp)
        np.save("H:/compile/result/" + experiment + "_" + model + "_lai_" + "current_filledNA" + ".npy",fill_lai)
        np.save("H:/compile/result/" + experiment + "_" + model + "_treeFrac_" + "current_filledNA" + ".npy", fill_tree)
        del data,data_re,fill_tree,fill_gpp,fill_lai,fill_mrso


for experiment in experiments:
    for model1 in models:
        model = model1[1]

        fill_mrso=np.load("H:/compile/result/"+experiment+"_"+model+"_mrso_"+"future"+".npy")
        fill_gpp = np.load("H:/compile/result/" + experiment + "_" + model + "_gpp_" + "future" + ".npy")
        fill_lai = np.load("H:/compile/result/" + experiment + "_" + model + "_lai_" + "future" + ".npy")
        fill_tree = np.load("H:/compile/result/" + experiment + "_" + model + "_treeFrac_" + "future" + ".npy")
        NA = fill_mrso[0, 0, 0]
        fill_mrso[fill_mrso==NA]=np.nan
        fill_gpp[np.isnan(fill_mrso)] = np.nan
        fill_lai[np.isnan(fill_mrso)] = np.nan
        fill_tree[np.isnan(fill_mrso)] = np.nan

        nrow = fill_mrso.shape[1]
        ncol = fill_mrso.shape[2]
        for i in range(fill_mrso.shape[0]):
            for j in range(200):
                data=fill_mrso[i]
                data_re=fill_mrso[i].copy()
                fill_data(data, data_re)
                data[np.isnan(data)]=data_re[np.isnan(data)]
                fill_mrso[i]=data
        fill_mrso[:,:,0][np.isnan(fill_mrso[:,:,0])]=fill_mrso[:,:,1][np.isnan(fill_mrso[:,:,0])]
        fill_mrso[:, :, -1][np.isnan(fill_mrso[:, :, -1])] = fill_mrso[:, :, -2][np.isnan(fill_mrso[:, :, -1])]
        fill_mrso[:, 0, :][np.isnan(fill_mrso[:, 0, :])] = fill_mrso[:, 1, :][np.isnan(fill_mrso[:, 0, :])]
        fill_mrso[:, -1, :][np.isnan(fill_mrso[:, -1, :])] = fill_mrso[:, -2, :][np.isnan(fill_mrso[:, -1, :])]

        for i in range(fill_gpp.shape[0]):
            for j in range(200):
                data = fill_gpp[i]
                data_re = fill_gpp[i].copy()
                fill_data(data, data_re)
                data[np.isnan(data)] = data_re[np.isnan(data)]
                fill_gpp[i] = data
        fill_gpp[:, :, 0][np.isnan(fill_gpp[:, :, 0])] = fill_gpp[:, :, 1][np.isnan(fill_gpp[:, :, 0])]
        fill_gpp[:, :, -1][np.isnan(fill_gpp[:, :, -1])] = fill_gpp[:, :, -2][np.isnan(fill_gpp[:, :, -1])]
        fill_gpp[:, 0, :][np.isnan(fill_gpp[:, 0, :])] = fill_gpp[:, 1, :][np.isnan(fill_gpp[:, 0, :])]
        fill_gpp[:, -1, :][np.isnan(fill_gpp[:, -1, :])] = fill_gpp[:, -2, :][np.isnan(fill_gpp[:, -1, :])]

        for i in range(fill_lai.shape[0]):
            for j in range(200):
                data = fill_lai[i]
                data_re = fill_lai[i].copy()
                fill_data(data, data_re)
                data[np.isnan(data)] = data_re[np.isnan(data)]
                fill_lai[i] = data
        fill_lai[:, :, 0][np.isnan(fill_lai[:, :, 0])] = fill_lai[:, :, 1][np.isnan(fill_lai[:, :, 0])]
        fill_lai[:, :, -1][np.isnan(fill_lai[:, :, -1])] = fill_lai[:, :, -2][np.isnan(fill_lai[:, :, -1])]
        fill_lai[:, 0, :][np.isnan(fill_lai[:, 0, :])] = fill_lai[:, 1, :][np.isnan(fill_lai[:, 0, :])]
        fill_lai[:, -1, :][np.isnan(fill_lai[:, -1, :])] = fill_lai[:, -2, :][np.isnan(fill_lai[:, -1, :])]

        for i in range(fill_tree.shape[0]):
            for j in range(200):
                data = fill_tree[i]
                data_re = fill_tree[i].copy()
                fill_data(data, data_re)
                data[np.isnan(data)] = data_re[np.isnan(data)]
                fill_tree[i] = data
        fill_tree[:, :, 0][np.isnan(fill_tree[:, :, 0])] = fill_tree[:, :, 1][np.isnan(fill_tree[:, :, 0])]
        fill_tree[:, :, -1][np.isnan(fill_tree[:, :, -1])] = fill_tree[:, :, -2][np.isnan(fill_tree[:, :, -1])]
        fill_tree[:, 0, :][np.isnan(fill_tree[:, 0, :])] = fill_tree[:, 1, :][np.isnan(fill_tree[:, 0, :])]
        fill_tree[:, -1, :][np.isnan(fill_tree[:, -1, :])] = fill_tree[:, -2, :][np.isnan(fill_tree[:, -1, :])]

        np.save("H:/compile/result/"+experiment+"_"+model+"_mrso_"+"future_filledNA"+".npy",fill_mrso)
        np.save("H:/compile/result/" + experiment + "_" + model + "_gpp_" + "future_filledNA"+ ".npy",fill_gpp)
        np.save("H:/compile/result/" + experiment + "_" + model + "_lai_" + "future_filledNA" + ".npy",fill_lai)
        np.save("H:/compile/result/" + experiment + "_" + model + "_treeFrac_" + "future_filledNA" + ".npy", fill_tree)
        del data, data_re, fill_tree, fill_gpp, fill_lai, fill_mrso


# calculate the ensemble mean

def im_bilinear_interpolation(img, out_dim):
    raw_time,raw_nlat, raw_nlon = img.shape
    out_nlat, out_nlon = out_dim[0], out_dim[1]
    if raw_nlat == out_nlat and raw_nlon == out_nlon:
        return img.copy()
    dst_img = np.zeros((raw_time, out_nlat, out_nlon), dtype=np.float)
    scale_lat, scale_lon = float(raw_nlat) / out_nlat, float(raw_nlon) / out_nlon
    for out_lat in range(out_nlat):
        for out_lon in range(out_nlon):
            #转化至WCG-84经纬度
            latout_WCG= (out_lat + 0.5) * 180/out_nlat
            lonout_WCG= (out_lon + 0.5) * 360/out_nlon

            #寻找原始图的lat和lon索引
            rawlat=int(np.floor((out_lat + 0.5)*scale_lat-0.5))
            rawlon=int(np.floor((out_lon + 0.5)*scale_lon-0.5))

            lat0_WCG=(rawlat + 0.5) * 180/raw_nlat
            lat1_WCG=(rawlat + 1.5) * 180/raw_nlat
            lon0_WCG=(rawlon + 0.5) * 360/raw_nlon
            lon1_WCG=(rawlon + 1.5) * 360/raw_nlon
            #zlat
            Rawlatmin=0.5*180/raw_nlat
            Rawlatmax=180-0.5*180/raw_nlat
            if latout_WCG>= Rawlatmax :
                rawlat0 = raw_nlat - 1
                rawlat1 = raw_nlat - 1
            elif latout_WCG<=Rawlatmin:
                rawlat0=0
                rawlat1=0
            else:
                rawlat0=rawlat
                rawlat1=rawlat+1

            #zlon
            Rawlonmin = 0.5 * 360 / raw_nlon
            Rawlonmax = 360 - 0.5 * 360 / raw_nlon
            if lonout_WCG>=Rawlonmax:
                rawlon0 = raw_nlon-1
                rawlon1 = 0
            elif lonout_WCG<=Rawlonmin:
                rawlon0 = raw_nlon - 1
                rawlon1 = 0
            else:
                rawlon0=rawlon
                rawlon1=rawlon+1

            # calculate the interpolation

            temp0 = (lon1_WCG - lonout_WCG) * img[:, rawlat0, rawlon0] + (lonout_WCG - lon0_WCG) * img[:, rawlat0,rawlon1]
            temp0=temp0/(360/raw_nlon)
            temp1 = (lon1_WCG - lonout_WCG) * img[:, rawlat1, rawlon0] + (lonout_WCG - lon0_WCG) * img[:, rawlat1,rawlon1]
            temp1=temp1/(360/raw_nlon)
            dst_img[:, out_lat, out_lon] = ((lat1_WCG - latout_WCG) * temp0 + (latout_WCG - lat0_WCG) * temp1)/(180/raw_nlat)

    return dst_img

experiments = ["1pctCO2"]
models=[
    ("IPSL", "IPSL-CM5A-LR"),
    ("MOHC", "HadGEM2-ES"),
    ("NOAA-GFDL", "GFDL-ESM2M"),
]

varatmo_rela=["evspsbl","pr","rsds","sfcWind"]
varatmo_abs=["hurs","tas","tasmax","tasmin"]
varland_rela=["gpp","lai","mrso"]
varland_abs=["treeFrac"]

for experiment in experiments:
    for var in ["evspsbl","pr","rsds","sfcWind","hurs","tas","tasmax","tasmin"]:
        for time in ["current","future"]:
            data1 = np.load("H:/compile/result/" + experiment + "_IPSL-CM5A-LR_"+var+"_" + time + ".npy")
            data2 = np.load("H:/compile/result/" + experiment + "_HadGEM2-ES_"+var+"_" + time + ".npy")
            data3 = np.load("H:/compile/result/" + experiment + "_GFDL-ESM2M_"+var+"_" + time + ".npy")
            data1 = im_bilinear_interpolation(data1, [360, 720])
            data2 = im_bilinear_interpolation(data2, [360, 720])
            data3 = im_bilinear_interpolation(data3, [360, 720])
            data = (data1 + data2 + data3) / 3
            np.save("H:/compile/result/" + experiment + "_ESM_"+var+"_" + time + ".npy", data)


for experiment in experiments:
    for var in ["gpp","lai","mrso","treeFrac"]:
        for time in ["current","future"]:
            data1 = np.load("H:/compile/result/" + experiment + "_IPSL-CM5A-LR_"+var+"_" + time + "_filledNA.npy")
            data2 = np.load("H:/compile/result/" + experiment + "_HadGEM2-ES_"+var+"_" + time + "_filledNA.npy")
            data3 = np.load("H:/compile/result/" + experiment + "_GFDL-ESM2M_"+var+"_" + time + "_filledNA.npy")
            data1 = im_bilinear_interpolation(data1, [360, 720])
            data2 = im_bilinear_interpolation(data2, [360, 720])
            data3 = im_bilinear_interpolation(data3, [360, 720])
            data = (data1 + data2 + data3) / 3
            np.save("H:/compile/result/" + experiment + "_ESM_"+var+"_" + time + ".npy", data)


########### calculate VAP and VPD using hurs
for experiment in experiments:
    for time in ["current", "future"]:
        dem=np.load('F:/E盘转移/research2/VPD/dem.npy')
        tas = np.load("H:/compile/result/"+ experiment+"_ESM_tas_" + time + ".npy")
        Pmsl = 1013.25
        Pmst = Pmsl * ((tas / (tas + 0.0065 * dem)) ** 5.625)
        Fw = 1 + 7 * 0.0001 + 3.46 * 0.000001 * Pmst
        SVP = 6.112 * Fw * np.exp((17.67 * (tas-273.16)) / ((tas-273.16) + 243.5))
        SVP=SVP/10  #单位转化为kPa
        hurs=np.load("H:/compile/result/"+ experiment+"_ESM_hurs_" + time + ".npy")
        vap = SVP * hurs/100
        VPD = SVP * (1-hurs/100)

        np.save("H:/compile/result/" + experiment + "_ESM_vap_" + time + ".npy", vap)
        np.save("H:/compile/result/" + experiment + "_ESM_vpd_" + time + ".npy", VPD)




from osgeo import gdal
def readtif(path):
    ds = gdal.Open(path)
    col = ds.RasterXSize
    row = ds.RasterYSize
    data = np.zeros([row,col])
    data[:, :] = ds.ReadAsArray(0, 0, col, row)
    return data



landsea=np.full([360,720],-9999)
landsea[0:300,:]=readtif(r"H:\relatioship\点位\landsea0.5.tif")

#### evspsbl ####     kg m-2 s-1  转为mm
for experiment in experiments:
    reference = np.load("H:/compile/result/historical_ESM_evspsbl_"+"current"+".npy")
    reference= reference *86400*30
    reference[reference<0.1]=0.1

    current = np.load("H:/compile/result/" + experiment + "_ESM_evspsbl_" + "current" + ".npy")
    current = current *86400*30
    current[current<0.1]=0.1

    future = np.load("H:/compile/result/" + experiment + "_ESM_evspsbl_" + "future" + ".npy")
    future=future*86400*30
    future[future<0.1]=0.1

    current_ratio=current/reference
    for m in range(12):
        slice = current_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        current_ratio[m][current_ratio[m] > max] = max
    future_ratio=future/current
    for m in range(12):
        slice = future_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        future_ratio[m][future_ratio[m] > max] = max

    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_ratio_evspsbl.npy",current_ratio)
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_ratio_evspsbl.npy",future_ratio)




#### hurs #### 相对湿度

for experiment in experiments:
    reference = np.load("H:/compile/result/historical_ESM_hurs_"+"current"+".npy")

    current = np.load("H:/compile/result/" + experiment + "_ESM_hurs_" + "current" + ".npy")

    future = np.load("H:/compile/result/" + experiment + "_ESM_hurs_" + "future" + ".npy")

    current_abs=current-reference
    future_abs=future-current

    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_abs_hurs.npy",current_abs)
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_abs_hurs.npy",future_abs)

#### pr ####   kg m-2 s-1  转为mm
for experiment in experiments:

    reference = np.load("H:/compile/result/historical_ESM_pr_"+"current"+".npy")
    reference= reference *86400*30
    reference[reference<0.1]=0.1

    current = np.load("H:/compile/result/" + experiment + "_ESM_pr_" + "current" + ".npy")
    current = current *86400*30
    current[current<0.1]=0.1

    future = np.load("H:/compile/result/" + experiment + "_ESM_pr_" + "future" + ".npy")
    future=future*86400*30
    future[future<0.1]=0.1

    current_ratio=current/reference
    for m in range(12):
        slice = current_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        current_ratio[m][current_ratio[m] > max] = max
    future_ratio=future/current
    for m in range(12):
        slice = future_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        future_ratio[m][future_ratio[m] > max] = max
    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_ratio_pr.npy",current_ratio)
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_ratio_pr.npy",future_ratio)


#### rsds ####  W/m2
for experiment in experiments:

    reference = np.load("H:/compile/result/historical_ESM_rsds_"+"current"+".npy")
    reference[reference<0.1]=0.1

    current = np.load("H:/compile/result/" + experiment + "_ESM_rsds_" + "current" + ".npy")
    current[current<0.1]=0.1

    future = np.load("H:/compile/result/" + experiment + "_ESM_rsds_" + "future" + ".npy")
    future[future<0.1]=0.1

    current_ratio=current/reference
    for m in range(12):
        slice = current_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        current_ratio[m][current_ratio[m] > max] = max
    future_ratio=future/current
    for m in range(12):
        slice = future_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        future_ratio[m][future_ratio[m] > max] = max
    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_ratio_rsds.npy",current_ratio)
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_ratio_rsds.npy",future_ratio)

#### sfcWind #### m/s
for experiment in experiments:

    reference = np.load("H:/compile/result/historical_ESM_sfcWind_"+"current"+".npy")
    reference[reference<0.001]=0.001

    current = np.load("H:/compile/result/" + experiment + "_ESM_sfcWind_" + "current" + ".npy")
    current[current<0.001]=0.001

    future = np.load("H:/compile/result/" + experiment + "_ESM_sfcWind_" + "future" + ".npy")
    future[future<0.001]=0.001

    current_ratio = current / reference
    for m in range(12):
        slice = current_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        current_ratio[m][current_ratio[m] > max] = max
    future_ratio = future / current
    for m in range(12):
        slice = future_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        future_ratio[m][future_ratio[m] > max] = max

    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_ratio_sfcWind.npy",current_ratio)
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_ratio_sfcWind.npy",future_ratio)


#### tas #### K
for experiment in experiments:

    reference = np.load("H:/compile/result/historical_ESM_tas_"+"current"+".npy")

    current = np.load("H:/compile/result/" + experiment + "_ESM_tas_" + "current" + ".npy")

    future = np.load("H:/compile/result/" + experiment + "_ESM_tas_" + "future" + ".npy")

    current_abs=current-reference
    future_abs=future-current

    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_abs_tas.npy",current_abs)
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_abs_tas.npy",future_abs)

#### tasmax #### K
for experiment in experiments:
    reference = np.load("H:/compile/result/historical_ESM_tasmax_"+"current"+".npy")

    current = np.load("H:/compile/result/" + experiment + "_ESM_tasmax_" + "current" + ".npy")

    future = np.load("H:/compile/result/" + experiment + "_ESM_tasmax_" + "future" + ".npy")

    current_abs=current-reference
    future_abs=future-current

    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_abs_tasmax.npy",current_abs)
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_abs_tasmax.npy",future_abs)

#### tasmin #### K
for experiment in experiments:
        reference = np.load("H:/compile/result/historical_ESM_tasmin_"+"current"+".npy")

        current = np.load("H:/compile/result/" + experiment + "_ESM_tasmin_" + "current" + ".npy")

        future = np.load("H:/compile/result/" + experiment + "_ESM_tasmin_" + "future" + ".npy")

        current_abs=current-reference
        future_abs=future-current

        np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_abs_tasmin.npy",current_abs)
        np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_abs_tasmin.npy",future_abs)



#### gpp #### g m-2 mon-1
for experiment in experiments:
        reference = np.load("H:/compile/result/historical_ESM_gpp_"+"current"+".npy")
        reference = reference * 86400 * 30*1000
        reference[reference<0.1]=0.1

        current = np.load("H:/compile/result/" + experiment + "_ESM_gpp_" + "current" + ".npy")
        current = current * 86400 * 30 * 1000
        current[current<0.1]=0.1

        future = np.load("H:/compile/result/" + experiment + "_ESM_gpp_" + "future" + ".npy")
        future = future * 86400 * 30 * 1000
        future[future<0.1]=0.1

        current_ratio = current / reference
        for m in range(12):
            slice = current_ratio[m].copy()
            slice[landsea == -9999] = np.nan
            max = np.nanquantile(slice, 0.98)
            current_ratio[m][current_ratio[m] > max] = max
        future_ratio = future / current
        for m in range(12):
            slice = future_ratio[m].copy()
            slice[landsea == -9999] = np.nan
            max = np.nanquantile(slice, 0.98)
            future_ratio[m][future_ratio[m] > max] = max

        np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_ratio_gpp.npy",current_ratio)
        np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_ratio_gpp.npy",future_ratio)


#### lai ####
for experiment in experiments:
    reference = np.load("H:/compile/result/historical_ESM_lai_"+"current"+".npy")
    reference[reference<0.01]=0.01

    current = np.load("H:/compile/result/" + experiment + "_ESM_lai_" + "current" + ".npy")
    current[current<0.01]=0.01

    future = np.load("H:/compile/result/" + experiment + "_ESM_lai_" + "future" + ".npy")
    future[future<0.01]=0.01

    current_ratio = current / reference
    for m in range(12):
        slice = current_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        current_ratio[m][current_ratio[m] > max] = max
    future_ratio = future / current
    for m in range(12):
        slice = future_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        future_ratio[m][future_ratio[m] > max] = max

    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_ratio_lai.npy",current_ratio)
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_ratio_lai.npy",future_ratio)

#### mrso #### kg/m2
for experiment in experiments:

    reference = np.load("H:/compile/result/historical_ESM_mrso_"+"current"+".npy")
    reference[reference<1]=1

    current = np.load("H:/compile/result/" + experiment + "_ESM_mrso_" + "current" + ".npy")
    current[current<1]=1

    future = np.load("H:/compile/result/" + experiment + "_ESM_mrso_" + "future" + ".npy")
    future[future<1]=1

    current_ratio = current / reference
    for m in range(12):
        slice = current_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        current_ratio[m][current_ratio[m] > max] = max
    future_ratio = future / current
    for m in range(12):
        slice = future_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        future_ratio[m][future_ratio[m] > max] = max

    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_ratio_mrso.npy",current_ratio)
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_ratio_mrso.npy",future_ratio)

#### treeFrac #### %
for experiment in experiments:
    reference = np.load("H:/compile/result/historical_ESM_treeFrac_"+"current"+".npy")
    reference = np.nanmean(reference,axis=0)
    current = np.load("H:/compile/result/" + experiment + "_ESM_treeFrac_" + "current" + ".npy")
    current = np.nanmean(current,axis=0)
    future = np.load("H:/compile/result/" + experiment + "_ESM_treeFrac_" + "future" + ".npy")
    future = np.nanmean(future,axis=0)
    current_abs=current-reference
    future_abs=future-current

    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_abs_treeFrac_Mean.npy",current_abs)
    print("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_abs_treeFrac_Mean.npy")
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_abs_treeFrac_Mean.npy",future_abs)
    print("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_abs_treeFrac_Mean.npy")




#vap
for experiment in experiments:
    reference = np.load("H:/compile/result/historical_ESM_vap_"+"current"+".npy")
    reference[reference < 0.01] = 0.01
    current = np.load("H:/compile/result/" + experiment + "_ESM_vap_" + "current" + ".npy")
    current[current < 0.01] = 0.01
    future = np.load("H:/compile/result/" + experiment + "_ESM_vap_" + "future" + ".npy")
    future[future < 0.01] = 0.01

    current_ratio = current / reference
    for m in range(12):
        slice = current_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        current_ratio[m][current_ratio[m] > max] = max
    future_ratio = future / current
    for m in range(12):
        slice = future_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        future_ratio[m][future_ratio[m] > max] = max

    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_ratio_vap.npy",current_ratio )
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_ratio_vap.npy",future_ratio)


#VPD
for experiment in experiments:
    reference = np.load("H:/compile/result/historical_ESM_vpd_"+"current"+".npy")
    reference[reference < 0.01] = 0.01
    current = np.load("H:/compile/result/" + experiment + "_ESM_vpd_" + "current" + ".npy")
    current[current < 0.01] = 0.01
    future = np.load("H:/compile/result/" + experiment + "_ESM_vpd_" + "future" + ".npy")
    future[future < 0.01] = 0.01

    current_ratio = current / reference
    for m in range(12):
        slice = current_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        current_ratio[m][current_ratio[m] > max] = max
    future_ratio = future / current
    for m in range(12):
        slice = future_ratio[m].copy()
        slice[landsea == -9999] = np.nan
        max = np.nanquantile(slice, 0.98)
        future_ratio[m][future_ratio[m] > max] = max

    np.save("H:/relatioship/点位/相对变化/"+experiment+"_ESM_current_ratio_vpd.npy",current_ratio )
    np.save("H:/relatioship/点位/相对变化/" + experiment + "_ESM_future_ratio_vpd.npy",future_ratio)
