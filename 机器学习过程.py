# 常用包库加载
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import *
from statsmodels.tsa import arima_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import datetime
from datetime import date, timedelta
import lightgbm as lgb
import netCDF4 as nc
from osgeo import gdal
import scipy.io as scio
from pdpbox import pdp, get_dataset, info_plots
import pickle

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
def readtif(path):
    ds = gdal.Open(path)
    col = ds.RasterXSize
    row = ds.RasterYSize
    data = np.zeros([row,col])
    data[:, :] = ds.ReadAsArray(0, 0, col, row)
    return data


from numba import jit
@jit
def fill_raster(data_real, data):
    for i in range(2,3598,1):
        for j in range(2,7198,1):
            data[i,j]=np.nanmean(data_real[(i-2):(i+3),j-2:(j+3)])


landsea=readtif("H:/基于关系的研究/点位/landsea.tif")

###############          下载数据情景数据       #################

# atomosphere variables download

from CEDA_download import *
group_models = [
    ("IPSL", "IPSL-CM5A-LR"),
    ("MOHC", "HadGEM2-ES"),
    ("NOAA-GFDL", "GFDL-ESM2M"),
]
experiments = ["esmControl", "esmFdbk2","esmFixClim2","historical","rcp45"]
freqs = ["mon", ]
realms=['atmos',]
cmor_tables=["Amon"]
ensembles = ["r1i1p1",]
variables=["evspsbl","hur","pr","rsds","sfcWind","tas","tasmax","tasmin"]
SAVE_PATH="H:/CO2_SENERIO/{experiment}/{model}"
datasets= get_datasets(group_models,experiments,freqs,realms,cmor_tables,ensembles,variables)
download_batch(datasets,SAVE_PATH,username='sunchuanlian001',password='54chuanlian',overwrite= False)

# land variables download

freqs = ["mon", ]
realms=['land',]
cmor_tables=["Lmon"]
ensembles = ["r1i1p1",]
variables=["gpp","lai","mrso","treeFrac"]
SAVE_PATH="H:/CO2_SENERIO/{experiment}/{model}"
datasets= get_datasets(group_models,experiments,freqs,realms,cmor_tables,ensembles,variables)
download_batch(datasets,SAVE_PATH,username='sunchuanlian001',password='54chuanlian',overwrite= False)

#数据拼接
###修改 esmControl 的名字
files=os.listdir("H:/CO2_SENERIO/esmControl/GFDL-ESM2M/")
for file in files:
    n1=file.split("_")
    start=int(n1[-1][0:6])+184900
    end = int(n1[-1][7:13]) + 184900
    n1[-1]=str(start)+"-"+str(end)+".nc"
    newname="_".join(n1)
    os.rename(os.path.join("H:/CO2_SENERIO/esmControl/GFDL-ESM2M/",file) ,os.path.join("H:/CO2_SENERIO/esmControl/GFDL-ESM2M/",newname))
    print(n1[-1])

###将不同年份组合

#映射为统一时间1950-01为1的时间轴
def z(year,month):
    result=12*(year-1950)+month
    return result

#标定slice
def slice_time(startyr,startmon,endyr,endmon,year_range):
    total_start_z=z(year_range['start'],1)
    total_end_z=z(year_range['end'],12)
    start_z=z(startyr,startmon)
    slice_start=max(start_z,total_start_z)-start_z+1
    end_z=z(endyr,endmon)
    slice_end=min(end_z,total_end_z)-start_z +1
    return [slice_start,slice_end,start_z,end_z]

##提取并筛选文件列表
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

experiments = ["esmControl", "esmFdbk2","esmFixClim2","historical"]
basepath='H:/CO2_SENERIO'
yearrange={'start':1950,'end':2100}

models=[
    ("IPSL", "IPSL-CM5A-LR"),
    ("MOHC", "HadGEM2-ES"),
    ("NOAA-GFDL", "GFDL-ESM2M"),
]


variables=["evspsbl","hurs","pr","rsds","sfcWind","tas","tasmax","tasmin","gpp","lai","mrso","treeFrac"]

for experiment in experiments:
    for model1 in models:
        model=model1[1]

        filedata=filterfile(experiment=experiment,model=model,basepath='H:/CO2_SENERIO',yearrange={'start':1950,'end':2100})
        print(experiment+' '+model)

        datadict=compile(filenames=filedata,experiment=experiment,model=model,vars=variables,basepath='H:/CO2_SENERIO',
                         yearrange={'start':1950,'end':2100})
        savedir='H:/compile'+"/"+experiment
        if os.path.exists(savedir)==False:
            os.makedirs(savedir)
        savepath=savedir+'/'+experiment+'_'+model+'.npy'
        np.save(savepath, datadict)

for experiment in experiments:
    for model1 in models:
        model=model1[1]
        data=np.load("H:/compile/"+experiment+"/"+experiment+"_"+model+".npy",allow_pickle=True).item()
        vars = data.keys()
        for var in vars:
            d = data[var]
            print(experiment+"/"+experiment+"_"+model+".npy"+var + " " + str(d.shape))


data=np.load("H:/compile/historical/historical_HadGEM2-ES.npy",allow_pickle=True).item()
vars=data.keys()
for var in vars:
    d=data[var]
    print(var+" "+str(d.shape))


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
            var_1984_2014=vdata[408:780]
            var_2070_2100=vdata[1440:1812]
            var_mont_1984_2014 = monthly(var_1984_2014)
            var_mont_2070_2100 = monthly(var_2070_2100)
            np.save("H:/compile/result/"+experiment+"_"+model+"_"+var+"_"+"current"+".npy",var_mont_1984_2014 )
            np.save("H:/compile/result/" + experiment + "_" + model + "_" + var + "_" + "future" + ".npy",
                    var_mont_2070_2100)

###########     计算相对变化比率（气温和treeFrac用绝对）     #########

###填补缺失

from numba import jit

@jit
def fill_data(data,data_re):
    for i in range(1,nrow-1,1):
        for j in range(1,ncol-1,1):
            data_re[i,j]=np.nanmean(data[(i-1):(i+2),j-1:(j+2)])

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

experiments = ["esmControl", "esmFdbk2","esmFixClim2","historical"]

models=[
    ("IPSL", "IPSL-CM5A-LR"),
    ("MOHC", "HadGEM2-ES"),
    ("NOAA-GFDL", "GFDL-ESM2M"),
]

varatmo_rela=["evspsbl","pr","rsds","sfcWind"]
varatmo_abs=["hurs","tas","tasmax","tasmin"]
varland_rela=["gpp","lai","mrso"]
varland_abs=["treeFrac"]

######## 组合平均值
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

########### hur改为VAP和VPD
for experiment in experiments:
    for time in ["current", "future"]:
        dem=np.load('E:/research2/VPD/dem.npy')
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



landsea=np.full([360,720],-9999)
landsea[0:300,:]=readtif("E:/AI/cover/landsea0.5.tif")





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

    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_ratio_evspsbl.npy",current_ratio)
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_ratio_evspsbl.npy",future_ratio)


#### hurs #### 相对湿度

for experiment in experiments:
    reference = np.load("H:/compile/result/historical_ESM_hurs_"+"current"+".npy")

    current = np.load("H:/compile/result/" + experiment + "_ESM_hurs_" + "current" + ".npy")

    future = np.load("H:/compile/result/" + experiment + "_ESM_hurs_" + "future" + ".npy")

    current_abs=current-reference
    future_abs=future-current

    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_abs_hurs.npy",current_abs)
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_abs_hurs.npy",future_abs)

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
    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_ratio_pr.npy",current_ratio)
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_ratio_pr.npy",future_ratio)


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
    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_ratio_rsds.npy",current_ratio)
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_ratio_rsds.npy",future_ratio)

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

    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_ratio_sfcWind.npy",current_ratio)
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_ratio_sfcWind.npy",future_ratio)


#### tas #### K
for experiment in experiments:

    reference = np.load("H:/compile/result/historical_ESM_tas_"+"current"+".npy")

    current = np.load("H:/compile/result/" + experiment + "_ESM_tas_" + "current" + ".npy")

    future = np.load("H:/compile/result/" + experiment + "_ESM_tas_" + "future" + ".npy")

    current_abs=current-reference
    future_abs=future-current

    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_abs_tas.npy",current_abs)
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_abs_tas.npy",future_abs)

#### tasmax #### K
for experiment in experiments:
    reference = np.load("H:/compile/result/historical_ESM_tasmax_"+"current"+".npy")

    current = np.load("H:/compile/result/" + experiment + "_ESM_tasmax_" + "current" + ".npy")

    future = np.load("H:/compile/result/" + experiment + "_ESM_tasmax_" + "future" + ".npy")

    current_abs=current-reference
    future_abs=future-current

    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_abs_tasmax.npy",current_abs)
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_abs_tasmax.npy",future_abs)

#### tasmin #### K
for experiment in experiments:
        reference = np.load("H:/compile/result/historical_ESM_tasmin_"+"current"+".npy")

        current = np.load("H:/compile/result/" + experiment + "_ESM_tasmin_" + "current" + ".npy")

        future = np.load("H:/compile/result/" + experiment + "_ESM_tasmin_" + "future" + ".npy")

        current_abs=current-reference
        future_abs=future-current

        np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_abs_tasmin.npy",current_abs)
        np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_abs_tasmin.npy",future_abs)

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

        np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_ratio_gpp.npy",current_ratio)
        np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_ratio_gpp.npy",future_ratio)


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

    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_ratio_lai.npy",current_ratio)
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_ratio_lai.npy",future_ratio)

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

    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_ratio_mrso.npy",current_ratio)
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_ratio_mrso.npy",future_ratio)

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

    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_abs_treeFrac_Mean.npy",current_abs)
    print("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_abs_treeFrac_Mean.npy")
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_abs_treeFrac_Mean.npy",future_abs)
    print("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_abs_treeFrac_Mean.npy")




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

    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_ratio_vap.npy",current_ratio )
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_ratio_vap.npy",future_ratio)


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

    np.save("H:/基于关系的研究/点位/相对变化/"+experiment+"_ESM_current_ratio_vpd.npy",current_ratio )
    np.save("H:/基于关系的研究/点位/相对变化/" + experiment + "_ESM_future_ratio_vpd.npy",future_ratio)








####    对每一个变量进行预测


landsea=np.full([3600,7200],-9999)
landsea[0:3000,:]=readtif("H:/CMIP5AI/landsea.tif")

experiments = ["esmControl", "esmFdbk2","esmFixClim2","historical"]
var1=[["evspsbl",0.1],["pr",0.1],["rsds",0.1],["sfcWind",0.001],["gpp",0.1],["lai",0.01],["mrso",1],["vpd",0.01],["vap",0.01]]
var2=["hur","tas","tasmax","tasmin"]

#aet
aetraw=np.load("H:/基于关系的研究/点位/原始变量/historical_aet1984_2014_monthly_005d.npy")
aetraw[aetraw<0.1]=0.1
np.save("H:/基于关系的研究/点位/原始变量/historical_aet1984_2014_monthly_005d.npy",aetraw)

for experiment in experiments:
    if experiment!="historical":
        aetC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = aetraw[m].copy()
            current_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_current_ratio_evspsbl_M"+str(month)+".tif")
            current_ratio[landsea==-9999] = np.nan
            aetC[m]=varraw*current_ratio
        aetC[aetC<0.1]=0.1
        aetF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= aetC [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_evspsbl_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            aetF[m]=varaw*future_ratio
        aetF[aetF < 0.1] = 0.1
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_aet2070_2100_monthly_005d.npy", aetF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_aet1984_2014_monthly_005d.npy", aetC)
    else:
        aetF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= aetraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_evspsbl_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            aetF[m]=varaw*future_ratio
        aetF[aetF < 0.1] = 0.1
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_aet2070_2100_monthly_005d.npy", aetF)
del aetraw,aetC,aetF


#pr
prraw=np.load("H:/基于关系的研究/点位/原始变量/historical_pr1984_2014_monthly_005d.npy")

for experiment in experiments:
    if experiment!="historical":
        prC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = prraw[m].copy()
            current_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_current_ratio_pr_M"+str(month)+".tif")
            current_ratio[landsea==-9999] = np.nan
            prC[m]=varraw*current_ratio
        prC[prC<0.1]=0.1
        prF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= prC[m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_pr_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            prF[m]=varaw*future_ratio
        prF[prF < 0.1] = 0.1
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_pr2070_2100_monthly_005d.npy", prF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_pr1984_2014_monthly_005d.npy", prC)
    else:
        prF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= prraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_pr_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            prF[m]=varaw*future_ratio
        prF[prF < 0.1] = 0.1
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_pr2070_2100_monthly_005d.npy", prF)
del prraw,prC,prF

#rsds
rsdsraw=np.load("H:/基于关系的研究/点位/原始变量/historical_rsds1984_2014_monthly_005d.npy")
rsdsraw[rsdsraw<0.1]=0.1

for experiment in experiments:
    if experiment!="historical":
        rsdsC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = rsdsraw[m].copy()
            current_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_current_ratio_rsds_M"+str(month)+".tif")
            current_ratio[landsea==-9999] = np.nan
            rsdsC[m]=varraw*current_ratio
        rsdsC[rsdsC<0.1]=0.1
        rsdsF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= rsdsC[m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_rsds_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            rsdsF[m]=varaw*future_ratio
        rsdsF[rsdsF < 0.1] = 0.1
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_rsds2070_2100_monthly_005d.npy", rsdsF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_rsds1984_2014_monthly_005d.npy", rsdsC)
    else:
        rsdsF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= rsdsraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_rsds_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            rsdsF[m]=varaw*future_ratio
        rsdsF[rsdsF < 0.1] = 0.1
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_rsds2070_2100_monthly_005d.npy", rsdsF)
del rsdsraw,rsdsC,rsdsF

#sfcWind
sfcWindraw=np.load("H:/基于关系的研究/点位/原始变量/historical_sfcWind1984_2014_monthly_005d.npy")
sfcWindraw[sfcWindraw<0.001]=0.001

for experiment in experiments:
    if experiment!="historical":
        sfcWindC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = sfcWindraw[m].copy()
            current_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_current_ratio_sfcWind_M"+str(month)+".tif")
            current_ratio[landsea==-9999] = np.nan
            sfcWindC[m]=varraw*current_ratio
        sfcWindC[sfcWindC<0.001]=0.001
        sfcWindF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= sfcWindC[m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_sfcWind_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            sfcWindF[m]=varaw*future_ratio
        sfcWindF[sfcWindF < 0.001] = 0.001
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_sfcWind2070_2100_monthly_005d.npy", sfcWindF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_sfcWind1984_2014_monthly_005d.npy", sfcWindC)
    else:
        sfcWindF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= sfcWindraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_sfcWind_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            sfcWindF[m]=varaw*future_ratio
        sfcWindF[sfcWindF < 0.001] = 0.001
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_sfcWind2070_2100_monthly_005d.npy", sfcWindF)
del sfcWindraw,sfcWindC,sfcWindF

#gpp
#gppraw=np.load("H:/基于关系的研究/点位/原始变量/GPP1984_2014_monthly.npy")
#gppraw=gppraw*30
#gppraw[gppraw<0.1]=0.1
#np.save("H:/基于关系的研究/点位/原始变量/historical_gpp1984_2014_monthly_005d.npy",gppraw)
"""
for i in range(12):
    data_mon=gppraw[i].copy()
    data_mon[np.where(np.isnan(data_mon) & (landsea!=-9999))]=0.1
    gppraw[i]=data_mon
"""
gppraw=np.load("H:/基于关系的研究/点位/原始变量/historical_gpp1984_2014_monthly_005d.npy") #检查最小值是否为0.1
for experiment in experiments:
    if experiment!="historical":
        gppC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = gppraw[m].copy()
            current_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_current_ratio_gpp_M"+str(month)+".tif")
            current_ratio[landsea==-9999] = np.nan
            gppC[m]=varraw*current_ratio
        gppC[gppC<0.1]=0.1
        gppF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= gppC[m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_gpp_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            gppF[m]=varaw*future_ratio
        gppF[gppF < 0.1] = 0.1
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_gpp2070_2100_monthly_005d.npy", gppF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_gpp1984_2014_monthly_005d.npy", gppC)
    else:
        gppF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= gppraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_gpp_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            gppF[m]=varaw*future_ratio
        gppF[gppF < 0.1] = 0.1
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_gpp2070_2100_monthly_005d.npy", gppF)
del gppraw,gppC,gppF


lairaw=np.load("H:/基于关系的研究/点位/原始变量/historical_lai1984_2014_monthly_005d.npy")
lairaw[lairaw<0.01]=0.01

for experiment in experiments:
    if experiment!="historical":
        laiC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = lairaw[m].copy()
            current_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_current_ratio_lai_M"+str(month)+".tif")
            current_ratio[landsea==-9999] = np.nan
            laiC[m]=varraw*current_ratio
        laiC[laiC<0.01]=0.01
        laiF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= laiC[m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_lai_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            laiF[m]=varaw*future_ratio
        laiF[laiF < 0.01] = 0.01
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_lai2070_2100_monthly_005d.npy", laiF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_lai1984_2014_monthly_005d.npy", laiC)
    else:
        laiF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= lairaw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_lai_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            laiF[m]=varaw*future_ratio
        laiF[laiF < 0.01] = 0.01
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_lai2070_2100_monthly_005d.npy", laiF)
del lairaw,laiC,laiF

#mrso
mrsoraw=np.load("H:/基于关系的研究/点位/原始变量/historical_mrso1984_2014_monthly_005d.npy")
mrsoraw[mrsoraw<0.01]=0.01

for experiment in experiments:
    if experiment!="historical":
        mrsoC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = mrsoraw[m].copy()
            current_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_current_ratio_mrso_M"+str(month)+".tif")
            current_ratio[landsea==-9999] = np.nan
            mrsoC[m]=varraw*current_ratio
        mrsoC[mrsoC<0.01]=0.01
        mrsoF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= mrsoC[m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_mrso_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            mrsoF[m]=varaw*future_ratio
        mrsoF[mrsoF < 0.01] = 0.01
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_mrso2070_2100_monthly_005d.npy", mrsoF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_mrso1984_2014_monthly_005d.npy", mrsoC)
    else:
        mrsoF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= mrsoraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_mrso_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            mrsoF[m]=varaw*future_ratio
        mrsoF[mrsoF < 0.01] = 0.01
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_mrso2070_2100_monthly_005d.npy", mrsoF)
del mrsoraw,mrsoC,mrsoF

#vap
vapraw=np.load("H:/基于关系的研究/点位/原始变量/historical_vap1984_2014_monthly_005d.npy")
vapraw[vapraw<0.01]=0.01

for experiment in experiments:
    if experiment!="historical":
        vapC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = vapraw[m].copy()
            current_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_current_ratio_vap_M"+str(month)+".tif")
            current_ratio[landsea==-9999] = np.nan
            vapC[m]=varraw*current_ratio
        vapC[vapC<0.01]=0.01
        vapF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= vapC[m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_vap_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            vapF[m]=varaw*future_ratio
        vapF[vapF < 0.01] = 0.01
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_vap2070_2100_monthly_005d.npy", vapF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_vap1984_2014_monthly_005d.npy", vapC)
    else:
        vapF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= vapraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_vap_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            vapF[m]=varaw*future_ratio
        vapF[vapF < 0.01] = 0.01
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_vap2070_2100_monthly_005d.npy", vapF)
del vapraw,vapC,vapF


#vpd
vpdraw=np.load("H:/基于关系的研究/点位/原始变量/historical_vpd1984_2014_monthly_005d.npy")
vpdraw[vpdraw<0.01]=0.01

for experiment in experiments:
    if experiment!="historical":
        vpdC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = vpdraw[m].copy()
            current_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_current_ratio_vpd_M"+str(month)+".tif")
            current_ratio[landsea==-9999] = np.nan
            vpdC[m]=varraw*current_ratio
        vpdC[vpdC<0.01]=0.01
        vpdF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= vpdC[m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_vpd_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            vpdF[m]=varaw*future_ratio
        vpdF[vpdF < 0.01] = 0.01
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_vpd2070_2100_monthly_005d.npy", vpdF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_vpd1984_2014_monthly_005d.npy", vpdC)
    else:
        vpdF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= vpdraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_vpd_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            vpdF[m]=varaw*future_ratio
        vpdF[vpdF < 0.01] = 0.01
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_vpd2070_2100_monthly_005d.npy", vpdF)
del vpdraw,vpdC,vpdF

#tas
tasraw=np.load("H:/基于关系的研究/点位/原始变量/historical_tas1984_2014_monthly_005d.npy")
#np.save("H:/基于关系的研究/点位/原始变量/historical_tas1984_2014_monthly_005d.npy",tasraw)

for experiment in experiments:
    if experiment!="historical":
        tasC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = tasraw[m].copy()
            current_abs=readtif("E:/Mapresample/"+experiment+"_ESM_current_abs_tas_M"+str(month)+".tif")
            current_abs[landsea==-9999] = np.nan
            tasC[m]=varraw+current_abs

        tasF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= tasC[m].copy()
            future_abs=readtif("E:/Mapresample/"+experiment+"_ESM_future_abs_tas_M"+str(month)+".tif")
            future_abs[landsea==-9999]=np.nan
            tasF[m]=varaw+future_abs
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_tas2070_2100_monthly_005d.npy", tasF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_tas1984_2014_monthly_005d.npy", tasC)
    else:
        tasF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= tasraw [m].copy()
            future_abs=readtif("E:/Mapresample/"+experiment+"_ESM_future_abs_tas_M"+str(month)+".tif")
            future_abs[landsea==-9999]=np.nan
            tasF[m]=varaw+future_abs
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_tas2070_2100_monthly_005d.npy", tasF)

del tasraw,tasC,tasF


#tasmin
tasminraw=np.load("H:/基于关系的研究/点位/原始变量/historical_tasmin1984_2014_monthly_005d.npy")
#np.save("H:/基于关系的研究/点位/原始变量/historical_tasmin1984_2014_monthly_005d.npy",tasminraw)

for experiment in experiments:
    if experiment!="historical":
        tasminC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = tasminraw[m].copy()
            current_abs=readtif("E:/Mapresample/"+experiment+"_ESM_current_abs_tasmin_M"+str(month)+".tif")
            current_abs[landsea==-9999] = np.nan
            tasminC[m]=varraw+current_abs

        tasminF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= tasminC[m].copy()
            future_abs=readtif("E:/Mapresample/"+experiment+"_ESM_future_abs_tasmin_M"+str(month)+".tif")
            future_abs[landsea==-9999]=np.nan
            tasminF[m]=varaw+future_abs
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_tasmin2070_2100_monthly_005d.npy", tasminF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_tasmin1984_2014_monthly_005d.npy", tasminC)
    else:
        tasminF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= tasminraw [m].copy()
            future_abs=readtif("E:/Mapresample/"+experiment+"_ESM_future_abs_tasmin_M"+str(month)+".tif")
            future_abs[landsea==-9999]=np.nan
            tasminF[m]=varaw+future_abs
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_tasmin2070_2100_monthly_005d.npy", tasminF)

del tasminraw,tasminC,tasminF



#tasmax
tasmaxraw=np.load("H:/基于关系的研究/点位/原始变量/historical_tasmax1984_2014_monthly_005d.npy")
#np.save("H:/基于关系的研究/点位/原始变量/historical_tasmax1984_2014_monthly_005d.npy",tasmaxraw)

for experiment in experiments:
    if experiment!="historical":
        tasmaxC = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month = m + 1
            varraw = tasmaxraw[m].copy()
            current_abs=readtif("E:/Mapresample/"+experiment+"_ESM_current_abs_tasmax_M"+str(month)+".tif")
            current_abs[landsea==-9999] = np.nan
            tasmaxC[m]=varraw+current_abs

        tasmaxF=np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= tasmaxC[m].copy()
            future_abs=readtif("E:/Mapresample/"+experiment+"_ESM_future_abs_tasmax_M"+str(month)+".tif")
            future_abs[landsea==-9999]=np.nan
            tasmaxF[m]=varaw+future_abs
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_tasmax2070_2100_monthly_005d.npy", tasmaxF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_tasmax1984_2014_monthly_005d.npy", tasmaxC)
    else:
        tasmaxF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= tasmaxraw [m].copy()
            future_abs=readtif("E:/Mapresample/"+experiment+"_ESM_future_abs_tasmax_M"+str(month)+".tif")
            future_abs[landsea==-9999]=np.nan
            tasmaxF[m]=varaw+future_abs
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_tasmax2070_2100_monthly_005d.npy", tasmaxF)

del tasmaxraw,tasmaxC,tasmaxF

##treeFrac
#treeFrac=np.load("H:/基于关系的研究/点位/原始变量/VCF_tree_1984_2014.npy")
#np.save("H:/基于关系的研究/点位/原始变量/historical_treeFrac1984_2014_yr_005d.npy",treeFrac)

treeFracraw=np.load("H:/基于关系的研究/点位/原始变量/historical_treeFrac1984_2014_yr_005d.npy")

for experiment in experiments:
    if experiment!="historical":

        current_abs=readtif("E:/Mapresample/"+experiment+"_esm_current_abs_treefrac_mean.tif")
        current_abs[landsea==-9999] = np.nan
        treeFracC=treeFracraw+current_abs

        treeFracC[treeFracC>100]=100
        treeFracC[treeFracC<0]=0

        future_abs=readtif("E:/Mapresample/"+experiment+"_esm_future_abs_treefrac_mean.tif")
        future_abs[landsea==-9999] = np.nan
        treeFracF=treeFracC+future_abs

        treeFracF[treeFracF>100]=100
        treeFracF[treeFracF<0]=0

        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_treeFrac2070_2100_yr_005d.npy", treeFracF)
        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_treeFrac1984_2014_yr_005d.npy", treeFracC)
    else:
        future_abs=readtif("E:/Mapresample/"+experiment+"_esm_future_abs_treefrac_mean.tif")
        future_abs[landsea==-9999] = np.nan
        treeFracF=treeFracraw+future_abs

        treeFracF[treeFracF>100]=100
        treeFracF[treeFracF<0]=0

        np.save("H:/基于关系的研究/点位/原始变量/" + experiment + "_treeFrac2070_2100_yr_005d.npy", treeFracF)

del treeFracraw,treeFracC,treeFracF


###固定变量的填充
vargs=["aspect","bedrock","CLYPPT","SLTPPT","SNDPPT","CRFVOL","dem","plan_curve","pro_curve","slope","sed_depth",
       "soilclass","TRI","TWI"]

landsea = np.full([3600, 7200], -9999)
landsea[0:3000, :] = readtif("H:/CMIP5AI/landsea.tif")
landsea=landsea.astype(float)
land_MCD=np.load("H:/基于关系的研究/Gimms输出/land_MCD_m.npy")
landsea[land_MCD==0]=-9999

for varg in vargs:
    data=np.load("H:/基于关系的研究/点位/情景变量/固定变量/"+varg+".npy")
    NA=data[0,0]
    data[data==NA]=np.nan
    data[landsea==-9999]=np.nan
    for i in range(100):
        Mask = np.full([3600, 7200], np.nan)
        Mask[(landsea == 1) & (np.isnan(data))] = -1000
        data2 = np.full([3600, 7200], np.nan)
        fill_raster(data, data2)
        data[Mask == -1000] = data2[Mask == -1000]
        print(varg+" has proceesed "+str(i))
    np.save("H:/基于关系的研究/点位/情景变量/固定变量/"+varg+".npy",data)

#变化变量填充
import os
dy_files=os.listdir("H:/基于关系的研究/点位/情景变量/")
fill_files=[]
for file in dy_files:
    if ("koppen" in file)==False:
        if ".npy" in file:
            if "ATMO+PHYS" in file:
                fill_files.append(file)

for file in fill_files:#不是适用于treeFrac
    datavar = np.load("H:/基于关系的研究/点位/情景变量/"+file)
    datavar [landsea == -9999] = np.nan
    for i in range(30):
        Mask = np.full([3600, 7200], np.nan)
        Mask[(landsea == 1) & (np.isnan(datavar))] = -1000
        datavar2 = np.full([3600, 7200], np.nan)
        fill_raster(datavar, datavar2)
        datavar[Mask == -1000] = datavar2[Mask == -1000]
        print(file+" has proceesed "+str(i))
    np.save("H:/基于关系的研究/点位/情景变量/"+file,datavar)

for experiment in experiments:
    for tm in ["1984_2014","2070_2100"]:
        datavar = np.load("H:/基于关系的研究/点位/原始变量/"+experiment+"_treeFrac"+tm+ "_yr_005d.npy")
        datavar[landsea == -9999] = np.nan
        for i in range(30):
            Mask = np.full([3600, 7200], np.nan)
            Mask[(landsea == 1) & (np.isnan(datavar))] = -1000
            datavar2 = np.full([3600, 7200], np.nan)
            fill_raster(datavar, datavar2)
            datavar[Mask == -1000] = datavar2[Mask == -1000]
            print(experiment+tm+" has proceesed "+str(i))
        np.save("H:/基于关系的研究/点位/情景变量/" +experiment+"_treeFrac"+tm+ ".npy", datavar)





###########soilcalss填充

def fill_raster_class(data, data2):
    for i in range(2,3598,1):
        for j in range(2,7198,1):
            numbers=data[(i-2):(i+3),(j-2):(j+3)]
            numbers2=numbers[np.isnan(numbers)==False]
            numbers2=numbers2.astype(int)
            if len(numbers2)>0:
                counts=np.bincount(numbers2)
                data2[i,j]=np.argmax(counts)


data=np.load("H:/基于关系的研究/点位/情景变量/固定变量/soilclass.npy")
NA = data[0, 0]
data[data == NA] = np.nan
data[landsea == -9999] = np.nan


for i in range(100):
    Mask = np.full([3600, 7200], np.nan)
    Mask[(landsea == 1) & (np.isnan(data))] = -1000
    data2 = np.full([3600, 7200], np.nan)
    fill_raster_class(data, data2)
    data[Mask == -1000] = data2[Mask == -1000]
    print("soilclass" + " has proceesed " + str(i))
np.save("H:/基于关系的研究/点位/情景变量/固定变量/soilclass.npy", data)
data=np.load("H:/基于关系的研究/点位/情景变量/固定变量/soilclass.npy")


############koppen气候分类填充
def koppen(ta,ppt):
    climate = np.full([3600, 7200], np.nan)
    for i in range(3600):
        for j in range(7200):
            avgtemp = ta[:,i,j].copy()
            precip= ppt[:,i,j].copy()
            totalprecip = sum(precip)

            if np.isnan(totalprecip)==False:

                if max(avgtemp) >= 10.0:
                    sortavgtemp = avgtemp
                    sortavgtemp.sort()
                    tempaboveten = np.shape(np.where(avgtemp > 10.0))[1]
                    aridity = np.mean(avgtemp) * 20.0
                    warmprecip = sum(precip[3:9])
                    coolprecip = sum(precip[0:3]) + sum(precip[9:12])
                    summerprecip=precip[3:9]
                    winterprecip=precip[[0,1,2,9,10,11]]
                    if i>=2160:
                        warmprecip = sum(precip[0:3]) + sum(precip[9:12])
                        coolprecip = sum(precip[3:9])
                        summerprecip = precip[[0, 1, 2, 9, 10, 11]]
                        winterprecip = precip[3:9]
                    if warmprecip >= (0.70*totalprecip):
                        aridity = aridity + 280.0
                    elif coolprecip >=  (0.70*totalprecip):
                        aridity = aridity
                    else:
                        aridity = aridity + 140

                    # B group
                    if aridity>totalprecip:
                        # Semi-Arid/Steppe (BS)
                        if totalprecip >= (aridity/2) :
                            # Hot Semi-Arid (BSh)
                            if np.mean(avgtemp) >= 18.0:
                                climate[i, j] = 6  # 'BSh'
                            # Cold Semi-Arid (BSk)
                            else:
                                climate[i, j] = 7  # 'BSk'
                        else:
                            if np.mean(avgtemp) >= 18.0:
                                climate[i, j] = 4  # 'BWh'
                            else:
                                climate[i, j] = 5  # 'BWk'

                    # Group A (Tropical)
                    elif min(avgtemp) >= 18.0:
                        # Tropical Rainforest
                        if min(precip) >= 60.0:
                            climate[i,j] = 3 # 'Af'

                        # Tropical Monsoon
                        elif min(precip) < 60.0 and min(precip) >= totalprecip*0.04:
                            climate[i,j] = 1 # 'Am'
                        else:
                            climate[i, j] = 2  # 'Aw'
                        continue

                    # Group C (Temperate)
                    elif min(avgtemp) >= -3 and min(avgtemp) <= 18.0:
                        if min(summerprecip)<30 and min(summerprecip)< max(winterprecip)/3 and  warmprecip <coolprecip: #alpha

                            if  max(avgtemp) >= 22.0 :
                                climate[i,j] = 10#'Csa'
                            # Temperate Oceanic (Cfb)
                            elif tempaboveten >= 4.0:
                                climate[i,j] = 8#'Csb'
                            # Subpolar Oceanic (Cfc)
                            else:
                                climate[i,j] = 9# 'Cfc'
                            continue
                        # Monsoon-influenced humid subtropical (Cwa)
                        elif min(winterprecip)< max(summerprecip)/10 and coolprecip<warmprecip: #beita
                            if max(avgtemp)>=22:
                                climate[i,j] = 13#'Cwa'
                            elif tempaboveten >= 4:
                                climate[i,j] = 11#'Cwb'
                            else:
                                climate[i, j] = 12  # 'Cwc'
                        else:
                            if max(avgtemp)>=22:
                                climate[i, j] = 14  # 'Cfa'
                            elif tempaboveten >= 4:
                                climate[i, j] = 15  # 'Cfb'
                            else:
                                climate[i, j] = 16  # 'Cfc'
                    #group D (Continental)
                    else:
                        if min(summerprecip)<30 and min(summerprecip)< max(winterprecip)/3 and  warmprecip <coolprecip: #alpha
                            if min(avgtemp) <=-36:
                                climate[i,j] = 17 #'Dsd'

                            elif max(avgtemp) >=22:
                                climate[i,j] = 18 #'Dsa'

                            elif tempaboveten >=4:
                                climate[i,j] = 19# 'Dsb'

                            else:
                                climate[i,j] = 20#'Dsc'

                        elif min(winterprecip)< max(summerprecip)/10 and coolprecip<warmprecip: #beita

                            if min(avgtemp) <=-36:
                                climate[i,j] = 21# 'Dwd'

                            elif max(avgtemp) >=22:
                                climate[i,j] = 22#'Dwa'

                            elif tempaboveten >= 4:
                                climate[i,j] = 23#'Dwb'

                            else:
                                climate[i,j] = 24 #'Dwc'
                        else:
                            if min(avgtemp) <=-36:
                                climate[i,j] = 25# 'Dfd'

                            elif max(avgtemp) >=22:
                                climate[i,j] = 26 #'Dfa'

                            elif tempaboveten >= 4:
                                climate[i,j] = 27 # 'Dfb'

                            else:
                                climate[i,j] = 28#'Dfc'

                # Group E (Polar and alpine)
                if max(avgtemp) < 10.0:
                    # Tundra (ET)
                    if max(avgtemp) > 0.0:
                        climate[i,j] = 29#'ET'
                    # Ice cap (EF)
                    else:
                        climate[i,j] = 30# 'EF'
    return climate


for experiment in experiments:
    for time in ["1984_2014","2070_2100"]:
        ta = np.load("H:/基于关系的研究/点位/原始变量/"+experiment+"_tas"+time+"_monthly_005d.npy")
        ppt = np.load("H:/基于关系的研究/点位/原始变量/"+experiment+"_pr"+time+"_monthly_005d.npy")
        result=koppen(ta, ppt)
        np.save("H:/基于关系的研究/点位/情景变量/"+experiment+"_koppen_"+time+".npy",result)

experiment="ATMO+PHYS"
for time in ["1984_2014", "2070_2100"]:
    ta = np.load("H:/基于关系的研究/点位/原始变量/" + experiment + "_tas" + time + "_monthly_005d.npy")
    ppt = np.load("H:/基于关系的研究/点位/原始变量/" + experiment + "_pr" + time + "_monthly_005d.npy")
    result = koppen(ta, ppt)
    np.save("H:/基于关系的研究/点位/情景变量/" + experiment + "_koppen_" + time + ".npy", result)

experiments1=["historical","esmControl"]
for experiment in experiments1:
    for time in ["1984_2014","2070_2100"]:
        data=np.load("H:/基于关系的研究/点位/情景变量/"+experiment+"_koppen"+time+".npy")
        for i in range(100):
            Mask = np.full([3600, 7200], np.nan)
            Mask[(landsea == 1) & (np.isnan(data))] = -1000
            data2 = np.full([3600, 7200], np.nan)
            fill_raster_class(data, data2)
            data[Mask == -1000] = data2[Mask == -1000]
            print("soilclass" + " has proceesed " + str(i))
        np.save("H:/基于关系的研究/点位/情景变量/" + experiment + "_koppenc" + time + ".npy", data)

data=np.load("H:/基于关系的研究/点位/情景变量/historical_koppenc1984_2014.npy")

##############  生物多样性填充
MAM=readtif("H:/基于关系的研究/BiodiversityMapping_TIFFs_2019_03d14/MAM_v.tif")
MAM[MAM==255]=np.nan
datavar=MAM
datavar[landsea==-9999]=np.nan
for i in range(50):
    Mask = np.full([3600, 7200], np.nan)
    Mask[(landsea == 1) & (np.isnan(datavar))] = -1000
    datavar2 = np.full([3600, 7200], np.nan)
    fill_raster(datavar, datavar2)
    datavar[Mask == -1000] = datavar2[Mask == -1000]
    print("MAM" + " has proceesed " + str(i))
np.save("H:/基于关系的研究/点位/result/MAM.npy",datavar)

AMP=readtif("H:/基于关系的研究/BiodiversityMapping_TIFFs_2019_03d14/AMP_v.tif")
AMP[AMP==255]=np.nan
AMP[np.isnan(AMP)&(landsea==1)]=0
datavar=AMP
datavar[landsea==-9999]=np.nan
for i in range(50):
    Mask = np.full([3600, 7200], np.nan)
    Mask[(landsea == 1) & (np.isnan(datavar))] = -1000
    datavar2 = np.full([3600, 7200], np.nan)
    fill_raster(datavar, datavar2)
    datavar[Mask == -1000] = datavar2[Mask == -1000]
    print("MAM" + " has proceesed " + str(i))
np.save("H:/基于关系的研究/点位/result/AMP.npy",datavar)


Bird=readtif("H:/基于关系的研究/BiodiversityMapping_TIFFs_2019_03d14/Birds_total_v.tif")
Bird[Bird>6000]=np.nan
Bird[np.isnan(Bird)&(landsea==1)]=0
np.save("H:/基于关系的研究/点位/result/Bird.npy",Bird)

#########################################################################################################################
landsea = np.full([3600, 7200], -9999)
landsea[0:3000, :] = np.load("H:/基于关系的研究/点位/landsea.npy")
landsea=landsea.astype(float)
land_MCD=np.load("H:/基于关系的研究/Gimms输出/land_MCD_m.npy")
landsea[land_MCD==0]=-9999


MAM=np.load("H:/基于关系的研究/点位/result/MAM.npy")
MAM[landsea==-9999]=np.nan
print(np.nanpercentile(MAM,99.9))
AMP=np.load("H:/基于关系的研究/点位/result/AMP.npy")
AMP[landsea==-9999]=np.nan
print(np.nanpercentile(AMP,99.9))
Bird=np.load("H:/基于关系的研究/点位/result/Bird.npy")
Bird[landsea==-9999]=np.nan
print(np.nanpercentile(Bird,99.9))



import arcpy # Importing the ArcPy module
from arcpy import env
import os
from arcpy.sa import *

"""  matlab 代码
%%%%%%%%%%%%%% 存储0.5的tiff %%%%%%%%%%%%
names=dir(fullfile('H:/基于关系的研究/点位/ESM相对变化/','*.npy'));
for i=1:84
    name=names(i).name;
    namerd=['H:/基于关系的研究/点位/ESM相对变化/',name];
    data=readNPY(namerd);
    for j =1:12
        st=['_M',num2str(j),'.tif'];
        namesave=strrep(name,'.npy',st);
        save=['H:/基于关系的研究/点位/Map/',namesave];
        datasave=data(j,:,:);
        datasave=reshape(datasave,[360,720]);
        geotiffwrite(save,datasave,R);
    end
end
"""
experiments = ["esmControl", "esmFdbk2","esmFixClim2","historical","rcp45"]
files=os.listdir("E:/Map/")
f=[]
for i in files:
    if ("esmControl" in i) |("esmFdbk2" in i):
        f.append(i)

for i in files:
    if ("historical" in i) :
        if ("future" in i):
            if ("vap" not in i):
                if("vpd" not in i):
                    f.append(i)


f=f[0:72]
for file in f:
#def gis(file):
    #栅格转点
    inRaster = "E:/Map/"+file
    outPoint = file.replace(".tif",".shp")
    field = "VALUE"
    savepath1="E:/Mapresample/"+ outPoint
    arcpy.RasterToPoint_conversion(inRaster, savepath1, field)

    inPntFeat = savepath1
    zField = "GRID_CODE"
    cellSize = 0.05
    splineType = "REGULARIZED"
    weight = 0.1

    arcpy.env.extent = "-180.0 -90.0 180.0 90.0"
    arcpy.CheckOutExtension("Spatial")
    # Execute Spline
    outSpline = Spline(inPntFeat, zField, cellSize, splineType, weight)

    outSpline.save(savepath1.replace(".shp",".tif"))








#############################      machine learning process      ################################

#data = pd.read_csv('H:/基于关系的研究/点位/result/data_reg3.csv')

data = pd.read_csv('H:/基于关系的研究/点位/result/data_reg_Vcv.csv')

data_koppenc = pd.get_dummies(data['koppenc'],prefix='koppenc')
data_soilclass = pd.get_dummies(data['soilclass'],prefix='soilclass')

koppenc_name=data_koppenc.columns
soilclass_name=data_soilclass.columns

data.drop('koppenc',axis=1,inplace = True)
data.drop('soilclass',axis=1,inplace = True)
data.drop('Unnamed: 0',axis=1,inplace = True)

data = pd.merge(data,data_koppenc,left_index = True,right_index = True)
data = pd.merge(data,data_soilclass,left_index = True,right_index = True)

factors = list(data.columns)
factors.remove('MAM')
factors.remove('AMP')
factors.remove('Bird')
factors = [element for element in factors if 'cv' not in element]
'''
for var in ['bio2','bio3','bio4','bio5','bio6','bio7','bio8','bio9','bio10','bio11','bio13','bio14','bio15','bio16','bio17','bio18','bio19']:
    factors.remove(var)
'''
colx = factors
coly1 = 'MAM'
coly2 = 'AMP'
coly3 = 'Birds'

x,y1,y2,y3 = data[colx],data['MAM'],data['AMP'],data['Bird']
y1=y1/192
y2=y2/113
y3=y3/587

#计时器
# 切分数据
x_train,x_test,y1_train,y1_test,y2_train,y2_test,y3_train,y3_test =train_test_split(x,y1,y2,y3,test_size=0.25)
"""
x_train.to_csv("H:/基于关系的研究/点位/result/模型参数/x_train.csv")
x_test.to_csv("H:/基于关系的研究/点位/result/模型参数/x_test.csv")
y1_train.to_csv("H:/基于关系的研究/点位/result/模型参数/y1_train.csv")
y1_test.to_csv("H:/基于关系的研究/点位/result/模型参数/y1_test.csv")
y2_train.to_csv("H:/基于关系的研究/点位/result/模型参数/y2_train.csv")
y2_test.to_csv("H:/基于关系的研究/点位/result/模型参数/y2_test.csv")
y3_train.to_csv("H:/基于关系的研究/点位/result/模型参数/y3_train.csv")
y3_test.to_csv("H:/基于关系的研究/点位/result/模型参数/y3_test.csv")
lat_test.to_csv("H:/基于关系的研究/点位/result/模型参数/lat_test.csv")
lon_test.to_csv("H:/基于关系的研究/点位/result/模型参数/lon_test.csv")
np.save("H:/基于关系的研究/点位/result/模型参数/koppenc_name.npy",koppenc_name)
np.save("H:/基于关系的研究/点位/result/模型参数/soilclass_name.npy",soilclass_name)
"""


x_train.drop('lat',axis=1,inplace = True)
x_train.drop('long',axis=1,inplace = True)

lat_test=x_test.lat
lon_test=x_test.long
x_test.drop('lat',axis=1,inplace = True)
x_test.drop('long',axis=1,inplace = True)

# 打包成lgb数据格式
lgb_train = lgb.Dataset(x_train, y1_train)
lgb_eval = lgb.Dataset(x_test, y1_test, reference=lgb_train)


# 参数设置  为回归形式
params = {
    'booster': 'gbtree',
    'objective': 'regression',
    'num_leaves': 31,
    'subsample': 0.8,
    'bagging_freq': 1,
    'feature_fraction ': 0.8,
    'slient': 1,
    'learning_rate ': 0.1,
    'seed': 0
}
# 最大迭代次数1000，设置更大迭代次数更多，迭代次数增加倾向于过拟合，但是精度更高。迭代次数更少越不倾向于过拟合，但是精度更低。
# 100R方约95%，1000约97%，可以通过迭代次数控制过拟合与精度 不过愿意迭代几万次直到评价准则收敛也可以
num_rounds = 200
print('Start training...')
# 训练模型
gbm1 = lgb.train(params,lgb_train,num_rounds,valid_sets=lgb_train,early_stopping_rounds=5)
#pickle.dump(gbm1,open("H:/基于关系的研究/点位/result/模型参数/gbm1.dat","wb"))

del data, data_koppenc,data_soilclass
del x_train,x_test,y1_train,y1_test

print('Start predicting...')
y1_pred = gbm1.predict(x_test, num_iteration=gbm1.best_iteration)
np.save("H:/基于关系的研究/点位/result/模型参数/y1_pred.npy",y1_pred)

plot_pdp(x_train=x_train,var_name="gpp_mean",xlab="gpp_mean",ylab="AMP_richness",taxa="AMP",model=gbm2)
plot_pdp(x_train=x_train,var_name="rsds_mean",xlab="rsds_mean",ylab="AMP_richness",taxa="AMP",model=gbm2)
plot_pdp(x_train=x_train,var_name="rsds_std",xlab="rsds_std",ylab="AMP_richness",taxa="AMP",model=gbm2)
plot_pdp(x_train=x_train,var_name="bio3",xlab="bio3",ylab="AMP_richness",taxa="AMP",model=gbm2)




###################  AMP

lgb_train = lgb.Dataset(x_train, y2_train)
lgb_eval = lgb.Dataset(x_test, y2_test, reference=lgb_train)

params = {
    'booster': 'gbtree',
    'objective': 'regression',
    'num_leaves': 31,
    'subsample': 0.8,
    'bagging_freq': 1,
    'feature_fraction ': 0.8,
    'slient': 1,
    'learning_rate ': 0.1,
    'seed': 0
}
num_rounds = 200
print('Start training...')
gbm2 = lgb.train(params, lgb_train, num_rounds, valid_sets=lgb_train, early_stopping_rounds=5)
pickle.dump(gbm2,open("H:/基于关系的研究/点位/result/模型参数/gbm2.dat","wb"))

print('Start predicting...')
y2_pred = gbm2.predict(x_test, num_iteration=gbm2.best_iteration)
np.save("H:/基于关系的研究/点位/result/模型参数/y2_pred.npy",y2_pred)




plt.figure(figsize=(6,8))
plt.subplot(1,2,1)





##################       Birds

lgb_train = lgb.Dataset(x_train, y3_train)
lgb_eval = lgb.Dataset(x_test, y3_test, reference=lgb_train)

params = {
    'booster': 'gbtree',
    'objective': 'regression',
    'num_leaves': 31,
    'subsample': 0.8,
    'bagging_freq': 1,
    'feature_fraction ': 0.8,
    'slient': 1,
    'learning_rate ': 0.1,
    'seed': 0
}
num_rounds = 200
print('Start training...')
gbm3 = lgb.train(params, lgb_train, num_rounds, valid_sets=lgb_train, early_stopping_rounds=5)
pickle.dump(gbm3,open("H:/基于关系的研究/点位/result/模型参数/gbm3.dat","wb"))

print('Start predicting...')
y3_pred = gbm3.predict(x_test, num_iteration=gbm3.best_iteration)
np.save("H:/基于关系的研究/点位/result/模型参数/y3_pred.npy",y3_pred)

# 计算R方
y3_bar = np.mean(y3_test)
SST = np.sum((y3_test - y3_bar) ** 2)
SSE = np.sum((y3_pred - y3_bar) ** 2)
print(SSE / SST)
RSME3=np.sqrt(np.sum((y3_test - y3_pred) ** 2) / len(y3_test))

y3_test.reset_index(drop=True,inplace = True)
df3 = pd.DataFrame([y3_test,y3_pred]).T
df3.columns = ['Birds','Birdspred']
df3.head(50)

x_refre=[0,600]
y_refre=[0,600]
plt.scatter(y3_test, y3_pred,s=5)
plt.plot(x_refre,y_refre,c="red",linestyle='--')




####################    验证图 ，散点

###  分biome
from numba import jit
def extract_value(map, points):
    coldata=np.full([points.shape[0],1],np.nan)
    for i in range(points.shape[0]):
        lat = points[i,0]
        lat = int((-lat+90)//0.05)
        long = points[i,1]
        long =int((long+180)//0.05)
        coldata[i,0]=map[lat,long]
    points=np.c_[points,coldata]
    return points
lat_test=pd.read_csv("H:/基于关系的研究/点位/result/模型参数/lat_test.csv")
lon_test=pd.read_csv("H:/基于关系的研究/点位/result/模型参数/lon_test.csv")
lon_test=np.array(lon_test).reshape(1,-1)
lat_test=np.array(lat_test).reshape(1,-1)
points=np.full([32199,2],np.nan)
points[:,0]=lat_test
points[:,1]=lon_test
land_MCD=np.load("H:/基于关系的研究/Gimms输出/land_MCD_m.npy")
land_MCD[land_MCD==1]=21
land_MCD[land_MCD==2]=21
land_MCD[land_MCD==3]=21
land_MCD[land_MCD==4]=21
land_MCD[land_MCD==5]=21
land_MCD[land_MCD==6]=22
land_MCD[land_MCD==7]=22
land_MCD[land_MCD==8]=23
land_MCD[land_MCD==9]=23
land_MCD[land_MCD==10]=23
land_MCD[land_MCD==11]=24
land_MCD[land_MCD==12]=24
land_MCD[land_MCD==13]=24
land_MCD[land_MCD==14]=24
land_MCD[land_MCD==15]=24
land_MCD[land_MCD==16]=24
land_MCD[land_MCD==17]=24



points=extract_value(land_MCD, points)
y1_test=pd.read_csv("H:/基于关系的研究/点位/result/模型参数/y1_test.csv")
y2_test=pd.read_csv("H:/基于关系的研究/点位/result/模型参数/y2_test.csv")
y3_test=pd.read_csv("H:/基于关系的研究/点位/result/模型参数/y3_test.csv")
points= np.c_[points,y1_test]
points= np.c_[points,y2_test]
points= np.c_[points,y3_test]

y1_pred=np.load("H:/基于关系的研究/点位/result/模型参数/y1_pred.npy")
y2_pred=np.load("H:/基于关系的研究/点位/result/模型参数/y2_pred.npy")
y3_pred=np.load("H:/基于关系的研究/点位/result/模型参数/y3_pred.npy")
points= np.c_[points,y1_pred]
points= np.c_[points,y2_pred]
points= np.c_[points,y3_pred]

points=pd.DataFrame(points)
points.columns = ["lat","long","landcover","y1_test","y2_test","y3_test","y1_pred","y2_pred","y3_pred"]

points.y1_test=points.y1_test/192
points.y2_test=points.y2_test/113
points.y3_test=points.y3_test/587

points.y1_pred=points.y1_pred/192
points.y2_pred=points.y2_pred/113
points.y3_pred=points.y3_pred/587


plt.scatter(points.y3_test,points.y3_pred,s=5,alpha=0.5)


#forests

def r2(y_true, y_pred):
    """
    Calculation of the unadjusted r-squared, goodness of fit metric
    """
    sse  = np.square( y_pred - y_true ).sum()
    sst  = np.square( y_true - y_true.mean() ).sum()
    return 1 - sse/sst

def rsme(y_true, y_pred):
    v=np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))
    return v

points.iloc[:,3:10]=points.iloc[:,3:10]*100

plt.figure(figsize=(17.5,10))

points1=points[points.landcover==21]
print(r2(points1.y1_test,points1.y1_pred))
print(rsme(points1.y1_test,points1.y1_pred))
plt.subplot(3,5,1)
plt.scatter(points1.y1_test, points1.y1_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed MAM")
#plt.ylabel("Projected MAM")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points1.y1_test,points1.y1_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points1.y1_test,points1.y1_pred))[0:5], fontsize=12)

points2=points[points.landcover==22]
print(r2(points2.y1_test,points2.y1_pred))
print(rsme(points2.y1_test,points2.y1_pred))
plt.subplot(3,5,2)
plt.scatter(points2.y1_test, points2.y1_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed MAM")
#plt.ylabel("Projected MAM")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points2.y1_test,points2.y1_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points2.y1_test,points2.y1_pred))[0:5], fontsize=12)

points3=points[points.landcover==23]
print(r2(points3.y1_test,points3.y1_pred))
print(rsme(points3.y1_test,points3.y1_pred))
plt.subplot(3,5,3)
plt.scatter(points3.y1_test, points3.y1_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed MAM")
#plt.ylabel("Projected MAM")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points3.y1_test,points3.y1_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points3.y1_test,points3.y1_pred))[0:5], fontsize=12)

points4=points[points.landcover==24]
print(r2(points4.y1_test,points4.y1_pred))
print(rsme(points4.y1_test,points4.y1_pred))
plt.subplot(3,5,4)
plt.scatter(points4.y1_test, points4.y1_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed MAM")
#plt.ylabel("Projected MAM")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points4.y1_test,points4.y1_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points4.y1_test,points4.y1_pred))[0:5], fontsize=12)

plt.subplot(3,5,5)
plt.scatter(points.y1_test, points.y1_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed MAM")
#plt.ylabel("Projected MAM")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points.y1_test,points.y1_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points.y1_test,points.y1_pred))[0:5], fontsize=12)


points5=points[points.landcover==21]
print(r2(points5.y2_test,points5.y2_pred))
print(rsme(points5.y2_test,points5.y2_pred))
plt.subplot(3,5,6)
plt.scatter(points5.y2_test, points5.y2_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed AMP")
#plt.ylabel("Projected AMP")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points5.y2_test,points5.y2_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points5.y2_test,points5.y2_pred))[0:5], fontsize=12)

points6=points[points.landcover==22]
print(r2(points6.y2_test,points6.y2_pred))
print(rsme(points6.y2_test,points6.y2_pred))
plt.subplot(3,5,7)
plt.scatter(points6.y2_test, points6.y2_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed AMP")
#plt.ylabel("Projected AMP")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points6.y2_test,points6.y2_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points6.y2_test,points6.y2_pred))[0:5], fontsize=12)


points7=points[points.landcover==23]
print(r2(points7.y2_test,points7.y2_pred))
print(rsme(points7.y2_test,points7.y2_pred))
plt.subplot(3,5,8)
plt.scatter(points7.y2_test, points7.y2_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed AMP")
#plt.ylabel("Projected AMP")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points7.y2_test,points7.y2_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points7.y2_test,points7.y2_pred))[0:5], fontsize=12)

points8=points[points.landcover==24]
print(r2(points8.y2_test,points8.y2_pred))
print(rsme(points8.y2_test,points8.y2_pred))
plt.subplot(3,5,9)
plt.scatter(points8.y2_test, points8.y2_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed AMP")
#plt.ylabel("Projected AMP")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points8.y2_test,points8.y2_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points8.y2_test,points8.y2_pred))[0:5], fontsize=12)

plt.subplot(3,5,10)
plt.scatter(points.y2_test, points.y2_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed AMP")
#plt.ylabel("Projected AMP")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points.y2_test,points.y2_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points.y2_test,points.y2_pred))[0:5], fontsize=12)


points9=points[points.landcover==21]
print(r2(points9.y3_test,points9.y3_pred))
print(rsme(points9.y3_test,points9.y3_pred))
plt.subplot(3,5,11)
plt.scatter(points9.y3_test, points9.y3_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed Bird")
#plt.ylabel("Projected Bird")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points9.y3_test,points9.y3_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points9.y3_test,points9.y3_pred))[0:5], fontsize=12)

points10=points[points.landcover==22]
print(r2(points10.y3_test,points10.y3_pred))
print(rsme(points10.y3_test,points10.y3_pred))
plt.subplot(3,5,12)
plt.scatter(points10.y3_test, points10.y3_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed Bird")
#plt.ylabel("Projected Bird")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points10.y3_test,points10.y3_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points10.y3_test,points10.y3_pred))[0:5], fontsize=12)

points11=points[points.landcover==23]
print(r2(points11.y3_test,points11.y3_pred))
print(rsme(points11.y3_test,points11.y3_pred))
plt.subplot(3,5,13)
plt.scatter(points11.y3_test, points11.y3_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed Bird")
#plt.ylabel("Projected Bird")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points11.y3_test,points11.y3_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points11.y3_test,points11.y3_pred))[0:5], fontsize=12)

points12=points[points.landcover==24]
print(r2(points12.y3_test,points12.y3_pred))
print(rsme(points12.y3_test,points12.y3_pred))
plt.subplot(3,5,14)
plt.scatter(points12.y3_test, points12.y3_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed Bird")
#plt.ylabel("Projected Bird")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points12.y3_test,points12.y3_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points12.y3_test,points12.y3_pred))[0:5], fontsize=12)

plt.subplot(3,5,15)
plt.scatter(points.y3_test, points.y3_pred,s=10,alpha=0.2)
plt.plot([0,105],[0,105],c="red",linestyle='--')
#plt.xlabel("Observed Bird")
#plt.ylabel("Projected Bird")
plt.yticks(rotation=90)
plt.text(10.5, 93.45, '$R^2=$'+str(r2(points.y3_test,points.y3_pred))[0:5], fontsize=12)
plt.text(10.5, 84, 'RSME='+str(rsme(points.y3_test,points.y3_pred))[0:5], fontsize=12)


plt.savefig("H:/基于关系的研究/点位/result/图/Fig1.tiff",dpi=500)




#######################     pdp分析
def plot_pdp(x_train,var_name,xlab,ylab,taxa,model):
    x_train_pdp=x_train
    pdp_AMP = pdp.pdp_isolate(model=model, dataset=x_train_pdp, model_features=x_train_pdp.columns , feature=var_name,num_grid_points=30)
    pdp.pdp_plot(pdp_AMP, var_name,center=True,figsize=(6,6))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig("H:/基于关系的研究/点位/result/图/"+taxa+"_"+var_name+".tiff",dpi=300)
    plt.close()
x_train=pd.read_csv("H:/基于关系的研究/点位/result/模型参数/x_train.csv")
plot_pdp(x_train=x_train,var_name="rsds_mean",xlab="rsds_mean",ylab="MAM_richness",taxa="MAM",model=gbm1)
plot_pdp(x_train=x_train,var_name="rsds_cv",xlab="rsds_cv",ylab="MAM_richness",taxa="MAM",model=gbm1)
plot_pdp(x_train=x_train,var_name="vap_cv",xlab="vap_cv",ylab="MAM_richness",taxa="MAM",model=gbm1)
#plot_pdp(x_train=x_train,var_name="sfcWind_mean",xlab="sfcWind_mean",ylab="MAM_richness",taxa="MAM",model=gbm1)

plot_pdp(x_train=x_train,var_name="rsds_mean",xlab="rsds_mean",ylab="AMP_richness",taxa="AMP",model=gbm2)
plot_pdp(x_train=x_train,var_name="rsds_cv",xlab="rsds_cv",ylab="AMP_richness",taxa="AMP",model=gbm2)
plot_pdp(x_train=x_train,var_name="vap_cv",xlab="vap_cv",ylab="AMP_richness",taxa="AMP",model=gbm2)
#plot_pdp(x_train=x_train,var_name="bio3",xlab="bio3",ylab="AMP_richness",taxa="AMP",model=gbm2)

plot_pdp(x_train=x_train,var_name="rsds_mean",xlab="rsds_mean",ylab="Bird_richness",taxa="Bird",model=gbm3)
plot_pdp(x_train=x_train,var_name="rsds_cv",xlab="rsds_cv",ylab="Bird_richness",taxa="Bird",model=gbm3)
plot_pdp(x_train=x_train,var_name="vap_cv",xlab="vap_cv",ylab="Bird_richness",taxa="Bird",model=gbm3)
#plot_pdp(x_train=x_train,var_name="sfcWind_mean",xlab="sfcWind_mean",ylab="Bird_richness",taxa="Bird",model=gbm3)

plt.figure(figsize=(10,7.5))
plt.subplots_adjust(wspace=0, hspace=0,left=0, bottom=0, right = 1, top = 1)
fig1=plt.imread("H:/基于关系的研究/点位/result/图/MAM_rsds_mean.tiff")
fig1=fig1[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,3,1)
plt.imshow(fig1)
plt.axis('off')

fig4=plt.imread("H:/基于关系的研究/点位/result/图/MAM_rsds_cv.tiff")
fig4=fig4[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,3,4)
plt.imshow(fig4)
plt.axis('off')

fig7=plt.imread("H:/基于关系的研究/点位/result/图/MAM_vap_cv.tiff")
fig7=fig7[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,3,7)
plt.imshow(fig7)
plt.axis('off')

'''
fig10=plt.imread("H:/基于关系的研究/点位/result/图/MAM_sfcWind_mean.tiff")
fig10=fig10[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(4,3,10)
plt.imshow(fig10)
plt.axis('off')
'''

fig2=plt.imread("H:/基于关系的研究/点位/result/图/AMP_rsds_mean.tiff")
fig2=fig2[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,3,2)
plt.imshow(fig2)
plt.axis('off')

fig5=plt.imread("H:/基于关系的研究/点位/result/图/AMP_rsds_cv.tiff")
fig5=fig5[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,3,5)
plt.imshow(fig5)
plt.axis('off')

fig8=plt.imread("H:/基于关系的研究/点位/result/图/AMP_vap_cv.tiff")
fig8=fig8[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,3,8)
plt.imshow(fig8)
plt.axis('off')

'''
fig11=plt.imread("H:/基于关系的研究/点位/result/图/AMP_bio3.tiff")
fig11=fig11[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(4,3,11)
plt.imshow(fig11)
plt.axis('off')
'''

fig3=plt.imread("H:/基于关系的研究/点位/result/图/Bird_rsds_mean.tiff")
fig3=fig3[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,3,3)
plt.imshow(fig3)
plt.axis('off')

fig6=plt.imread("H:/基于关系的研究/点位/result/图/Bird_rsds_cv.tiff")
fig6=fig6[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,3,6)
plt.imshow(fig6)
plt.axis('off')

fig9=plt.imread("H:/基于关系的研究/点位/result/图/Bird_vap_cv.tiff")
fig9=fig9[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,3,9)
plt.imshow(fig9)
plt.axis('off')

'''
fig12=plt.imread("H:/基于关系的研究/点位/result/图/Bird_sfcWind_mean.tiff")
fig12=fig12[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(4,3,12)
plt.imshow(fig12)
plt.axis('off')
'''

plt.savefig("H:/基于关系的研究/点位/result/图/pdp_all.tiff",dpi=500)
plt.savefig("H:/基于关系的研究/点位/result/图/FigS3.tiff",dpi=500)



img1=plt.imread("H:/基于关系的研究/点位/result/图/MAM_rsds_std.tiff")
img1=img1[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,2,2)
plt.imshow(img1)
plt.axis('off')

img2=plt.imread("H:/基于关系的研究/点位/result/图/MAM_rsds_mean.tiff")
img2=img2[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,2,4)
plt.imshow(img2)
plt.axis('off')

img3=plt.imread("H:/基于关系的研究/点位/result/图/MAM_rsds_mean.tiff")
img3=img3[np.r_[np.arange(300,500,1),np.arange(600,1800,1)],:,:]
plt.subplot(3,2,6)
plt.imshow(img3)
plt.axis('off')

plt.savefig("H:/基于关系的研究/点位/result/图/MAM_all.tiff",dpi=500)









#########   情景模拟   ##########

sample=pd.read_csv("H:/基于关系的研究/点位/All_predictor已填补/data_points.csv")
sample=sample[sample.land_MCD_m>0]
sample=sample[sample.landsea==1]
sample=sample.iloc[:,0:2]

lgb.plot_impoFrtance(gbm1, max_num_features=50,figsize=(4,9),height=0.8)
plt.show()
plt.savefig("H:/基于关系的研究/点位/result/图/importance_MAM.tiff",dpi=500)

pdp_MAM = pdp.pdp_isolate(model=gbm1, dataset=x_train, model_features=factors , feature='rsds_std',num_grid_points=30)
pdp.pdp_plot(pdp_MAM, 'rsds_std',center==True,figsize=(5,5))
plt.savefig("H:/基于关系的研究/点位/result/图/pdp_MAM_rsds_std.tiff",dpi=500)

pdp_MAM = pdp.pdp_isolate(model=gbm1, dataset=x_test, model_features=factors , feature='rsds_mean',num_grid_points=30)
pdp.pdp_plot(pdp_MAM, 'rsds_mean',center=False,figsize=(5,5))
plt.savefig("H:/基于关系的研究/点位/result/图/pdp_MAM_rsds_mean.tiff",dpi=500)

pdp_MAM = pdp.pdp_isolate(model=gbm1, dataset=x_test, model_features=factors , feature='vap_std',num_grid_points=30)
pdp.pdp_plot(pdp_MAM, 'vap_std',center=True,figsize=(5,5))
plt.savefig("H:/基于关系的研究/点位/result/图/pdp_MAM_vap_std.tiff",dpi=500)

pdp_MAM = pdp.pdp_isolate(model=gbm1, dataset=x_test, model_features=factors , feature='rsds_min',num_grid_points=30)
pdp.pdp_plot(pdp_MAM, 'rsds_min',center=True,figsize=(5,5))
plt.savefig("H:/基于关系的研究/点位/result/图/pdp_MAM_rsds_min.tiff",dpi=500)

pdp_MAM = pdp.pdp_isolate(model=gbm1, dataset=x_test, model_features=factors , feature='bio2',num_grid_points=30)
pdp.pdp_plot(pdp_MAM, 'bio2',center=True,figsize=(5,5))
plt.savefig("H:/基于关系的研究/点位/result/图/pdp_MAM_bio2.tiff",dpi=500)

pdp_MAM = pdp.pdp_isolate(model=gbm1, dataset=x_test, model_features=factors , feature='sfcWind_mean',num_grid_points=30)
pdp.pdp_plot(pdp_MAM, 'sfcWind_mean',center=True,figsize=(5,5))


