import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
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


landsea=np.full([3600,7200],-9999)
landsea=np.full([3600,7200],-9999)
landsea[0:3000,:]=readtif(r"H:\relatioship\点位\landsea.tif")
experiments =["1pctCO2"] #["esmControl", "esmFdbk2","esmFixClim2","historical"]
var1=[["evspsbl",0.1],["pr",0.1],["rsds",0.1],["sfcWind",0.001],["gpp",0.1],["lai",0.01],["mrso",1],["vpd",0.01],["vap",0.01]]
var2=["hur","tas","tasmax","tasmin"]

#aet
aetraw=np.load("H:/relatioship/点位/原始变量/historical_aet1984_2014_monthly_005d.npy")
aetraw[aetraw<0.1]=0.1
np.save("H:/relatioship/点位/原始变量/historical_aet1984_2014_monthly_005d.npy",aetraw)

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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_aet2070_2100_monthly_005d.npy", aetF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_aet1984_2014_monthly_005d.npy", aetC)
    else:
        aetF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= aetraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_evspsbl_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            aetF[m]=varaw*future_ratio
        aetF[aetF < 0.1] = 0.1
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_aet2070_2100_monthly_005d.npy", aetF)
del aetraw,aetC,aetF


#pr
prraw=np.load("H:/relatioship/点位/原始变量/historical_pr1984_2014_monthly_005d.npy")

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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_pr2070_2100_monthly_005d.npy", prF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_pr1984_2014_monthly_005d.npy", prC)
    else:
        prF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= prraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_pr_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            prF[m]=varaw*future_ratio
        prF[prF < 0.1] = 0.1
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_pr2070_2100_monthly_005d.npy", prF)
del prraw,prC,prF

#rsds
rsdsraw=np.load("H:/relatioship/点位/原始变量/historical_rsds1984_2014_monthly_005d.npy")
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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_rsds2070_2100_monthly_005d.npy", rsdsF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_rsds1984_2014_monthly_005d.npy", rsdsC)
    else:
        rsdsF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= rsdsraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_rsds_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            rsdsF[m]=varaw*future_ratio
        rsdsF[rsdsF < 0.1] = 0.1
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_rsds2070_2100_monthly_005d.npy", rsdsF)
del rsdsraw,rsdsC,rsdsF

#sfcWind
sfcWindraw=np.load("H:/relatioship/点位/原始变量/historical_sfcWind1984_2014_monthly_005d.npy")
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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_sfcWind2070_2100_monthly_005d.npy", sfcWindF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_sfcWind1984_2014_monthly_005d.npy", sfcWindC)
    else:
        sfcWindF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= sfcWindraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_sfcWind_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            sfcWindF[m]=varaw*future_ratio
        sfcWindF[sfcWindF < 0.001] = 0.001
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_sfcWind2070_2100_monthly_005d.npy", sfcWindF)
del sfcWindraw,sfcWindC,sfcWindF

#gpp
#gppraw=np.load("H:/relatioship/点位/原始变量/GPP1984_2014_monthly.npy")
#gppraw=gppraw*30
#gppraw[gppraw<0.1]=0.1
#np.save("H:/relatioship/点位/原始变量/historical_gpp1984_2014_monthly_005d.npy",gppraw)
"""
for i in range(12):
    data_mon=gppraw[i].copy()
    data_mon[np.where(np.isnan(data_mon) & (landsea!=-9999))]=0.1
    gppraw[i]=data_mon
"""
gppraw=np.load("H:/relatioship/点位/原始变量/historical_gpp1984_2014_monthly_005d.npy") #检查最小值是否为0.1
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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_gpp2070_2100_monthly_005d.npy", gppF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_gpp1984_2014_monthly_005d.npy", gppC)
    else:
        gppF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= gppraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_gpp_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            gppF[m]=varaw*future_ratio
        gppF[gppF < 0.1] = 0.1
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_gpp2070_2100_monthly_005d.npy", gppF)
del gppraw,gppC,gppF


lairaw=np.load("H:/relatioship/点位/原始变量/historical_lai1984_2014_monthly_005d.npy")
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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_lai2070_2100_monthly_005d.npy", laiF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_lai1984_2014_monthly_005d.npy", laiC)
    else:
        laiF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= lairaw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_lai_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            laiF[m]=varaw*future_ratio
        laiF[laiF < 0.01] = 0.01
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_lai2070_2100_monthly_005d.npy", laiF)
del lairaw,laiC,laiF

#mrso
mrsoraw=np.load("H:/relatioship/点位/原始变量/historical_mrso1984_2014_monthly_005d.npy")
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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_mrso2070_2100_monthly_005d.npy", mrsoF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_mrso1984_2014_monthly_005d.npy", mrsoC)
    else:
        mrsoF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= mrsoraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_mrso_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            mrsoF[m]=varaw*future_ratio
        mrsoF[mrsoF < 0.01] = 0.01
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_mrso2070_2100_monthly_005d.npy", mrsoF)
del mrsoraw,mrsoC,mrsoF

#vap
vapraw=np.load("H:/relatioship/点位/原始变量/historical_vap1984_2014_monthly_005d.npy")
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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_vap2070_2100_monthly_005d.npy", vapF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_vap1984_2014_monthly_005d.npy", vapC)
    else:
        vapF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= vapraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_vap_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            vapF[m]=varaw*future_ratio
        vapF[vapF < 0.01] = 0.01
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_vap2070_2100_monthly_005d.npy", vapF)
del vapraw,vapC,vapF


#vpd
vpdraw=np.load("H:/relatioship/点位/原始变量/historical_vpd1984_2014_monthly_005d.npy")
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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_vpd2070_2100_monthly_005d.npy", vpdF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_vpd1984_2014_monthly_005d.npy", vpdC)
    else:
        vpdF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= vpdraw [m].copy()
            future_ratio=readtif("E:/Mapresample/"+experiment+"_ESM_future_ratio_vpd_M"+str(month)+".tif")
            future_ratio[landsea==-9999]=np.nan
            vpdF[m]=varaw*future_ratio
        vpdF[vpdF < 0.01] = 0.01
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_vpd2070_2100_monthly_005d.npy", vpdF)
del vpdraw,vpdC,vpdF

#tas
tasraw=np.load("H:/relatioship/点位/原始变量/historical_tas1984_2014_monthly_005d.npy")
#np.save("H:/relatioship/点位/原始变量/historical_tas1984_2014_monthly_005d.npy",tasraw)

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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_tas2070_2100_monthly_005d.npy", tasF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_tas1984_2014_monthly_005d.npy", tasC)
    else:
        tasF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= tasraw [m].copy()
            future_abs=readtif("E:/Mapresample/"+experiment+"_ESM_future_abs_tas_M"+str(month)+".tif")
            future_abs[landsea==-9999]=np.nan
            tasF[m]=varaw+future_abs
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_tas2070_2100_monthly_005d.npy", tasF)

del tasraw,tasC,tasF


#tasmin
tasminraw=np.load("H:/relatioship/点位/原始变量/historical_tasmin1984_2014_monthly_005d.npy")
#np.save("H:/relatioship/点位/原始变量/historical_tasmin1984_2014_monthly_005d.npy",tasminraw)

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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_tasmin2070_2100_monthly_005d.npy", tasminF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_tasmin1984_2014_monthly_005d.npy", tasminC)
    else:
        tasminF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= tasminraw [m].copy()
            future_abs=readtif("E:/Mapresample/"+experiment+"_ESM_future_abs_tasmin_M"+str(month)+".tif")
            future_abs[landsea==-9999]=np.nan
            tasminF[m]=varaw+future_abs
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_tasmin2070_2100_monthly_005d.npy", tasminF)

del tasminraw,tasminC,tasminF



#tasmax
tasmaxraw=np.load("H:/relatioship/点位/原始变量/historical_tasmax1984_2014_monthly_005d.npy")
#np.save("H:/relatioship/点位/原始变量/historical_tasmax1984_2014_monthly_005d.npy",tasmaxraw)

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
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_tasmax2070_2100_monthly_005d.npy", tasmaxF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_tasmax1984_2014_monthly_005d.npy", tasmaxC)
    else:
        tasmaxF = np.full([12, 3600, 7200], np.nan)
        for m in range(12):
            month=m+1
            varaw= tasmaxraw [m].copy()
            future_abs=readtif("E:/Mapresample/"+experiment+"_ESM_future_abs_tasmax_M"+str(month)+".tif")
            future_abs[landsea==-9999]=np.nan
            tasmaxF[m]=varaw+future_abs
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_tasmax2070_2100_monthly_005d.npy", tasmaxF)

del tasmaxraw,tasmaxC,tasmaxF

##treeFrac
#treeFrac=np.load("H:/relatioship/点位/原始变量/VCF_tree_1984_2014.npy")
#np.save("H:/relatioship/点位/原始变量/historical_treeFrac1984_2014_yr_005d.npy",treeFrac)

treeFracraw=np.load("H:/relatioship/点位/原始变量/historical_treeFrac1984_2014_yr_005d.npy")

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

        np.save("H:/relatioship/点位/原始变量/" + experiment + "_treeFrac2070_2100_yr_005d.npy", treeFracF)
        np.save("H:/relatioship/点位/原始变量/" + experiment + "_treeFrac1984_2014_yr_005d.npy", treeFracC)
    else:
        future_abs=readtif("E:/Mapresample/"+experiment+"_esm_future_abs_treefrac_mean.tif")
        future_abs[landsea==-9999] = np.nan
        treeFracF=treeFracraw+future_abs

        treeFracF[treeFracF>100]=100
        treeFracF[treeFracF<0]=0

        np.save("H:/relatioship/点位/原始变量/" + experiment + "_treeFrac2070_2100_yr_005d.npy", treeFracF)

del treeFracraw,treeFracC,treeFracF


###固定变量的填充
vargs=["aspect","bedrock","CLYPPT","SLTPPT","SNDPPT","CRFVOL","dem","plan_curve","pro_curve","slope","sed_depth",
       "soilclass","TRI","TWI"]

landsea = np.full([3600, 7200], -9999)
landsea[0:3000, :] = readtif("H:/CMIP5AI/landsea.tif")
landsea=landsea.astype(float)
land_MCD=np.load("H:/relatioship/Gimms输出/land_MCD_m.npy")
landsea[land_MCD==0]=-9999

for varg in vargs:
    data=np.load("H:/relatioship/点位/情景变量/固定变量/"+varg+".npy")
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
    np.save("H:/relatioship/点位/情景变量/固定变量/"+varg+".npy",data)

#变化变量填充
import os
dy_files=os.listdir("H:/relatioship/点位/情景变量/")
fill_files=[]
for file in dy_files:
    if ("koppen" in file)==False:
        if ".npy" in file:
            if "ATMO+PHYS" in file:
                fill_files.append(file)

for file in fill_files:#不是适用于treeFrac
    datavar = np.load("H:/relatioship/点位/情景变量/"+file)
    datavar [landsea == -9999] = np.nan
    for i in range(30):
        Mask = np.full([3600, 7200], np.nan)
        Mask[(landsea == 1) & (np.isnan(datavar))] = -1000
        datavar2 = np.full([3600, 7200], np.nan)
        fill_raster(datavar, datavar2)
        datavar[Mask == -1000] = datavar2[Mask == -1000]
        print(file+" has proceesed "+str(i))
    np.save("H:/relatioship/点位/情景变量/"+file,datavar)

for experiment in experiments:
    for tm in ["1984_2014","2070_2100"]:
        datavar = np.load("H:/relatioship/点位/原始变量/"+experiment+"_treeFrac"+tm+ "_yr_005d.npy")
        datavar[landsea == -9999] = np.nan
        for i in range(30):
            Mask = np.full([3600, 7200], np.nan)
            Mask[(landsea == 1) & (np.isnan(datavar))] = -1000
            datavar2 = np.full([3600, 7200], np.nan)
            fill_raster(datavar, datavar2)
            datavar[Mask == -1000] = datavar2[Mask == -1000]
            print(experiment+tm+" has proceesed "+str(i))
        np.save("H:/relatioship/点位/情景变量/" +experiment+"_treeFrac"+tm+ ".npy", datavar)





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


data=np.load("H:/relatioship/点位/情景变量/固定变量/soilclass.npy")
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
np.save("H:/relatioship/点位/情景变量/固定变量/soilclass.npy", data)
data=np.load("H:/relatioship/点位/情景变量/固定变量/soilclass.npy")


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
        ta = np.load("H:/relatioship/点位/原始变量/"+experiment+"_tas"+time+"_monthly_005d.npy")
        ppt = np.load("H:/relatioship/点位/原始变量/"+experiment+"_pr"+time+"_monthly_005d.npy")
        result=koppen(ta, ppt)
        np.save("H:/relatioship/点位/情景变量/"+experiment+"_koppen_"+time+".npy",result)

experiment="ATMO+PHYS"
for time in ["1984_2014", "2070_2100"]:
    ta = np.load("H:/relatioship/点位/原始变量/" + experiment + "_tas" + time + "_monthly_005d.npy")
    ppt = np.load("H:/relatioship/点位/原始变量/" + experiment + "_pr" + time + "_monthly_005d.npy")
    result = koppen(ta, ppt)
    np.save("H:/relatioship/点位/情景变量/" + experiment + "_koppen_" + time + ".npy", result)

experiments1=["historical","esmControl"]
for experiment in experiments1:
    for time in ["1984_2014","2070_2100"]:
        data=np.load("H:/relatioship/点位/情景变量/"+experiment+"_koppen"+time+".npy")
        for i in range(100):
            Mask = np.full([3600, 7200], np.nan)
            Mask[(landsea == 1) & (np.isnan(data))] = -1000
            data2 = np.full([3600, 7200], np.nan)
            fill_raster_class(data, data2)
            data[Mask == -1000] = data2[Mask == -1000]
            print("soilclass" + " has proceesed " + str(i))
        np.save("H:/relatioship/点位/情景变量/" + experiment + "_koppenc" + time + ".npy", data)

data=np.load("H:/relatioship/点位/情景变量/historical_koppenc1984_2014.npy")





##############  生物多样性填充
MAM=readtif("H:/relatioship/BiodiversityMapping_TIFFs_2019_03d14/MAM_v.tif")
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
np.save("H:/relatioship/点位/result/MAM.npy",datavar)

AMP=readtif("H:/relatioship/BiodiversityMapping_TIFFs_2019_03d14/AMP_v.tif")
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
np.save("H:/relatioship/点位/result/AMP.npy",datavar)


Bird=readtif("H:/relatioship/BiodiversityMapping_TIFFs_2019_03d14/Birds_total_v.tif")
Bird[Bird>6000]=np.nan
Bird[np.isnan(Bird)&(landsea==1)]=0
np.save("H:/relatioship/点位/result/Bird.npy",Bird)

#########################################################################################################################
landsea = np.full([3600, 7200], -9999)
landsea[0:3000, :] = np.load("H:/relatioship/点位/landsea.npy")
landsea=landsea.astype(float)
land_MCD=np.load("H:/relatioship/Gimms输出/land_MCD_m.npy")
landsea[land_MCD==0]=-9999


MAM=np.load("H:/relatioship/点位/result/MAM.npy")
MAM[landsea==-9999]=np.nan
print(np.nanpercentile(MAM,99.9))
AMP=np.load("H:/relatioship/点位/result/AMP.npy")
AMP[landsea==-9999]=np.nan
print(np.nanpercentile(AMP,99.9))
Bird=np.load("H:/relatioship/点位/result/Bird.npy")
Bird[landsea==-9999]=np.nan
print(np.nanpercentile(Bird,99.9))