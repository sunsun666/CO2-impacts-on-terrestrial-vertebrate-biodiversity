from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

def lat_change(Map):
    calcu=np.full([150,4],np.nan)
    for i in range(150):
        calcu[i,0]=i*20+10
        d=Map[i*20:(i+1)*20,:]
        d=d[np.isnan(d)==False]
        if len(d)>3:
            calcu[i,1]=np.nanmean(d)
            ci = stats.t.interval(0.95, len(d)-1, loc=np.nanmean(d), scale=stats.sem(d))
            calcu[i,2]=ci[0]
            calcu[i, 3] = ci[1]
    return calcu

def long_change(Map):
    calcu=np.full([360,4],np.nan)
    for i in range(360):
        calcu[i,0]=i*20+10
        d=Map[:,i*20:(i+1)*20]
        d=d[np.isnan(d)==False]
        if len(d)>3:
            calcu[i,1]=np.nanmean(d)
            ci = stats.t.interval(0.95, len(d)-1, loc=np.nanmean(d), scale=stats.sem(d))
            calcu[i,2]=ci[0]
            calcu[i, 3] = ci[1]
    return calcu
MAM_historical_C=np.load("H:/基于关系的研究/点位/result/MAM_pred_historical_1984_2014.npy")
MAM_historical_C=MAM_historical_C/192
AMP_historical_C=np.load("H:/基于关系的研究/点位/result/AMP_pred_historical_1984_2014.npy")
AMP_historical_C=AMP_historical_C/113
Bird_historical_C=np.load("H:/基于关系的研究/点位/result/Bird_pred_historical_1984_2014.npy")
Bird_historical_C=Bird_historical_C/587

result_MAM=lat_change(MAM_historical_C)
result_AMP=lat_change(AMP_historical_C)
result_Bird=lat_change(Bird_historical_C)


M_historical_C=(MAM_historical_C+AMP_historical_C+Bird_historical_C)/3
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,figsize=(8,10/3))
plt.subplots_adjust(wspace=0, hspace=0,left=0, bottom=0, right = 1, top = 1)
plt.subplot(1,1,1)
plt.imshow(M_historical_C,vmin=0, vmax=1,cmap='gist_earth_r')
plt.axis('off')
ax1 = fig.add_axes([0, 0, 0.15, 1])
ax1.plot(result_MAM[:,1],3000-result_MAM[:,0],color="#E6AF0C")
ax1.plot(result_AMP[:,1],3000-result_AMP[:,0],color="#78A51E")
ax1.plot(result_Bird[:,1],3000-result_Bird[:,0],color="#E52287")
ax1.plot((result_Bird[:,1]+result_MAM[:,1]+result_AMP[:,1])/3,3000-result_Bird[:,0],color="#646464",linewidth=3)
ax1.patch.set_alpha(0.0)
plt.ylim(0,3000)
plt.axis('off')
plt.savefig("H:/基于关系的研究/点位/result/图/Figures_oct/FigS3.tiff",dpi=300)



##FigS4
MAM_esmFdbk2_C=np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFdbk2_1984_2014.npy")
MAM_esmFdbk2_C=MAM_esmFdbk2_C/192
AMP_esmFdbk2_C=np.load("H:/基于关系的研究/点位/result2/AMP_pred_esmFdbk2_1984_2014.npy")
AMP_esmFdbk2_C=AMP_esmFdbk2_C/113
Bird_esmFdbk2_C=np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFdbk2_1984_2014.npy")
Bird_esmFdbk2_C=Bird_esmFdbk2_C/587

result_esmFdbk2_MAM=lat_change(MAM_esmFdbk2_C)
result_esmFdbk2_AMP=lat_change(AMP_esmFdbk2_C)
result_esmFdbk2_Bird=lat_change(Bird_esmFdbk2_C)

MAM_esmFixClim2_C=np.load("H:/基于关系的研究/点位/result/MAM_pred_esmFixClim2_1984_2014.npy")
MAM_esmFixClim2_C=MAM_esmFixClim2_C/192
AMP_esmFixClim2_C=np.load("H:/基于关系的研究/点位/result/AMP_pred_esmFixClim2_1984_2014.npy")
AMP_esmFixClim2_C=AMP_esmFixClim2_C/113
Bird_esmFixClim2_C=np.load("H:/基于关系的研究/点位/result/Bird_pred_esmFixClim2_1984_2014.npy")
Bird_esmFixClim2_C=Bird_esmFixClim2_C/587
result_esmFixClim2_MAM=lat_change(MAM_esmFixClim2_C)
result_esmFixClim2_AMP=lat_change(AMP_esmFixClim2_C)
result_esmFixClim2_Bird=lat_change(Bird_esmFixClim2_C)

MAM_ATMO_PHYS_C=np.load("H:/基于关系的研究/点位/result/MAM_pred_ATMO+PHYS_1984_2014.npy")
MAM_ATMO_PHYS_C=MAM_ATMO_PHYS_C/192
AMP_ATMO_PHYS_C=np.load("H:/基于关系的研究/点位/result/AMP_pred_ATMO+PHYS_1984_2014.npy")
AMP_ATMO_PHYS_C=AMP_ATMO_PHYS_C/113
Bird_ATMO_PHYS_C=np.load("H:/基于关系的研究/点位/result/Bird_pred_ATMO+PHYS_1984_2014.npy")
Bird_ATMO_PHYS_C=Bird_ATMO_PHYS_C/587
result_ATMO_PHYS_MAM=lat_change(MAM_ATMO_PHYS_C)
result_ATMO_PHYS_AMP=lat_change(AMP_ATMO_PHYS_C)
result_ATMO_PHYS_Bird=lat_change(Bird_ATMO_PHYS_C)


M_esmFdbk2_C=(MAM_esmFdbk2_C+AMP_esmFdbk2_C+Bird_esmFdbk2_C)/3
M_esmFixClim2_C=(MAM_esmFixClim2_C+AMP_esmFixClim2_C+Bird_esmFixClim2_C)/3
M_ATMO_PHYS_C=(MAM_ATMO_PHYS_C+AMP_ATMO_PHYS_C+Bird_ATMO_PHYS_C)/3

x=[270,540,810,1080]
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True,figsize=(8,10))
plt.subplots_adjust(wspace=0, hspace=0,left=0, bottom=0, right = 1, top = 1)

plt.subplot(3,1,1)
plt.imshow(M_esmFdbk2_C,vmin=0, vmax=1,cmap='gist_earth_r')
plt.axis('off')
plt.vlines(x,0,50)
plt.axvline(x=270,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=540,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=810,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=1080,c='grey',ls='--',lw=1,alpha=0.7)

plt.subplot(3,1,2)
plt.imshow(M_esmFixClim2_C,vmin=0, vmax=1,cmap='gist_earth_r')
plt.axis('off')
plt.axvline(x=270,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=540,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=810,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=1080,c='grey',ls='--',lw=1,alpha=0.7)

plt.subplot(3,1,3)
plt.imshow(M_ATMO_PHYS_C,vmin=0, vmax=1,cmap='gist_earth_r')
plt.axis('off')
plt.vlines(x,2950,3000)
plt.axvline(x=270,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=540,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=810,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=1080,c='grey',ls='--',lw=1,alpha=0.7)
plt.xlim(0,7200)
plt.ylim(3000,0)

ax1 = fig.add_axes([0, 2/3, 0.15, 1/3])
ax1.plot(result_esmFdbk2_MAM[:,1],3000-result_esmFdbk2_MAM[:,0],color="#E6AF0C")
ax1.plot(result_esmFdbk2_AMP[:,1],3000-result_esmFdbk2_AMP[:,0],color="#78A51E")
ax1.plot(result_esmFdbk2_Bird[:,1],3000-result_esmFdbk2_Bird[:,0],color="#E52287")
ax1.plot((result_esmFdbk2_Bird[:,1]+result_esmFdbk2_MAM[:,1]+result_esmFdbk2_AMP[:,1])/3,3000-result_esmFdbk2_Bird[:,0],color="#646464",linewidth=3)
ax1.patch.set_alpha(0.0)
plt.xlim(0,0.8)
plt.axis('off')

ax2 = fig.add_axes([0, 1/3, 0.15, 1/3])
ax2.plot(result_esmFixClim2_MAM[:,1],3000-result_esmFixClim2_MAM[:,0],color="#E6AF0C")
ax2.plot(result_esmFixClim2_AMP[:,1],3000-result_esmFixClim2_AMP[:,0],color="#78A51E")
ax2.plot(result_esmFixClim2_Bird[:,1],3000-result_esmFixClim2_Bird[:,0],color="#E52287")
ax2.plot((result_esmFixClim2_Bird[:,1]+result_esmFixClim2_MAM[:,1]+result_esmFixClim2_AMP[:,1])/3,3000-result_esmFixClim2_Bird[:,0],color="#646464",linewidth=3)
ax2.patch.set_alpha(0.0)
plt.xlim(0,0.8)
plt.axis('off')

ax3 = fig.add_axes([0, 0, 0.15, 1/3])
ax3.plot(result_ATMO_PHYS_MAM[:,1],3000-result_ATMO_PHYS_MAM[:,0],color="#E6AF0C")
ax3.plot(result_ATMO_PHYS_AMP[:,1],3000-result_ATMO_PHYS_AMP[:,0],color="#78A51E")
ax3.plot(result_ATMO_PHYS_Bird[:,1],3000-result_ATMO_PHYS_Bird[:,0],color="#E52287")
ax3.plot((result_ATMO_PHYS_Bird[:,1]+result_ATMO_PHYS_MAM[:,1]+result_ATMO_PHYS_AMP[:,1])/3,3000-result_ATMO_PHYS_Bird[:,0],color="#646464",linewidth=3)
ax3.patch.set_alpha(0.0)
plt.ylim(0,3000)
plt.xlim(0,0.8)
plt.axis('off')

plt.savefig("H:/基于关系的研究/点位/result/图/Figures_oct/FigS4/FigS4.tiff",dpi=300)

####FigS5
MAM_esmFdbk2_C=np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFdbk2_1984_2014.npy")
MAM_esmFdbk2_F=np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFdbk2_2070_2100.npy")
MAM_esmFdbk2_delta=MAM_esmFdbk2_F-MAM_esmFdbk2_C
MAM_esmFdbk2_delta=MAM_esmFdbk2_delta*100/192
AMP_esmFdbk2_C=np.load("H:/基于关系的研究/点位/result2/AMP_pred_esmFdbk2_1984_2014.npy")
AMP_esmFdbk2_F=np.load("H:/基于关系的研究/点位/result2/AMP_pred_esmFdbk2_2070_2100.npy")
AMP_esmFdbk2_delta=AMP_esmFdbk2_F-AMP_esmFdbk2_C
AMP_esmFdbk2_delta=AMP_esmFdbk2_delta*100/113
Bird_esmFdbk2_C=np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFdbk2_1984_2014.npy")
Bird_esmFdbk2_F=np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFdbk2_2070_2100.npy")
Bird_esmFdbk2_delta=Bird_esmFdbk2_F-Bird_esmFdbk2_C
Bird_esmFdbk2_delta=Bird_esmFdbk2_delta*100/587

MAM_esmFixClim2_C=np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFixClim2_1984_2014.npy")
MAM_esmFixClim2_F=np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFixClim2_2070_2100.npy")
MAM_esmFixClim2_delta=MAM_esmFixClim2_F-MAM_esmFixClim2_C
MAM_esmFixClim2_delta=MAM_esmFixClim2_delta*100/192
AMP_esmFixClim2_C=np.load("H:/基于关系的研究/点位/result/AMP_pred_esmFixClim2_1984_2014.npy")
AMP_esmFixClim2_F=np.load("H:/基于关系的研究/点位/result/AMP_pred_esmFixClim2_2070_2100.npy")
AMP_esmFixClim2_delta=AMP_esmFixClim2_F-AMP_esmFixClim2_C
AMP_esmFixClim2_delta=AMP_esmFixClim2_delta*100/113
Bird_esmFixClim2_C=np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFixClim2_1984_2014.npy")
Bird_esmFixClim2_F=np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFixClim2_2070_2100.npy")
Bird_esmFixClim2_delta=Bird_esmFixClim2_F-Bird_esmFixClim2_C
Bird_esmFixClim2_delta=Bird_esmFixClim2_delta*100/587

MAM_ATMO_PHYS_C=np.load("H:/基于关系的研究/点位/result2/MAM_pred_ATMO+PHYS_1984_2014.npy")
MAM_ATMO_PHYS_F=np.load("H:/基于关系的研究/点位/result2/MAM_pred_ATMO+PHYS_2070_2100.npy")
MAM_ATMO_PHYS_delta=MAM_ATMO_PHYS_F-MAM_ATMO_PHYS_C
MAM_ATMO_PHYS_delta=MAM_ATMO_PHYS_delta*100/192
AMP_ATMO_PHYS_C=np.load("H:/基于关系的研究/点位/result2/AMP_pred_ATMO+PHYS_1984_2014.npy")
AMP_ATMO_PHYS_F=np.load("H:/基于关系的研究/点位/result2/AMP_pred_ATMO+PHYS_2070_2100.npy")
AMP_ATMO_PHYS_delta=AMP_ATMO_PHYS_F-AMP_ATMO_PHYS_C
AMP_ATMO_PHYS_delta=AMP_ATMO_PHYS_delta*100/113
Bird_ATMO_PHYS_C=np.load("H:/基于关系的研究/点位/result2/Bird_pred_ATMO+PHYS_1984_2014.npy")
Bird_ATMO_PHYS_F=np.load("H:/基于关系的研究/点位/result2/Bird_pred_ATMO+PHYS_2070_2100.npy")
Bird_ATMO_PHYS_delta=Bird_ATMO_PHYS_F-Bird_ATMO_PHYS_C
Bird_ATMO_PHYS_delta=Bird_ATMO_PHYS_delta*100/587
del MAM_esmFdbk2_C,MAM_esmFdbk2_F,AMP_esmFdbk2_C,AMP_esmFdbk2_F,Bird_esmFdbk2_C,Bird_esmFdbk2_F,\
    MAM_esmFixClim2_C,MAM_esmFixClim2_F,AMP_esmFixClim2_C,AMP_esmFixClim2_F,Bird_esmFixClim2_C,Bird_esmFixClim2_F,\
    MAM_ATMO_PHYS_C,MAM_ATMO_PHYS_F,AMP_ATMO_PHYS_C,AMP_ATMO_PHYS_F,Bird_ATMO_PHYS_C,Bird_ATMO_PHYS_F





result_esmFdbk2_MAM=lat_change(MAM_esmFdbk2_delta)
result_esmFdbk2_AMP=lat_change(AMP_esmFdbk2_delta)
result_esmFdbk2_Bird=lat_change(Bird_esmFdbk2_delta)

result_esmFixClim2_MAM=lat_change(MAM_esmFixClim2_delta)
result_esmFixClim2_AMP=lat_change(AMP_esmFixClim2_delta)
result_esmFixClim2_Bird=lat_change(Bird_esmFixClim2_delta)

result_ATMO_PHYS_MAM=lat_change(MAM_ATMO_PHYS_delta)
result_ATMO_PHYS_AMP=lat_change(AMP_ATMO_PHYS_delta)
result_ATMO_PHYS_Bird=lat_change(Bird_ATMO_PHYS_delta)

M_esmFdbk2_delta=(MAM_esmFdbk2_delta+AMP_esmFdbk2_delta+Bird_esmFdbk2_delta)/3
M_esmFixClim2_delta=(MAM_esmFixClim2_delta+AMP_esmFixClim2_delta+Bird_esmFixClim2_delta)/3
M_ATMO_PHYS_delta=(MAM_ATMO_PHYS_delta+AMP_ATMO_PHYS_delta+Bird_ATMO_PHYS_delta)/3



x=[270+54,540+54,810+54]
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True,figsize=(8,10))
plt.subplots_adjust(wspace=0, hspace=0,left=0, bottom=0, right = 1, top = 1)

plt.subplot(3,1,1)
plt.imshow(M_esmFdbk2_delta,vmin=-10, vmax=10,cmap='PiYG')
plt.axis('off')
plt.vlines(x,0,50)
plt.axvline(x=270+54,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=540+54,c='grey',ls='-',lw=1,alpha=0.8)
plt.axvline(x=810+54,c='grey',ls='--',lw=1,alpha=0.7)


plt.subplot(3,1,2)
plt.imshow(M_esmFixClim2_delta,vmin=-10, vmax=10,cmap='PiYG')
plt.axis('off')
plt.axvline(x=270+54,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=540+54,c='grey',ls='-',lw=1,alpha=0.8)
plt.axvline(x=810+54,c='grey',ls='--',lw=1,alpha=0.7)


plt.subplot(3,1,3)
plt.imshow(M_ATMO_PHYS_delta,vmin=-10, vmax=10,cmap='PiYG')
plt.axis('off')
plt.axvline(x=270+54,c='grey',ls='--',lw=1,alpha=0.7)
plt.axvline(x=540+54,c='grey',ls='-',lw=1,alpha=0.8)
plt.axvline(x=810+54,c='grey',ls='--',lw=1,alpha=0.7)





ax1 = fig.add_axes([0, 2/3, 0.2625, 1/3])
ax1.plot(result_esmFdbk2_MAM[:,1],3000-result_esmFdbk2_MAM[:,0],color="#E6AF0C",alpha=0.8)
ax1.plot(result_esmFdbk2_AMP[:,1],3000-result_esmFdbk2_AMP[:,0],color="#78A51E",alpha=0.8)
ax1.plot(result_esmFdbk2_Bird[:,1],3000-result_esmFdbk2_Bird[:,0],color="#E52287",alpha=0.8)
ax1.plot((result_esmFdbk2_Bird[:,1]+result_esmFdbk2_MAM[:,1]+result_esmFdbk2_AMP[:,1])/3,3000-result_esmFdbk2_Bird[:,0],color="#646464",linewidth=3,alpha=0.8)
ax1.patch.set_alpha(0.0)
plt.xlim(-5.5,12)
plt.axis('off')

ax2 = fig.add_axes([0, 1/3,  0.2625, 1/3])
ax2.plot(result_esmFixClim2_MAM[:,1],3000-result_esmFixClim2_MAM[:,0],color="#E6AF0C",alpha=0.8)
ax2.plot(result_esmFixClim2_AMP[:,1],3000-result_esmFixClim2_AMP[:,0],color="#78A51E",alpha=0.8)
ax2.plot(result_esmFixClim2_Bird[:,1],3000-result_esmFixClim2_Bird[:,0],color="#E52287",alpha=0.8)
ax2.plot((result_esmFixClim2_Bird[:,1]+result_esmFixClim2_MAM[:,1]+result_esmFixClim2_AMP[:,1])/3,3000-result_esmFixClim2_Bird[:,0],color="#646464",linewidth=3,alpha=0.8)
ax2.patch.set_alpha(0.0)
plt.xlim(-5.5,12)
plt.axis('off')

ax3 = fig.add_axes([0, 0,  0.2625, 1/3])
ax3.plot(result_ATMO_PHYS_MAM[:,1],3000-result_ATMO_PHYS_MAM[:,0],color="#E6AF0C",alpha=0.8)
ax3.plot(result_ATMO_PHYS_AMP[:,1],3000-result_ATMO_PHYS_AMP[:,0],color="#78A51E",alpha=0.8)
ax3.plot(result_ATMO_PHYS_Bird[:,1],3000-result_ATMO_PHYS_Bird[:,0],color="#E52287",alpha=0.8)
ax3.plot((result_ATMO_PHYS_Bird[:,1]+result_ATMO_PHYS_MAM[:,1]+result_ATMO_PHYS_AMP[:,1])/3,3000-result_ATMO_PHYS_Bird[:,0],color="#646464",linewidth=3,alpha=0.8)
ax3.patch.set_alpha(0.0)
plt.ylim(0,3000)
plt.xlim(-5.5,12)
plt.axis('off')

plt.savefig("H:/基于关系的研究/点位/result/图/Figures_oct/FigS5/FigS5.tiff",dpi=300)

#FigS5fu
hotspots=pd.read_csv("H:/基于关系的研究/点位/result/图/Figures_oct/FigS6/hotspots_result.csv")
hotspots2=hotspots.copy()
order=[25,6,0,5,29,1,19,16,35,30,
       3,9,18,2,27,17,7,34,11,13,
       4,15,12,10,20,32,26,21,28,24,
       31,14,33,8,22,23]
for i in range(36):
    hotspots2.iloc[i]=hotspots.iloc[order[i]]




fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,figsize=(10,8))
plt.subplots_adjust(wspace=0, hspace=0,left=0, bottom=0, right = 1, top = 3000/5760)
plt.imshow(delta)


plt.figure(figsize=(12,2.5))
plt.axhline(0,c='grey',ls='-',lw=1.5,alpha=0.85)
for i in range(36):
    plt.axvline(i,c='grey',ls='--',lw=1,alpha=0.7)
plt.scatter(hotspots.ID,hotspots.MAM_esmFdbk2,color="#E6AF0C",alpha=0.8,s=90)
plt.scatter(hotspots.ID,hotspots.AMP_esmFdbk2,color="#78A51E",alpha=0.8,s=90)
plt.scatter(hotspots.ID,hotspots.Bird_esmFdbk2,color="#E52287",alpha=0.8,s=90)
plt.scatter(hotspots.ID,(hotspots.MAM_esmFdbk2+hotspots.AMP_esmFdbk2+hotspots.Bird_esmFdbk2)/3,color="black",alpha=0.8,s=50)
plt.xlim(-1,36)
plt.ylim(-4.5,4.5)
plt.axis('off')





plt.figure(figsize=(12,2.5))
plt.axhline(0,c='grey',ls='-',lw=1.5,alpha=0.85)
for i in range(36):
    plt.axvline(i,c='grey',ls='--',lw=1,alpha=0.7)
plt.scatter(hotspots.ID,hotspots.MAM_esmFdbk2,color="#E6AF0C",alpha=0.8,s=90)
plt.scatter(hotspots.ID,hotspots.AMP_esmFdbk2,color="#78A51E",alpha=0.8,s=90)
plt.scatter(hotspots.ID,hotspots.Bird_esmFdbk2,color="#E52287",alpha=0.8,s=90)
plt.scatter(hotspots.ID,(hotspots.MAM_esmFdbk2+hotspots.AMP_esmFdbk2+hotspots.Bird_esmFdbk2)/3,color="black",alpha=0.8,s=50)
plt.xlim(-1,36)
plt.ylim(-4.5,4.5)
plt.axis('off')




for i in range(36):
    x=7200*(i+1)/37
    y=200
    plt.text(x,y,str(i+1))



ba=7200/37
h_lines1=[1400,2600,2360,2153,2000,1000,1492,1300,1130,1800,
          1410,1647,1150,2474,2368,2281,2165,1792,1332,1008,
          882,2082,1395,1178,965,1424,2455,1163,1797,1584,
          1836,1124,2397,1903,2218,2674]
h_lines2=[]
for i in range(36):
    h_lines2.append((i+1)*ba)
h_lines3=[ba,2150,2580,2530,2075,6*ba,1784,8*ba,9*ba,2000,
          11*ba,3386,3444,3996,3938,4233,4305,4262,4397,4441,
          4388,4572,5520,5176,5022,26*ba,5936,5573,29*ba,5980,
          31*ba,32*ba,6643,34*ba,6899,36*ba]




hotmap_ATMO_PHYS=readtif("H:/SA Fig1/hotspot_ATMO+PHYS.tif")
hotmap_ATMO_PHYS[hotmap_ATMO_PHYS<-1000]=np.nan
hotmap_ATMO_PHYS[np.where((hotmap_ATMO_PHYS>0)&(hotmap_ATMO_PHYS<1.5))]=1
hotmap_ATMO_PHYS[np.where((hotmap_ATMO_PHYS>-1.5)&(hotmap_ATMO_PHYS<0))]=-1
hotmap_ATMO_PHYS[np.where((hotmap_ATMO_PHYS>1.5)&(hotmap_ATMO_PHYS<3))]=1.5
hotmap_ATMO_PHYS[np.where((hotmap_ATMO_PHYS>-3)&(hotmap_ATMO_PHYS<-1.5))]=-1.5
hotmap_ATMO_PHYS[hotmap_ATMO_PHYS>3]=2
hotmap_ATMO_PHYS[hotmap_ATMO_PHYS<-3]=-2

hotmap_ATMO=readtif("H:/SA Fig1/hotspot_ATMO.tif")
hotmap_ATMO[hotmap_ATMO<-1000]=np.nan
hotmap_ATMO[np.where((hotmap_ATMO>0)&(hotmap_ATMO<1.5))]=1
hotmap_ATMO[np.where((hotmap_ATMO>-1.5)&(hotmap_ATMO<0))]=-1
hotmap_ATMO[np.where((hotmap_ATMO>1.5)&(hotmap_ATMO<3))]=1.5
hotmap_ATMO[np.where((hotmap_ATMO>-3)&(hotmap_ATMO<-1.5))]=-1.5
hotmap_ATMO[hotmap_ATMO>3]=2
hotmap_ATMO[hotmap_ATMO<-3]=-2

hotmap_PHYS=readtif("H:/SA Fig1/hotspot_PHYS.tif")
hotmap_PHYS[hotmap_PHYS<-1000]=np.nan
hotmap_PHYS[np.where((hotmap_PHYS>0)&(hotmap_PHYS<1.5))]=1
hotmap_PHYS[np.where((hotmap_PHYS>-1.5)&(hotmap_PHYS<0))]=-1
hotmap_PHYS[np.where((hotmap_PHYS>1.5)&(hotmap_PHYS<3))]=1.5
hotmap_PHYS[np.where((hotmap_PHYS>-3)&(hotmap_PHYS<-1.5))]=-1.5
hotmap_PHYS[hotmap_PHYS>3]=2
hotmap_PHYS[hotmap_PHYS<-3]=-2




landsea=np.load("H:/基于关系的研究/点位/landsea.npy")
landsea[landsea==-9999]=np.nan


fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,figsize=(10,3398/720))
plt.subplots_adjust(wspace=0, hspace=0,left=0, bottom=0, right = 1, top = 3000/3398)
plt.imshow(landsea,alpha=0.075)
plt.imshow(hotmap_ATMO_PHYS,vmin=-2.5,vmax=2.5,cmap="PiYG")
plt.hlines(h_lines1,h_lines2,h_lines3,linestyle="--",colors="black",linewidth=1,alpha=0.1)
plt.vlines(h_lines2,750,h_lines1,linestyle="--",colors="black",linewidth=1,alpha=0.1)
plt.axis('off')

ax1 = fig.add_axes([0, 2300/3398, 1, 1094/3398])
ax1.scatter(hotspots2.index,hotspots2.MAM_ATMO_PHYS,color="#E6AF0C",alpha=0.6,s=90)
ax1.scatter(hotspots2.index,hotspots2.AMP_ATMO_PHYS,color="#78A51E",alpha=0.6,s=90)
ax1.scatter(hotspots2.index,hotspots2.Bird_ATMO_PHYS,color="#E52287",alpha=0.6,s=90)
ax1.scatter(hotspots2.index,(hotspots2.MAM_ATMO_PHYS+hotspots2.AMP_ATMO_PHYS+hotspots2.Bird_ATMO_PHYS)/3,color="black",alpha=0.7,s=50)
plt.axhline(0,c='grey',ls='-',lw=1.5,alpha=0.85)
for i in range(36):
    plt.axvline(i,c='grey',ls='--',lw=1,alpha=0.7)
plt.xlim(-1,36)
plt.ylim(-5,8)
plt.axis('off')
plt.yticks([-3,0,3,6])
ax1.patch.set_alpha(0.0)
plt.savefig("H:/基于关系的研究/点位/result/图/Figures_oct/FigS5/FigS5fu_ATMO_PHYS.tiff",dpi=300)


fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,figsize=(10,3398/720))
plt.subplots_adjust(wspace=0, hspace=0,left=0, bottom=0, right = 1, top = 3000/3398)
plt.imshow(landsea,alpha=0.075)
plt.imshow(hotmap_ATMO,vmin=-2.5,vmax=2.5,cmap="PiYG")
plt.hlines(h_lines1,h_lines2,h_lines3,linestyle="--",colors="black",linewidth=1,alpha=0.1)
plt.vlines(h_lines2,750,h_lines1,linestyle="--",colors="black",linewidth=1,alpha=0.1)
plt.axis('off')

ax1 = fig.add_axes([0, 2300/3398, 1, 1094/3398])
ax1.scatter(hotspots2.index,hotspots2.MAM_esmFdbk2,color="#E6AF0C",alpha=0.6,s=90)
ax1.scatter(hotspots2.index,hotspots2.AMP_esmFdbk2,color="#78A51E",alpha=0.6,s=90)
ax1.scatter(hotspots2.index,hotspots2.Bird_esmFdbk2,color="#E52287",alpha=0.6,s=90)
ax1.scatter(hotspots2.index,(hotspots2.MAM_esmFdbk2+hotspots2.AMP_esmFdbk2+hotspots2.Bird_esmFdbk2)/3,color="black",alpha=0.7,s=50)
plt.axhline(0,c='grey',ls='-',lw=1.5,alpha=0.85)
for i in range(36):
    plt.axvline(i,c='grey',ls='--',lw=1,alpha=0.7)
plt.xlim(-1,36)
plt.ylim(-5,8)
plt.axis('off')
ax1.patch.set_alpha(0.0)
plt.savefig("H:/基于关系的研究/点位/result/图/Figures_oct/FigS5/FigS5fu_ATMO.tiff",dpi=300)



fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,figsize=(10,3398/720))
plt.subplots_adjust(wspace=0, hspace=0,left=0, bottom=0, right = 1, top = 3000/3398)
plt.imshow(landsea,alpha=0.075)
plt.imshow(hotmap_PHYS,vmin=-2.5,vmax=2.5,cmap="PiYG")
plt.hlines(h_lines1,h_lines2,h_lines3,linestyle="--",colors="black",linewidth=1,alpha=0.1)
plt.vlines(h_lines2,750,h_lines1,linestyle="--",colors="black",linewidth=1,alpha=0.1)
plt.axis('off')

ax1 = fig.add_axes([0, 2300/3398, 1, 1094/3398])
ax1.scatter(hotspots2.index,hotspots2.MAM_esmFixClim2,color="#E6AF0C",alpha=0.6,s=90)
ax1.scatter(hotspots2.index,hotspots2.AMP_esmFixClim2,color="#78A51E",alpha=0.6,s=90)
ax1.scatter(hotspots2.index,hotspots2.Bird_esmFixClim2,color="#E52287",alpha=0.6,s=90)
ax1.scatter(hotspots2.index,(hotspots2.MAM_esmFixClim2+hotspots2.AMP_esmFixClim2+hotspots2.Bird_esmFixClim2)/3,color="black",alpha=0.7,s=50)
plt.axhline(0,c='grey',ls='-',lw=1.5,alpha=0.85)
for i in range(36):
    plt.axvline(i,c='grey',ls='--',lw=1,alpha=0.7)
plt.xlim(-1,36)
plt.ylim(-5,8)
plt.axis('off')
ax1.patch.set_alpha(0.0)
plt.savefig("H:/基于关系的研究/点位/result/图/Figures_oct/FigS5/FigS5fu_PHYS.tiff",dpi=300)



#######################################          速度           ############################################
## 物种迁移速率
for experiment in experiments:
    for sp in taxas:
        data_C = np.load("H:/基于关系的研究/点位/result2/"+sp+"_pred_"+experiment+"_1984_2014.npy")

        data_F = np.load("H:/基于关系的研究/点位/result2/" + sp + "_pred_" + experiment + "_2070_2100.npy")

        x_direction, y_direction, rises = slope_aspect(data_C )
        delta_data = data_F-data_C
        migrate=np.full([4,3000,7200],np.nan)
        z=abs(delta_data/rises)

        x_real=z*x_direction/np.sqrt(x_direction**2+y_direction**2)
        y_real = z * y_direction / np.sqrt(x_direction ** 2 + y_direction ** 2)

        migrate[0] = x_real
        migrate[1] = y_real
        migrate[2] = z
        migrate[3]= delta_data
        np.save("H:/基于关系的研究/点位/result2/"+sp+"_pred_"+experiment+"_migrate.npy",migrate)

for experiment in experiments:
    for sp in taxas:
        data_C = np.load("H:/基于关系的研究/点位/result2/"+sp+"_pred_"+experiment+"_1984_2014.npy")
        data_C,data_num = resample25(data_C)
        for rep in range(30):
            data_C_fill = fill_data(data_C)
            data_C[np.isnan(data_C)]=data_C_fill[np.isnan(data_C)]
        x_direction,y_direction,rises=slope_aspect(data_C)
        data_F = np.load("H:/基于关系的研究/点位/result2/" + sp + "_pred_" + experiment + "_2070_2100.npy")
        data_F,data_num2 = resample25(data_F)
        delta_data = data_F-data_C
        delta_data[data_num <=500]=np.nan
        migrate=np.full([4,60,144],np.nan)
        z=abs(delta_data/rises)

        x_real=z*x_direction/np.sqrt(x_direction**2+y_direction**2)
        y_real = z * y_direction / np.sqrt(x_direction ** 2 + y_direction ** 2)

        migrate[0] = x_real
        migrate[1] = y_real
        migrate[2] = z
        migrate[3]= delta_data
        np.save("H:/基于关系的研究/点位/result2/"+sp+"_pred_"+experiment+"_migrate2.npy",migrate)










fig, ax = plt.subplots(3, 1, sharex=True, sharey=True,figsize=(7.2,9))
plt.subplots_adjust(wspace=0, hspace=0,left=0, bottom=0, right = 1, top = 1)



##图a velocity ATMO情景
plt.subplot(3,1,1)
MAM_mitigrate = np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFdbk2_migrate.npy")
MAM_mitigrate2 = np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFdbk2_migrate2.npy")
AMP_mitigrate=np.load("H:/基于关系的研究/点位/result2/AMP_pred_esmFdbk2_migrate.npy")
AMP_mitigrate2=np.load("H:/基于关系的研究/点位/result2/AMP_pred_esmFdbk2_migrate2.npy")
Bird_mitigrate=np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFdbk2_migrate.npy")
Bird_mitigrate2=np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFdbk2_migrate2.npy")

All_mitigrate=(MAM_mitigrate+AMP_mitigrate+Bird_mitigrate)/3
All_mitigrate2=(MAM_mitigrate2+AMP_mitigrate2+Bird_mitigrate2)/3
change_esmFdbk2_5km=(MAM_mitigrate[3]*100/192+AMP_mitigrate[3]*100/113+Bird_mitigrate[3]*100/587)/3
change_esmFdbk2_250km=(MAM_mitigrate2[3]*100/192+AMP_mitigrate2[3]*100/113+Bird_mitigrate2[3]*100/587)/3

velocity=np.sqrt(All_mitigrate[2]*100/86)
velocity[change_esmFdbk2_5km<0]=velocity[change_esmFdbk2_5km<0]*(-1)
plt.imshow(velocity,vmax=17.32,vmin=-17.32,cmap="BrBG")
plt.axis('off')

x_real=All_mitigrate2[0]/(111.325*0.05)
y_real=All_mitigrate2[1]/(111.325*0.05)
x_real=x_real*100/86
y_real=y_real*100/86

z=All_mitigrate2[2]
z_max=np.nanpercentile(z,99.5)
z[z>=z_max]=np.nan
zz=change_esmFdbk2_250km
x_plot=x_real/np.sqrt(z)
y_plot=y_real/np.sqrt(z)
for i in range(60):
    for j in range(144):
        if np.isnan(z[i,j])==False:
            if zz[i,j]>0:
                plt.quiver(j*50+25, i*50+25,x_plot[i,j]*50,y_plot[i,j]*50,
                           color='black', width=0.002, scale_units='xy', scale=1,alpha=0.65)
            else:
                plt.quiver(j*50+25, i*50+25,x_plot[i,j]*50,y_plot[i,j]*50,
                           color='blueviolet', width=0.002, scale_units='xy', scale=1,alpha=0.8)

##图b velocity PHYS情景
plt.subplot(3,1,2)
MAM_mitigrate = np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFixClim2_migrate.npy")
MAM_mitigrate2 = np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFixClim2_migrate2.npy")
AMP_mitigrate=np.load("H:/基于关系的研究/点位/result2/AMP_pred_esmFixClim2_migrate.npy")
AMP_mitigrate2=np.load("H:/基于关系的研究/点位/result2/AMP_pred_esmFixClim2_migrate2.npy")
Bird_mitigrate=np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFixClim2_migrate.npy")
Bird_mitigrate2=np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFixClim2_migrate2.npy")

All_mitigrate=(MAM_mitigrate+AMP_mitigrate+Bird_mitigrate)/3
All_mitigrate2=(MAM_mitigrate2+AMP_mitigrate2+Bird_mitigrate2)/3
change_esmFixClim2_5km=(MAM_mitigrate[3]*100/192+AMP_mitigrate[3]*100/113+Bird_mitigrate[3]*100/587)/3
change_esmFixClim2_250km=(MAM_mitigrate2[3]*100/192+AMP_mitigrate2[3]*100/113+Bird_mitigrate2[3]*100/587)/3

velocity=np.sqrt(All_mitigrate[2]*100/86)
velocity[change_esmFixClim2_5km<0]=velocity[change_esmFixClim2_5km<0]*(-1)
plt.imshow(velocity,vmax=17.32,vmin=-17.32,cmap="BrBG")
plt.axis('off')

x_real=All_mitigrate2[0]/(111.325*0.05)
y_real=All_mitigrate2[1]/(111.325*0.05)
x_real=x_real*100/86
y_real=y_real*100/86

z=All_mitigrate2[2]
z_max=np.nanpercentile(z,99.5)
z[z>=z_max]=np.nan
zz=change_esmFixClim2_250km
x_plot=x_real/np.sqrt(z)
y_plot=y_real/np.sqrt(z)
for i in range(60):
    for j in range(144):
        if np.isnan(z[i,j])==False:
            if zz[i,j]>0:
                plt.quiver(j*50+25, i*50+25,x_plot[i,j]*50,y_plot[i,j]*50,
                           color='black', width=0.002, scale_units='xy', scale=1,alpha=0.65)
            else:
                plt.quiver(j*50+25, i*50+25,x_plot[i,j]*50,y_plot[i,j]*50,
                           color='blueviolet', width=0.002, scale_units='xy', scale=1,alpha=0.8)

##图c velocity ATMO+PHYS情景
plt.subplot(3,1,3)
MAM_mitigrate = np.load("H:/基于关系的研究/点位/result2/MAM_pred_ATMO+PHYS_migrate.npy")
MAM_mitigrate2 = np.load("H:/基于关系的研究/点位/result2/MAM_pred_ATMO+PHYS_migrate2.npy")
AMP_mitigrate=np.load("H:/基于关系的研究/点位/result2/AMP_pred_ATMO+PHYS_migrate.npy")
AMP_mitigrate2=np.load("H:/基于关系的研究/点位/result2/AMP_pred_ATMO+PHYS_migrate2.npy")
Bird_mitigrate=np.load("H:/基于关系的研究/点位/result2/Bird_pred_ATMO+PHYS_migrate.npy")
Bird_mitigrate2=np.load("H:/基于关系的研究/点位/result2/Bird_pred_ATMO+PHYS_migrate2.npy")

All_mitigrate=(MAM_mitigrate+AMP_mitigrate+Bird_mitigrate)/3
All_mitigrate2=(MAM_mitigrate2+AMP_mitigrate2+Bird_mitigrate2)/3
change_ATMO_PHYS_5km=(MAM_mitigrate[3]*100/192+AMP_mitigrate[3]*100/113+Bird_mitigrate[3]*100/587)/3
change_ATMO_PHYS_250km=(MAM_mitigrate2[3]*100/192+AMP_mitigrate2[3]*100/113+Bird_mitigrate2[3]*100/587)/3

velocity=np.sqrt(All_mitigrate[2]*100/86)
velocity[change_ATMO_PHYS_5km<0]=velocity[change_ATMO_PHYS_5km<0]*(-1)
plt.imshow(velocity,vmax=17.32,vmin=-17.32,cmap="BrBG")
plt.axis('off')

x_real=All_mitigrate2[0]/(111.325*0.05)
y_real=All_mitigrate2[1]/(111.325*0.05)
x_real=x_real*100/86
y_real=y_real*100/86

z=All_mitigrate2[2]
z_max=np.nanpercentile(z,99.5)
z[z>=z_max]=np.nan
zz=change_ATMO_PHYS_250km
x_plot=x_real/np.sqrt(z)
y_plot=y_real/np.sqrt(z)
for i in range(60):
    for j in range(144):
        if np.isnan(z[i,j])==False:
            if zz[i,j]>0:
                plt.quiver(j*50+25, i*50+25,x_plot[i,j]*50,y_plot[i,j]*50,
                           color='black', width=0.002, scale_units='xy', scale=1,alpha=0.65)
            else:
                plt.quiver(j*50+25, i*50+25,x_plot[i,j]*50,y_plot[i,j]*50,
                           color='blueviolet', width=0.002, scale_units='xy', scale=1, alpha=0.8)
plt.axhline(2650,xmin=4200/7200, xmax=5700/7200,linestyle='--',color = 'r',linewidth = 0.5)

b2_raw=[0,0.1,0.5,1,5,10,25,50,100,200,300]
b2=np.sqrt([0,0.1,0.5,1,5,10,25,50,100,200,300])
for i in range(11):
    plt.quiver(i * 150 + 4200, 2650, 0, np.sqrt(b2[i])*50,
               color='black', width=0.002, scale_units='xy', scale=1, alpha=0.65)
    plt.text(i * 150 + 4150, 2550-np.sqrt(b2[i])*50,str(b2_raw[i]),rotation=90,fontsize=10)
for i in range(11):
    plt.quiver(i * 150 + 4200, 2650, 0, -np.sqrt(b2[i])*50,
               color='blueviolet', width=0.002, scale_units='xy', scale=1, alpha=0.65)


#velocity图的曲线图
ax1 = fig.add_axes([0, 2/3, 0.15, 1/3])
ax1.plot(table2["esmFdbk2_MAM"]/86,3000-table2["lat"],color="#E6AF0C",alpha=0.8)
ax1.plot(table2["esmFdbk2_AMP"]/86,3000-table2["lat"],color="#78A51E",alpha=0.8)
ax1.plot(table2["esmFdbk2_Bird"]/86,3000-table2["lat"],color="#E52287",alpha=0.8)
ax1.plot(table2["esmFdbk2"]/86,3000-table2["lat"],color="#646464",linewidth=3,alpha=0.8)
ax1.axvline(x=-0.5,c='grey',ls='--',lw=1,alpha=0.7)
ax1.axvline(x=0,c='grey',ls='-',lw=1,alpha=0.7)
ax1.axvline(x=0.5,c='grey',ls='--',lw=1,alpha=0.7)
ax1.patch.set_alpha(0.0)
plt.xlim(-1,1)
plt.ylim(0,3000)
plt.axis('off')
ax2 = fig.add_axes([0, 1/3, 0.15, 1/3])
ax2.plot(table2["esmFixClim2_MAM"]/86,3000-table2["lat"],color="#E6AF0C",alpha=0.8)
ax2.plot(table2["esmFixClim2_AMP"]/86,3000-table2["lat"],color="#78A51E",alpha=0.8)
ax2.plot(table2["esmFixClim2_Bird"]/86,3000-table2["lat"],color="#E52287",alpha=0.8)
ax2.plot(table2["esmFixClim2"]/86,3000-table2["lat"],color="#646464",linewidth=3,alpha=0.8)
ax2.axvline(x=-0.5,c='grey',ls='--',lw=1,alpha=0.7)
ax2.axvline(x=0,c='grey',ls='-',lw=1,alpha=0.7)
ax2.axvline(x=0.5,c='grey',ls='--',lw=1,alpha=0.7)
ax2.patch.set_alpha(0.0)
plt.xlim(-1,1)
plt.ylim(0,3000)
plt.axis('off')
ax3 = fig.add_axes([0, 0, 0.15, 1/3])
ax3.plot(table2["ATMO+PHYS_MAM"]/86,3000-table2["lat"],color="#E6AF0C",alpha=0.8)
ax3.plot(table2["ATMO+PHYS_AMP"]/86,3000-table2["lat"],color="#78A51E",alpha=0.8)
ax3.plot(table2["ATMO+PHYS_Bird"]/86,3000-table2["lat"],color="#E52287",alpha=0.8)
ax3.plot(table2["ATMO+PHYS"]/86,3000-table2["lat"],color="#646464",linewidth=3,alpha=0.8)
ax3.axvline(x=-0.5,c='grey',ls='--',lw=1,alpha=0.7)
ax3.axvline(x=0,c='grey',ls='-',lw=1,alpha=0.7)
ax3.axvline(x=0.5,c='grey',ls='--',lw=1,alpha=0.7)
ax3.patch.set_alpha(0.0)
plt.xlim(-1,1)
plt.ylim(0,3000)
plt.axis('off')
plt.savefig("H:/基于关系的研究/点位/result/图/velocity/velocity.tiff",dpi=500)

#velocity图制表
MAM_mitigrate2 = np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFdbk2_migrate2.npy")
AMP_mitigrate2=np.load("H:/基于关系的研究/点位/result2/AMP_pred_esmFdbk2_migrate2.npy")
Bird_mitigrate2=np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFdbk2_migrate2.npy")

All_mitigrate2=(MAM_mitigrate2+AMP_mitigrate2+Bird_mitigrate2)/3
change_esmFdbk2_250km=(MAM_mitigrate2[3]*100/192+AMP_mitigrate2[3]*100/113+Bird_mitigrate2[3]*100/587)/3
All_mitigrate2[3]=change_esmFdbk2_250km
np.save("H:/基于关系的研究/点位/result2/all_pred_esmFdbk2_migrate2.npy")


table=pd.DataFrame({"lat":np.full([60],np.nan),
                    "esmFdbk2_MAM_po":np.full([60],np.nan),
                    "esmFdbk2_AMP_po":np.full([60],np.nan),
                    "esmFdbk2_Bird_po":np.full([60],np.nan),
                    "esmFixClim2_MAM_po":np.full([60],np.nan),
                    "esmFixClim2_AMP_po":np.full([60],np.nan),
                    "esmFixClim2_Bird_po":np.full([60],np.nan),
                    "ATMO+PHYS_MAM_po":np.full([60],np.nan),
                    "ATMO+PHYS_AMP_po":np.full([60],np.nan),
                    "ATMO+PHYS_Bird_po":np.full([60],np.nan),

                    "esmFdbk2_MAM_ne": np.full([60], np.nan),
                    "esmFdbk2_AMP_ne": np.full([60], np.nan),
                    "esmFdbk2_Bird_ne": np.full([60], np.nan),
                    "esmFixClim2_MAM_ne": np.full([60], np.nan),
                    "esmFixClim2_AMP_ne": np.full([60], np.nan),
                    "esmFixClim2_Bird_ne": np.full([60], np.nan),
                    "ATMO+PHYS_MAM_ne": np.full([60], np.nan),
                    "ATMO+PHYS_AMP_ne": np.full([60], np.nan),
                    "ATMO+PHYS_Bird_ne": np.full([60], np.nan),
                    "esmFdbk2_po":np.full([60], np.nan),
                    "esmFdbk2_ne":np.full([60], np.nan),
                    "esmFixClim2_po": np.full([60], np.nan),
                    "esmFixClim2_ne": np.full([60], np.nan),
                    "ATMO+PHYS_po": np.full([60], np.nan),
                    "ATMO+PHYS_ne": np.full([60], np.nan),
                    })

taxas=["MAM","AMP","Bird"]
experiments=["esmFdbk2","esmFixClim2","ATMO+PHYS"]
for experiment in experiments:
    for sp in taxas:
        data=np.load("H:/基于关系的研究/点位/result2/"+sp+"_pred_"+experiment+"_migrate2.npy")
        var=experiment+"_"+sp
        data_po=data[2].copy()
        data_po[data[3]<0]=np.nan
        po_ts=np.nanmean(data_po,axis=1)
        table[experiment+"_"+sp+"_po"]=po_ts
        data_ne=data[2].copy()
        data_ne[data[3]>0]=np.nan
        ne_ts=np.nanmean(data_ne,axis=1)
        table[experiment+"_"+sp+"_ne"]=ne_ts

table.esmFdbk2_po=(table.esmFdbk2_MAM_po+table.esmFdbk2_AMP_po+table.esmFdbk2_Bird_po)/3
table.esmFixClim2_po=(table.esmFixClim2_MAM_po+table.esmFixClim2_AMP_po+table.esmFixClim2_Bird_po)/3
table["ATMO+PHYS_po"]=(table["ATMO+PHYS_MAM_po"]+table["ATMO+PHYS_AMP_po"]+table["ATMO+PHYS_Bird_po"])/3
table.esmFdbk2_ne=(table.esmFdbk2_MAM_ne+table.esmFdbk2_AMP_ne+table.esmFdbk2_Bird_ne)/3
table.esmFixClim2_ne=(table.esmFixClim2_MAM_ne+table.esmFixClim2_AMP_ne+table.esmFixClim2_Bird_ne)/3
table["ATMO+PHYS_ne"]=(table["ATMO+PHYS_MAM_ne"]+table["ATMO+PHYS_AMP_ne"]+table["ATMO+PHYS_Bird_ne"])/3
table["lat"]=table.index*50+25

###

table2=pd.DataFrame({"lat":np.full([60],np.nan),
                    "esmFdbk2_MAM":np.full([60],np.nan),
                    "esmFdbk2_AMP":np.full([60],np.nan),
                    "esmFdbk2_Bird":np.full([60],np.nan),
                    "esmFixClim2_MAM":np.full([60],np.nan),
                    "esmFixClim2_AMP":np.full([60],np.nan),
                    "esmFixClim2_Bird":np.full([60],np.nan),
                    "ATMO+PHYS_MAM":np.full([60],np.nan),
                    "ATMO+PHYS_AMP":np.full([60],np.nan),
                    "ATMO+PHYS_Bird":np.full([60],np.nan),
                    })

taxas=["MAM","AMP","Bird"]
experiments=["esmFdbk2","esmFixClim2","ATMO+PHYS"]
for experiment in experiments:
    for sp in taxas:
        data=np.load("H:/基于关系的研究/点位/result2/"+sp+"_pred_"+experiment+"_migrate2.npy")
        var=experiment+"_"+sp
        data_all=data[2].copy()
        data_all[data[3]<0]=data_all[data[3]<0]*(-1)
        all=np.nanmean(data_all,axis=1)
        table2[experiment+"_"+sp]=all
table2["lat"]=table2.index*50+25

table2["esmFdbk2"]=(table2["esmFdbk2_MAM"]+table2["esmFdbk2_AMP"]+table2["esmFdbk2_Bird"])/3
table2["esmFixClim2"]=(table2["esmFixClim2_MAM"]+table2["esmFixClim2_AMP"]+table2["esmFixClim2_Bird"])/3
table2["ATMO+PHYS"]=(table2["ATMO+PHYS_MAM"]+table2["ATMO+PHYS_AMP"]+table2["ATMO+PHYS_Bird"])/3



plt.plot(table2["esmFdbk2_MAM"],3000-table2["lat"],color="#E6AF0C",alpha=0.8)
plt.plot(table2["esmFdbk2_AMP"],3000-table2["lat"],color="#78A51E",alpha=0.8)
plt.plot(table2["esmFdbk2_Bird"],3000-table2["lat"],color="#E52287",alpha=0.8)
plt.plot(table2["ATMO+PHYS_ne"],3000-table["lat"],color="#646464",linewidth=3,alpha=0.8)

##绘制方向的分布
import math
MAM_mitigrate2 = np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFdbk2_migrate2.npy")
def aspect(x,y):
    aspect = 57.29578 * math.atan2(y, x)
    if aspect < 0:
        cell = 90.0 - aspect
    elif aspect > 90.0:
        cell = 360.0 - aspect + 90.0
    else:
        cell = 90.0 - aspect
    return cell


def dir_cal(data):
    mit_dir = np.full([60, 144], np.nan)
    for i in range(60):
        for j in range(144):
            mit_dir[i, j] = aspect(data[0, i, j], data[1, i, j])
    dir_ana = np.full([60, 60], np.nan)
    for line in range(60):
        one=mit_dir[line]
        one=one[np.isnan(one)==False]
        amount=len(one)
        if(amount>0):
            for d in range(60):
                d_num=one[np.where((one>d*6)&(one<=(d+1)*6))]
                len_d=len(d_num)
                dir_ana[line,d]=(len_d/amount)*100
    dir_ana_re=np.full([120, 60], np.nan)
    for i in range(60):
        dir_ana_re[i*2:(i+1)*2, 0:15] = dir_ana[i, 45:60]
        dir_ana_re[i*2:(i+1)*2, 15:60] = dir_ana[i, 0:45]
    return dir_ana_re

ana=dir_cal(MAM_mitigrate2)
plt.imshow(ana,cmap="Greys",vmax=20)
plt.yticks([36,72,108],[r'45°N',r'0',r'45°S'])
plt.xticks([15,30,45],[r'North',r'East','South'])
plt.ylim(120,0)
plt.xlim(0,60)


def create_cmap(R,G,B):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(R/256, 1, N)
    vals[:, 1] = np.linspace(G/256, 1, N)
    vals[:, 2] = np.linspace(B/256, 1, N)
    vals=np.flipud(vals)
    newcmp = ListedColormap(vals)
    return newcmp

MAM_ATMO = np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFdbk2_migrate2.npy")
AMP_ATMO =np.load("H:/基于关系的研究/点位/result2/AMP_pred_esmFdbk2_migrate2.npy")
Bird_ATMO =np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFdbk2_migrate2.npy")
all_ATMO=(MAM_ATMO +AMP_ATMO+Bird_ATMO)/3

MAM_PHYS = np.load("H:/基于关系的研究/点位/result2/MAM_pred_esmFixClim2_migrate2.npy")
AMP_PHYS =np.load("H:/基于关系的研究/点位/result2/AMP_pred_esmFixClim2_migrate2.npy")
Bird_PHYS =np.load("H:/基于关系的研究/点位/result2/Bird_pred_esmFixClim2_migrate2.npy")
all_PHYS=(MAM_PHYS +AMP_PHYS+Bird_PHYS)/3

MAM_ATMO_PHYS = np.load("H:/基于关系的研究/点位/result2/MAM_pred_ATMO+PHYS_migrate2.npy")
AMP_ATMO_PHYS =np.load("H:/基于关系的研究/点位/result2/AMP_pred_ATMO+PHYS_migrate2.npy")
Bird_ATMO_PHYS =np.load("H:/基于关系的研究/点位/result2/Bird_pred_ATMO+PHYS_migrate2.npy")
all_ATMO_PHYS =(MAM_ATMO +AMP_ATMO+Bird_ATMO)/3


plt.figure(figsize=(8,11))
plt.subplots_adjust(wspace=0.05, hspace=0.05,left=0.01, bottom=0.01, right = 0.99, top = 0.99)

plt.subplot(3,4,4)
ana_all_ATMO=dir_cal(all_ATMO)
plt.imshow(ana_all_ATMO,cmap=create_cmap(100,100,100),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.subplot(3,4,1)
ana_Bird_ATMO=dir_cal(Bird_ATMO)
plt.imshow(ana_Bird_ATMO,cmap=create_cmap(231,60,150),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.subplot(3,4,2)
ana_MAM_ATMO=dir_cal(MAM_ATMO)
plt.imshow(ana_MAM_ATMO,cmap=create_cmap(234,187,49),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.subplot(3,4,3)
ana_AMP_ATMO=dir_cal(AMP_ATMO)
plt.imshow(ana_AMP_ATMO,cmap=create_cmap(139,177,63),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.subplot(3,4,8)
ana_all_PHYS=dir_cal(all_PHYS)
plt.imshow(ana_all_PHYS,cmap=create_cmap(100,100,100),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.subplot(3,4,5)
ana_Bird_PHYS=dir_cal(Bird_PHYS)
plt.imshow(ana_Bird_PHYS,cmap=create_cmap(231,60,150),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.subplot(3,4,6)
ana_MAM_PHYS=dir_cal(MAM_PHYS)
plt.imshow(ana_MAM_PHYS,cmap=create_cmap(234,187,49),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.subplot(3,4,7)
ana_AMP_PHYS=dir_cal(AMP_PHYS)
plt.imshow(ana_AMP_PHYS,cmap=create_cmap(139,177,63),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)


plt.subplot(3,4,12)
ana_all_ATMO_PHYS=dir_cal(all_ATMO_PHYS)
plt.imshow(ana_all_ATMO_PHYS,cmap=create_cmap(100,100,100),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.subplot(3,4,9)
ana_Bird_ATMO_PHYS=dir_cal(Bird_ATMO_PHYS)
plt.imshow(ana_Bird_ATMO_PHYS,cmap=create_cmap(231,60,150),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.subplot(3,4,10)
ana_MAM_ATMO_PHYS=dir_cal(MAM_ATMO_PHYS)
plt.imshow(ana_MAM_ATMO_PHYS,cmap=create_cmap(234,187,49),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.subplot(3,4,11)
ana_AMP_ATMO_PHYS=dir_cal(AMP_ATMO_PHYS)
plt.imshow(ana_AMP_ATMO_PHYS,cmap=create_cmap(139,177,63),vmax=20)
plt.yticks([36,72,108],[r'',r'',r''],rotation=90,)
plt.xticks([15,30,45],[r'',r'',r''])
plt.ylim(120,0)
plt.xlim(0,60)

plt.savefig("H:/基于关系的研究/点位/result/图/Figures_oct/Dir/dir_distribution.tiff",dpi=300)

plt.imshow(MAM_ATMO[0])
cut=MAM_ATMO[:,18:54,:]
#正
s1=cut[2].copy()
s1[cut[3]<0]=np.nan
po=np.nansum(s1)

s2=cut[2].copy()
s2[cut[3]>0]=np.nan
ne=np.nansum(s2)


################################### Fig S2


################################### Fig S3