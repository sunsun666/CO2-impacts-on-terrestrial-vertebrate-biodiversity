from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import *
from statsmodels.tsa import arima_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor #随机森林
from sklearn import svm#支撑向量机
from sklearn.neural_network import MLPRegressor#人工神经网络
import pandas as pd
from bayes_opt import BayesianOptimization
import pickle
#############################      machine learning process      ################################


#method1 LightGBM

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




#method2 随机森林
#MAM
{'target': 0.9699696838082013,
 'params': {'max_depth': 271.61081668218185,
  'max_features': 0.22953039933500957,
  'min_samples_leaf': 1.0,
  'min_samples_split': 6.082743114126411,
  'n_estimators': 2101.9393226227703}}



#method3 人工神经网络





#method4 支持向量机



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




#try different machine learning method
import time
import pandas as pd

def get_r2(X_test,Y_test,model):
    """
    Calculation of the unadjusted r-squared, goodness of fit metric
    """
    y_pred=model.predict(X_test)
    y_true=np.asarray(Y_test.iloc[:,0])
    sse  = np.square(y_pred - y_true).sum()
    sst  = np.square(y_true - y_true.mean()).sum()
    return 1 - sse/sst

pounds = {
    'subsample': (0.1,1),
    'bagging_freq': (1,10),
    'feature_fraction': (0.5,1),
    'slient': (0.1,1),
    'learning_rate': (0,0.5),
}

for sp in range(3):
    spp=str(sp+1)
    x_train = pd.read_parquet("H:/relatioship/点位/result/模型参数/data_sp_x_train.parquet")
    x_test = pd.read_parquet("H:/relatioship/点位/result/模型参数/data_sp_x_test.parquet")
    y_train = pd.read_parquet("H:/relatioship/点位/result/模型参数/data_sp_y"+spp+"_train.parquet")
    y_test = pd.read_parquet("H:/relatioship/点位/result/模型参数/data_sp_y"+spp+"_test.parquet")

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    def AUC_max(subsample,bagging_freq,feature_fraction,slient,learning_rate):
        params = {
            'booster': 'gbtree',
            'objective': 'regression',
            'num_leaves': 31,
            'subsample': subsample,
            'bagging_freq': int(bagging_freq),
            'feature_fraction ': feature_fraction,
            'slient': slient,
            'learning_rate ': learning_rate,
            'seed': 0
        }
        LGBM = lgb.train(params,lgb_train,num_rounds,valid_sets=lgb_train,early_stopping_rounds=5)

        LGBM_AUC = get_r2(x_test, y_test, model=LGBM)
        return  LGBM_AUC

    start_time = time.time()
    optimizer = BayesianOptimization(
        f=AUC_max,
        pbounds=pounds,
        random_state=1
    )
    optimizer.maximize(
        init_points=100,
        n_iter=10,
    )

    end_time = time.time()
    execution_time = str(int(end_time - start_time))
    df=pd.DataFrame(optimizer.res)
    df.to_csv("C:/Users/dell/Desktop/生境适宜性/Earths future/revision/LGBM"+spp+"_time"+execution_time+".csv")

import pickle

