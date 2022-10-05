#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix,roc_auc_score
import sklearn
from sklearn.tree import DecisionTreeClassifier
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.distance import geodesic
from geopy import Point
import math
import datetime
import datetime as dt
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import Pool, CatBoostClassifier



# In[2]:


tours_convoy=pd.read_csv('tour_convoy.csv')
bikers=pd.read_csv('bikers.csv')
bikers_net=pd.read_csv('bikers_network.csv')
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
tours=pd.read_csv('tours.csv')
tours.rename(columns={'biker_id': 'organizer'}, inplace=True)



# In[3]:


def from_bikers(x):
    try:
        return x[['latitude', 'longitude']] if x['latitude']==x['latitude'] else bikers.loc[x['organizer'], ['biker_latitude', 'biker_longitude']].rename({'biker_latitude': 'latitude', 'biker_longitude': 'longitude'})
    except:
        return pd.Series({'latitude': None, 'longitude': None})

def total(tid, col):
    try:
        return len(tours_convoy.loc[tid, col].split(' '))
    except:
        return 0

    
    
def friends_with_org(x):
    biker = x['biker_id']
    org = x['organizer']
    
    return int(bikers_net.loc[biker, 'friends'].find(org)!=-1)

### Biker id counts in Train and Test Data
train_counts = train_data['biker_id'].value_counts()
test_counts = test_data['biker_id'].value_counts()


### adding the values to table
def find_test_count(x):
    try:
        return test_counts[x]
    except:
        return 0
    
def find_train_count(x):
    try:
        return train_counts[x]
    except:
        return 0
    
def total_friends(x):
    return len(bikers_net.loc[x, 'friends'].split(' ')[:-1])


# In[4]:


train_data = train_data.merge(bikers, on='biker_id').merge(tours, on='tour_id')
test_data = test_data.merge(bikers, on='biker_id').merge(tours, on='tour_id')

tours_convoy.set_index('tour_id', inplace=True)
bikers.set_index('biker_id', inplace=True)
# tours.set_index('tour_id', inplace=True)
bikers_net.set_index('biker_id', inplace=True)


# In[5]:


class MyTransformer:
    def __init__(self, n_clusters=20):
        self.cluster_gen = KMeans(n_clusters)
        self.n_clusters = n_clusters
        self.empty = True
    
    def fit(self, cp):
        df = cp.copy()
        
        try:
            self.cluster_gen.fit(df.loc[:, 'w1': 'w_other'])
            self.empty = False

            return self
        except:
            print("Error Thrown")
            self.cluster_gen = KMeans(self.n_clusters)
            self.empty = False
                
            return self
        
    def transform(self, df):
        
        if self.empty:
            Exception("Transformer Needs to be fit first!")
        
        cp = df.copy()

        cp['total_maybe'] = cp['tour_id'].apply(total, args=['maybe'])
        cp['total_notgoing'] = cp['tour_id'].apply(total, args=['not_going'])
        cp['total_going'] = cp['tour_id'].apply(total, args=['going'])
        cp['total_invited'] = cp['tour_id'].apply(total, args=['invited'])


        ### Whether or not biker is friends with organizer
        cp['friends_with_org'] = cp.apply(friends_with_org, axis=1)
        cp['total_friends'] = cp['biker_id'].apply(total_friends)
        
        
        cp['train_count'] = cp['biker_id'].apply(find_train_count)
        cp['test_count'] = cp['biker_id'].apply(find_test_count)

        cp['test_count'].fillna(0, inplace=True)
        cp['train_count'].fillna(0, inplace=True)
        

        cp.loc[:, 'timestamp'] = pd.to_datetime(cp['timestamp'], format='%d-%m-%Y %H:%M:%S')
        cp.loc[:, 'week'] = cp['timestamp'].apply(lambda x: x.dayofweek)
        cp.loc[:, 'month'] = cp['timestamp'].apply(lambda x: x.month)
        cp.loc[:, 'year'] = cp['timestamp'].apply(lambda x: x.year)
        cp.loc[:, 'day'] = cp['timestamp'].apply(lambda x: x.day)


        cp.loc[:, 'bornIn'].replace('None', -1, inplace=True)
        cp.loc[:, 'bornIn'] = cp['bornIn'].astype('int')
        cp.loc[:, 'bornIn'].replace(-1, cp['bornIn'].mode()[0], inplace=True)


        cp.loc[:, 'age'] = 2012 - cp['bornIn']

        cp.loc[:, 'tour_date'] = pd.to_datetime(cp['tour_date'], format='%d-%m-%Y')
        cp.loc[:, 'tour_week'] = cp['tour_date'].apply(lambda x: x.dayofweek)
        cp.loc[:, 'tour_month'] = cp['tour_date'].apply(lambda x: x.month)
        cp.loc[:, 'tour_year'] = cp['tour_date'].apply(lambda x: x.year)
        cp.loc[:, 'tour_day'] = cp['tour_date'].apply(lambda x: x.day)
        
        cp.loc[:, 'tour_on_weekend'] = cp['tour_date'].apply(lambda x : (x.dayofweek > 4))
        cp.loc[:, 'tour_quarter'] = ((cp['tour_month'])/3).apply(np.ceil)


        cp.loc[:, 'member_since'] = pd.to_datetime(cp['member_since'], format='%d-%m-%Y')
        cp.loc[:, 'member_week'] = cp['member_since'].apply(lambda x: x.dayofweek)
        cp.loc[:, 'member_month'] = cp['member_since'].apply(lambda x: x.month)
        cp.loc[:, 'member_year'] = cp['member_since'].apply(lambda x: x.year)
        cp.loc[:, 'member_day'] = cp['member_since'].apply(lambda x: x.day)
        
        cp.loc[:, 'tour_on_weekend'] = cp['tour_on_weekend'].astype('int')
        
        cp.loc[:, 'membership_time'] = ((cp['tour_date'] - cp['member_since']).apply(lambda x: x.total_seconds()))
        
        cp.loc[:, 'time_gap'] = ((cp['tour_date'] - cp['timestamp']).apply(lambda x: x.total_seconds()))
        
        cp.loc[:, 'group_time'] = ((cp['member_since'] - cp['timestamp']).apply(lambda x: x.total_seconds()))

        hold = cp.loc[:, ['biker_id', 'tour_id', 'time_gap']].groupby(by='biker_id').agg({'time_gap': 'min'})
        cp.loc[:, 'special'] = cp.apply(lambda x: 1 if x['time_gap']==0 else (hold.loc[x['biker_id'], 'time_gap']/x['time_gap']), axis=1)
        
        return cp
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    


# In[6]:


trans = MyTransformer()
train_data = trans.fit_transform(train_data)
test_data = trans.transform(test_data)
tours_convoy.reset_index(inplace=True)
bikers.reset_index(inplace=True)
bikers_net.reset_index(inplace=True)


# In[7]:


areas = bikers['area'].unique()
b=pd.DataFrame(areas,columns=['area'])

def func_lat_long(location):
    geolocator = Nominatim(user_agent="myApp")
    try:
        location = geolocator.geocode(location)
        return pd.Series({'latitude': location.latitude, 'longitude': location.longitude})
    except:
        return pd.Series({'latitude': None, 'longitude': None})

b1=pd.DataFrame()
b1=b['area'].apply(lambda x:func_lat_long(x))
b1['area']=areas
for i in range(bikers.shape[0]):
    area=bikers.loc[i,'area']
    try:
        lat=b1.loc[b1['area'] == area].iloc[0,0]
        long=b1.loc[b1['area'] == area].iloc[0,1]
        bikers.at[i,'lat_bikers']=lat
        bikers.at[i,'long_bikers']=long
    except:
        pass

bikers.set_index('biker_id',inplace=True)
bikers_net.set_index('biker_id',inplace=True)
registered = bikers.index.to_list()



# In[8]:


def friends_coords(x):
    lat = []
    long = []
    r=bikers.loc[x]
    if(np.isnan(r['lat_bikers'])):
        try:
            row = bikers_net.loc[x]
            if(not row['friends']): Exception("lonely bitch")
            for friend in row['friends'].split(' '):
                if(friend in registered):
                    try:
                        lat.append(bikers.loc[friend]['lat_bikers'])
                        long.append(bikers.loc[friend]['long_bikers'])
                    except:
                        pass
            if(len(lat)==0): Exception('none registered')
            lat=[x for x in lat if (math.isnan(x) == False)]
            long=[x for x in long if (math.isnan(x) == False)]
            return sum(lat)/len(lat),sum(long)/len(long)
        except:
            return None,None
    else:
        return r['lat_bikers'],r['long_bikers']



bikers1=bikers.reset_index()
b=bikers1['biker_id'].apply(lambda x:friends_coords(x))
b=pd.DataFrame(b)
b.columns=['lat_long']
bikers.reset_index(inplace=True)
bikers=pd.merge(bikers,b,left_index = True,right_index=True)



# In[9]:


def lat(x):
    try:
        return x[0]
    except:
        return np.nan
def long(x):
    try:
        return x[1]
    except:
        return np.nan

bikers['lat_bikers']=bikers['lat_long'].apply(lambda x:lat(x))
bikers['long_bikers']=bikers['lat_long'].apply(lambda x:long(x))

bikers = bikers.filter(['biker_id','lat_bikers','long_bikers'], axis=1)
train_data=pd.merge(train_data,bikers,how='inner',left_on=['biker_id'],right_on=['biker_id'])

tours = tours.filter(['tour_id','latitude','longitude'], axis=1)
train_data=pd.merge(train_data,tours,how='inner',left_on=['tour_id'],right_on=['tour_id'])

bikers.set_index('biker_id',inplace=True)
tours_convoy.set_index('tour_id',inplace=True)

registered_bikers=bikers.index.to_list()
registered_tours = tours_convoy.index.to_list()

train_data.set_index('tour_id',inplace=True)
train_data1=train_data.reset_index()
tour_data=train_data1.drop_duplicates(subset='tour_id', keep='first')
tour_data = tour_data.filter(['tour_id','latitude_y','longitude_y'], axis=1)
tour_data.set_index('tour_id',inplace=True)
tour_data1=tour_data.reset_index()

def tour_coords(x):
    lat = []
    long = []
    r=tour_data.loc[x]
    if(np.isnan(r['latitude_y'])):
        try:
            row = tours_convoy.loc[x]
            if(not row['going']): Exception("bad tour")
            for friend in row['going'].split(' '):
                if(friend in registered_bikers):
                    try:
                        lat.append(bikers.loc[friend]['lat_bikers'])
                        long.append(bikers.loc[friend]['long_bikers'])
                    except:
                        pass
            if(len(lat)==0): Exception('none registered')
            lat=[x for x in lat if (math.isnan(x) == False)]
            long=[x for x in long if (math.isnan(x) == False)]
            return sum(lat)/len(lat),sum(long)/len(long)
        except:
            return np.nan,np.nan
    else:
        return r['latitude_y'],r['longitude_y']


t=tour_data1['tour_id'].apply(lambda x:tour_coords(x))
t=pd.DataFrame(t)
t.columns=['lat_long_tour']
tour_data1=pd.merge(tour_data1,t,how='inner',left_index=True,right_index=True)

tour_data1['lat']=tour_data1['lat_long_tour'].apply(lambda x:lat(x))
tour_data1['long']=tour_data1['lat_long_tour'].apply(lambda x:long(x))
tour_data1.set_index('tour_id',inplace=True)
tour_data2 = tour_data1.filter(['lat','long'], axis=1)
tour_data2.columns=['lat_tour','long_tour']
train_data=pd.merge(train_data,tour_data2,how='inner',left_on=['tour_id'],right_on=['tour_id'])

def lat_long_tour(row):
    try:
        return Point(latitude=row['lat_tour'], longitude=row['long_tour'])
    except ValueError:
        return np.nan

def lat_long_biker(row):
    try:
        return Point(latitude=row['lat_bikers'], longitude=row['long_bikers'])
    except ValueError:
        return np.nan

train_data['point_biker'] = train_data.apply(lambda row:lat_long_biker(row), axis=1)
train_data['point_tour'] = train_data.apply(lambda row:lat_long_tour(row), axis=1)

def distance(row):
    try:
        return geodesic(row['point_biker'], row['point_tour']).km
    except ValueError:
        return np.nan

train_data['distance_km'] = train_data.apply(lambda row:distance(row), axis=1)
train_data.reset_index(inplace=True)
tours_convoy.reset_index(inplace=True)
bikers.reset_index(inplace=True)
bikers_net.reset_index(inplace=True)







# In[10]:


test_data=pd.merge(test_data,bikers,how='inner',left_on=['biker_id'],right_on=['biker_id'])

tours = tours.filter(['tour_id','latitude','longitude'], axis=1)
test_data=pd.merge(test_data,tours,how='inner',left_on=['tour_id'],right_on=['tour_id'])


bikers.set_index('biker_id',inplace=True)
tours_convoy.set_index('tour_id',inplace=True)

registered_bikers=bikers.index.to_list()
registered_tours = tours_convoy.index.to_list()

test_data.set_index('tour_id',inplace=True)
test_data1=test_data.reset_index()
tour_data_test=test_data1.drop_duplicates(subset='tour_id', keep='first')
tour_data_test = tour_data_test.filter(['tour_id','latitude_y','longitude_y'], axis=1)
tour_data_test.set_index('tour_id',inplace=True)
tour_data_test1=tour_data_test.reset_index()


def tour_coords_test(x):
    lat = []
    long = []
    r=tour_data_test.loc[x]
    if(np.isnan(r['latitude_y'])):
        try:
            row = tours_convoy.loc[x]
            if(not row['going']): Exception("bad tour")
            for friend in row['going'].split(' '):
                if(friend in registered_bikers):
                    try:
                        lat.append(bikers.loc[friend]['lat_bikers'])
                        long.append(bikers.loc[friend]['long_bikers'])
                    except:
                        pass
            if(len(lat)==0): Exception('none registered')
            lat=[x for x in lat if (math.isnan(x) == False)]
            long=[x for x in long if (math.isnan(x) == False)]
            return sum(lat)/len(lat),sum(long)/len(long)
        except:
            return np.nan,np.nan
    else:
        return r['latitude_y'],r['longitude_y']


t_test=tour_data_test1['tour_id'].apply(lambda x:tour_coords_test(x))
t_test=pd.DataFrame(t_test)
t_test.columns=['lat_long_tour']
tour_data_test1=pd.merge(tour_data_test1,t_test,how='inner',left_index=True,right_index=True)

tour_data_test1['lat']=tour_data_test1['lat_long_tour'].apply(lambda x:lat(x))
tour_data_test1['long']=tour_data_test1['lat_long_tour'].apply(lambda x:long(x))
tour_data_test1.set_index('tour_id',inplace=True)
tour_data_test2 = tour_data_test1.filter(['lat','long'], axis=1)
tour_data_test2.columns=['lat_tour','long_tour']
test_data=pd.merge(test_data,tour_data_test2,how='inner',left_on=['tour_id'],right_on=['tour_id'])


test_data['point_biker'] = test_data.apply(lambda row:lat_long_biker(row), axis=1)
test_data['point_tour'] = test_data.apply(lambda row:lat_long_tour(row), axis=1)

test_data['distance_km'] = test_data.apply(lambda row:distance(row), axis=1)

test_data.reset_index(inplace=True)
tours_convoy.reset_index(inplace=True)
bikers.reset_index(inplace=True)


# In[11]:


test_data['going']=0
test_data['maybe']=0
test_data['not_going']=0
test_data['inv']=0

bikers_net.set_index('biker_id', inplace=True)

for index, row in tours_convoy.iterrows(): 
    list_going=[i for i in str(row['going']).split()]  
    list_maybe=[i for i in str(row['maybe']).split()]
    list_inv=[i for i in str(row['invited']).split()]
    list_not_going=[i for i in str(row['not_going']).split()]
    tour=row['tour_id']
    l=test_data.index[test_data['tour_id'] ==tour].tolist()
    for i in l:
        biker_id=test_data.iloc[i,1]
        friend=bikers_net.loc[biker_id][0]
        friend=friend.split()
        for j in friend:
            if j in list_going:
                test_data.loc[i,'going']=test_data.loc[i,'going']+1
        for j in friend:
            if j in list_maybe:
                test_data.loc[i,'maybe']=test_data.loc[i,'maybe']+1
        for j in friend:
            if j in list_not_going:
                test_data.loc[i,'not_going']=test_data.loc[i,'not_going']+1
        for j in friend:
            if j in list_inv:
                test_data.loc[i,'inv']=test_data.loc[i,'inv']+1
                
bikers_net.reset_index(inplace=True)


train_data['going']=0
train_data['maybe']=0
train_data['not_going']=0
train_data['inv']=0

bikers_net.set_index('biker_id', inplace=True)

for index, row in tours_convoy.iterrows(): 
    list_going=[i for i in str(row['going']).split()]  
    list_maybe=[i for i in str(row['maybe']).split()]
    list_inv=[i for i in str(row['invited']).split()]
    list_not_going=[i for i in str(row['not_going']).split()]
    tour=row['tour_id']
    l=train_data.index[train_data['tour_id'] ==tour].tolist()
    for i in l:
        biker_id=train_data.iloc[i,1]
        friend=bikers_net.loc[biker_id][0]
        friend=friend.split()
        for j in friend:
            if j in list_going:
                train_data.loc[i,'going']=train_data.loc[i,'going']+1
        for j in friend:
            if j in list_maybe:
                train_data.loc[i,'maybe']=train_data.loc[i,'maybe']+1
        for j in friend:
            if j in list_not_going:
                train_data.loc[i,'not_going']=train_data.loc[i,'not_going']+1
        for j in friend:
            if j in list_inv:
                train_data.loc[i,'inv']=train_data.loc[i,'inv']+1
                
bikers_net.reset_index(inplace=True)




# In[12]:


train_data["gender"].fillna(train_data["gender"].mode()[0], inplace = True) 
train_data["time_zone"].fillna(train_data["time_zone"].mode()[0], inplace = True) 

test_data["gender"].fillna(test_data["gender"].mode()[0], inplace = True) 
test_data["time_zone"].fillna(test_data["time_zone"].mode()[0], inplace = True) 

test_data1 = test_data.copy()
train_data1 = train_data.copy()



# In[30]:


test_data = test_data1.copy()
train_data = train_data1.copy()

l=[]

for i in range(train_data.shape[0]):
    if(train_data.loc[i,'like']):
        l.append(1)
    else:
        l.append(0)
    
l=pd.DataFrame(l,columns=['labels'])
#l.to_csv('y_train.csv')
y=l

train_data.drop(columns=['like','dislike','state','pincode','biker_id','tour_id','timestamp','area'
                         ,'member_week','member_day','member_year','lat_tour','long_tour'
                         ,'latitude_y','longitude_y'
                         ,'week','month','year','day','tour_year','tour_quarter'
                         ,'member_since','organizer','train_count','test_count'
                         ,'tour_date','point_biker'
                         ,'point_tour','city','country'],axis=1,inplace=True)


test_data.drop(columns= ['state','pincode','biker_id','tour_id','timestamp','area'
                         ,'member_week','member_day','member_year','lat_tour','long_tour'
                         ,'latitude_y','longitude_y'
                         ,'week','month','year','day','tour_year','tour_quarter'
                         ,'member_since','organizer','train_count','test_count'
                         ,'tour_date','point_biker'
                         ,'point_tour','city','country'],axis=1,inplace=True)

train_data['word_sum']=train_data.loc[:, 'w1':'w100'].sum(axis=1)
train_data['word_ratio']=train_data['word_sum']/train_data['w_other']

test_data['word_sum']=test_data.loc[:, 'w1':'w100'].sum(axis=1)
test_data['word_ratio']=test_data['word_sum']/train_data['w_other']


x_train=train_data.loc[:, 'w1':'w100']
x_test=test_data.loc[:, 'w1':'w100']

kmeans = KMeans(n_clusters=15,random_state=0)
kmeans.fit(x_train)
x_cluster_train = kmeans.predict(x_train)
x_cluster_test = kmeans.predict(x_test)

x_cluster_train=pd.DataFrame(x_cluster_train,columns=['cluster'])
x_cluster_test=pd.DataFrame(x_cluster_test,columns=['cluster'])

train_data=pd.merge(train_data,x_cluster_train,how='inner',left_index=True,right_index=True)
test_data=pd.merge(test_data,x_cluster_test,how='inner',left_index=True,right_index=True)

train_data['cluster']=train_data['cluster'].map(str)
test_data['cluster']=test_data['cluster'].map(str)

train_data['time_zone']=train_data['time_zone'].map(float)
test_data['time_zone']=test_data['time_zone'].map(float)

train_data['membership_time']=train_data['membership_time']/(3600*24)
train_data['time_gap']=train_data['time_gap']/(3600*24)
train_data['group_time']=train_data['group_time']/(3600*24)

test_data['membership_time']=test_data['membership_time']/(3600*24)
test_data['time_gap']=test_data['time_gap']/(3600*24)
test_data['group_time']=test_data['group_time']/(3600*24)


train_data['tour_week']=train_data['tour_week'].map(str)
test_data['tour_week']=test_data['tour_week'].map(str)

train_data['invited']=train_data['invited'].map(str)
test_data['invited']=test_data['invited'].map(str)

train_data['friends_with_org']=train_data['friends_with_org'].map(str)
test_data['friends_with_org']=test_data['friends_with_org'].map(str)

train_data['tour_month']=train_data['tour_month'].map(str)
test_data['tour_month']=test_data['tour_month'].map(str)

#train_data['tour_quarter']=train_data['tour_quarter'].map(str)
#test_data['tour_quarter']=test_data['tour_quarter'].map(str)

train_data['tour_day']=train_data['tour_day'].map(str)
test_data['tour_day']=test_data['tour_day'].map(str)

train_data['member_month']=train_data['member_month'].map(str)
test_data['member_month']=test_data['member_month'].map(str)


for c in train_data.columns:
    col_type = train_data[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        train_data[c] = train_data[c].astype('category')
        
for c in test_data.columns:
    col_type = test_data[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        test_data[c] = test_data[c].astype('category')
        
        
X_train, X_valid, y_train, y_valid = train_test_split(train_data, y,train_size=0.8, test_size=0.2,random_state=42) 


model=lgb.LGBMClassifier(random_state=42,n_estimators=600,learning_rate=0.01,objective='binary',
                        metric='AUC',max_depth=-1,num_leaves=64,first_metric_only=False,feature_fraction=0.4)


model.fit(X_train,y_train)
pred_test_prob=model.predict_proba(test_data)

test_ids=test_data1[['tour_id','biker_id']]
test_ids['likeliness']=pred_test_prob[:,1]
test_ids.sort_values(['biker_id', 'likeliness'], ascending=[True, False],inplace=True)
df=test_ids.groupby('biker_id')

bikers=[]
tour_id=[]
for key,value in df:
    bikers.append(key)
    tour=[]
    for j in value['tour_id']:
        tour.append(j)
    tour_id.append(tour)
    
tour=[]
for i in tour_id:
    tour.append(" ".join(i))

submission= pd.DataFrame(list(zip(bikers, tour)),columns =['biker_id', 'tour_id'])
#submission.to_csv('sub5.csv')

test_ids.sort_values(['biker_id', 'tour_id'], ascending=[True, True],inplace=True)
#test.to_csv('pred_cat_cluster_15.csv')
test_id_lgb=test_ids.copy()


# In[31]:


test_data = test_data1.copy()
train_data = train_data1.copy()

l=[]

for i in range(train_data.shape[0]):
    if(train_data.loc[i,'like']):
        l.append(1)
    else:
        l.append(0)
    
l=pd.DataFrame(l,columns=['labels'])
y=l

train_data.drop(columns=['like','dislike','state','pincode','biker_id','tour_id','timestamp','area'
                         ,'member_week','member_day','member_year','lat_tour','long_tour'
                         ,'latitude_y','longitude_y'
                         ,'week','month','year','day','tour_year','tour_quarter'
                         ,'member_since','organizer','train_count','test_count'
                         ,'tour_date','point_biker'
                         ,'point_tour','city','country'],axis=1,inplace=True)


test_data.drop(columns= ['state','pincode','biker_id','tour_id','timestamp','area'
                         ,'member_week','member_day','member_year','lat_tour','long_tour'
                         ,'latitude_y','longitude_y'
                         ,'week','month','year','day','tour_year','tour_quarter'
                         ,'member_since','organizer','train_count','test_count'
                         ,'tour_date','point_biker'
                         ,'point_tour','city','country'],axis=1,inplace=True)


train_data['word_sum']=train_data.loc[:, 'w1':'w100'].sum(axis=1)
train_data['word_ratio']=train_data['word_sum']/train_data['w_other']

test_data['word_sum']=test_data.loc[:, 'w1':'w100'].sum(axis=1)
test_data['word_ratio']=test_data['word_sum']/train_data['w_other']


x_train=train_data.loc[:, 'w1':'w100']
x_test=test_data.loc[:, 'w1':'w100']

kmeans = KMeans(n_clusters=15,random_state=0)
kmeans.fit(x_train)
x_cluster_train = kmeans.predict(x_train)
x_cluster_test = kmeans.predict(x_test)

x_cluster_train=pd.DataFrame(x_cluster_train,columns=['cluster'])
x_cluster_test=pd.DataFrame(x_cluster_test,columns=['cluster'])

train_data=pd.merge(train_data,x_cluster_train,how='inner',left_index=True,right_index=True)
test_data=pd.merge(test_data,x_cluster_test,how='inner',left_index=True,right_index=True)

train_data['cluster']=train_data['cluster'].map(str)
test_data['cluster']=test_data['cluster'].map(str)

train_data['time_zone']=train_data['time_zone'].map(float)
test_data['time_zone']=test_data['time_zone'].map(float)

train_data['membership_time']=train_data['membership_time']/(3600*24)
train_data['time_gap']=train_data['time_gap']/(3600*24)
train_data['group_time']=train_data['group_time']/(3600*24)

test_data['membership_time']=test_data['membership_time']/(3600*24)
test_data['time_gap']=test_data['time_gap']/(3600*24)
test_data['group_time']=test_data['group_time']/(3600*24)


train_data['tour_week']=train_data['tour_week'].map(str)
test_data['tour_week']=test_data['tour_week'].map(str)

train_data['invited']=train_data['invited'].map(str)
test_data['invited']=test_data['invited'].map(str)

train_data['friends_with_org']=train_data['friends_with_org'].map(str)
test_data['friends_with_org']=test_data['friends_with_org'].map(str)

train_data['tour_month']=train_data['tour_month'].map(str)
test_data['tour_month']=test_data['tour_month'].map(str)

train_data['tour_day']=train_data['tour_day'].map(str)
test_data['tour_day']=test_data['tour_day'].map(str)

train_data['member_month']=train_data['member_month'].map(str)
test_data['member_month']=test_data['member_month'].map(str)


X_train, X_valid, y_train, y_valid = train_test_split(train_data, y,train_size=0.8, test_size=0.2,random_state=42) 

from catboost import Pool, CatBoostClassifier

cat_features=['location_id','language_id','gender','tour_week','invited','friends_with_org','cluster'
             ,'tour_month','tour_day','member_month']

train_label = y_train
eval_label = y_valid


train_dataset = Pool(data=X_train,label=train_label,cat_features=cat_features)

eval_dataset = Pool(data=X_valid,label=eval_label,cat_features=cat_features)

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=5000,
                           learning_rate=0.01,
                           depth=8,loss_function='Logloss',eval_metric='AUC',use_best_model=True,random_seed=42
                          ,l2_leaf_reg=8)


model.fit(train_dataset,plot=True,eval_set=eval_dataset)
pred_val_prob=model.predict_proba(X_valid)
pred_test_prob=model.predict_proba(test_data)
pred_val=model.predict(X_valid)
pred_test=model.predict(test_data)

test_ids=test_data1[['tour_id','biker_id']]
test_ids['likeliness']=pred_test_prob[:,1]
test_ids.sort_values(['biker_id', 'likeliness'], ascending=[True, False],inplace=True)
df=test_ids.groupby('biker_id')

bikers=[]
tour_id=[]
for key,value in df:
    bikers.append(key)
    tour=[]
    for j in value['tour_id']:
        tour.append(j)
    tour_id.append(tour)
    
tour=[]
for i in tour_id:
    tour.append(" ".join(i))

submission= pd.DataFrame(list(zip(bikers, tour)),columns =['biker_id', 'tour_id'])

test_ids.sort_values(['biker_id', 'tour_id'], ascending=[True, True],inplace=True)
test_id_cat=test_ids.copy()



# In[32]:


test_id=test_id_cat.copy()
pred=(test_id_cat['likeliness'])

test_id['likeliness']=pred

test_id.sort_values(['biker_id', 'likeliness'], ascending=[True, False],inplace=True)
df=test_id.groupby('biker_id')

bikers=[]
tour_id=[]
for key,value in df:
    bikers.append(key)
    tour=[]
    for j in value['tour_id']:
        tour.append(j)
    tour_id.append(tour)
    
tour=[]
for i in tour_id:
    tour.append(" ".join(i))

submission= pd.DataFrame(list(zip(bikers, tour)),columns =['biker_id', 'tour_id'])
submission.to_csv('CE18B063_CE18B016_1.csv',index=False)


# In[33]:


test_data = test_data1.copy()
train_data = train_data1.copy()
l=[]

for i in range(train_data.shape[0]):
    if(train_data.loc[i,'like']):
        l.append(1)
    else:
        l.append(0)
    
l=pd.DataFrame(l,columns=['labels'])
y=l

train_data.drop(columns=['like','dislike','state','pincode','biker_id','tour_id','timestamp','area'
                         ,'member_week','member_month','member_day','member_year'
                         ,'week','month','year','day','tour_year'
                         ,'special','lat_tour','long_tour'
                         ,'member_since','lat_bikers','long_bikers','organizer'
                         ,'tour_date','latitude_x','longitude_x','latitude_y','longitude_y'
                         ,'point_biker','point_tour','city','country'],axis=1,inplace=True)


test_data.drop(columns=['state','pincode','biker_id','tour_id','timestamp','area'
                         ,'member_week','member_month','member_day','member_year'
                         ,'week','month','year','day','tour_year'
                         ,'special','lat_tour','long_tour'
                         ,'member_since','lat_bikers','long_bikers','organizer'
                         ,'tour_date','latitude_x','longitude_x','latitude_y','longitude_y'
                         ,'point_biker','point_tour','city','country'],axis=1,inplace=True)



train_data['word_sum']=train_data.loc[:, 'w1':'w100'].sum(axis=1)
train_data['word_ratio']=train_data['word_sum']/train_data['w_other']

test_data['word_sum']=test_data.loc[:, 'w1':'w100'].sum(axis=1)
test_data['word_ratio']=test_data['word_sum']/train_data['w_other']


x_train=train_data.loc[:, 'w1':'w100']
x_test=test_data.loc[:, 'w1':'w100']

kmeans = KMeans(n_clusters=15,random_state=0)
kmeans.fit(x_train)
x_cluster_train = kmeans.predict(x_train)
x_cluster_test = kmeans.predict(x_test)

x_cluster_train=pd.DataFrame(x_cluster_train,columns=['cluster'])
x_cluster_test=pd.DataFrame(x_cluster_test,columns=['cluster'])

train_data=pd.merge(train_data,x_cluster_train,how='inner',left_index=True,right_index=True)
test_data=pd.merge(test_data,x_cluster_test,how='inner',left_index=True,right_index=True)

train_data['cluster']=train_data['cluster'].map(str)
test_data['cluster']=test_data['cluster'].map(str)

train_data['time_zone']=train_data['time_zone'].map(float)
test_data['time_zone']=test_data['time_zone'].map(float)

train_data['membership_time']=train_data['membership_time']/(3600*24)
train_data['time_gap']=train_data['time_gap']/(3600*24)
train_data['group_time']=train_data['group_time']/(3600*24)

test_data['membership_time']=test_data['membership_time']/(3600*24)
test_data['time_gap']=test_data['time_gap']/(3600*24)
test_data['group_time']=test_data['group_time']/(3600*24)


train_data['tour_week']=train_data['tour_week'].map(str)
test_data['tour_week']=test_data['tour_week'].map(str)

train_data['invited']=train_data['invited'].map(str)
test_data['invited']=test_data['invited'].map(str)

train_data['friends_with_org']=train_data['friends_with_org'].map(str)
test_data['friends_with_org']=test_data['friends_with_org'].map(str)

train_data['tour_month']=train_data['tour_month'].map(str)
test_data['tour_month']=test_data['tour_month'].map(str)

train_data['tour_quarter']=train_data['tour_quarter'].map(str)
test_data['tour_quarter']=test_data['tour_quarter'].map(str)

train_data['tour_day']=train_data['tour_day'].map(str)
test_data['tour_day']=test_data['tour_day'].map(str)

train_data["gender"].fillna(train_data["gender"].mode()[0], inplace = True) 
train_data["time_zone"].fillna(train_data["time_zone"].mode()[0], inplace = True) 

test_data["gender"].fillna(test_data["gender"].mode()[0], inplace = True) 
test_data["time_zone"].fillna(test_data["time_zone"].mode()[0], inplace = True) 


X_train, X_valid, y_train, y_valid = train_test_split(train_data, y,train_size=0.8, test_size=0.2,random_state=42) 

from catboost import Pool, CatBoostClassifier

#cat_features = [0,1]
cat_features=['location_id','language_id','gender','tour_week','invited','friends_with_org','cluster'
             ,'tour_month','tour_quarter','tour_day']

train_label = y_train
eval_label = y_valid


train_dataset = Pool(data=X_train,label=train_label,cat_features=cat_features)

eval_dataset = Pool(data=X_valid,label=eval_label,cat_features=cat_features)

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=5000,
                           learning_rate=0.01,
                           depth=8,loss_function='Logloss',eval_metric='AUC',use_best_model=True,random_seed=42
                          ,l2_leaf_reg=8)

model.fit(train_dataset,plot=True,eval_set=eval_dataset)
pred_val_prob=model.predict_proba(X_valid)
pred_test_prob=model.predict_proba(test_data)
pred_val=model.predict(X_valid)
pred_test=model.predict(test_data)

test_ids=test_data1[['tour_id','biker_id']]
test_ids['likeliness']=pred_test_prob[:,1]
test_ids.sort_values(['biker_id', 'likeliness'], ascending=[True, False],inplace=True)
df=test_ids.groupby('biker_id')

bikers=[]
tour_id=[]
for key,value in df:
    bikers.append(key)
    tour=[]
    for j in value['tour_id']:
        tour.append(j)
    tour_id.append(tour)
    
tour=[]
for i in tour_id:
    tour.append(" ".join(i))

submission= pd.DataFrame(list(zip(bikers, tour)),columns =['biker_id', 'tour_id'])

test_ids.sort_values(['biker_id', 'tour_id'], ascending=[True, True],inplace=True)

test_id_cat1=test_ids.copy()


# In[34]:


test_id=test_id_cat1.copy()
pred=(test_id_cat1['likeliness']+test_id_lgb['likeliness'])/2

test_id['likeliness']=pred

test_id.sort_values(['biker_id', 'likeliness'], ascending=[True, False],inplace=True)
df=test_id.groupby('biker_id')

bikers=[]
tour_id=[]
for key,value in df:
    bikers.append(key)
    tour=[]
    for j in value['tour_id']:
        tour.append(j)
    tour_id.append(tour)
    
tour=[]
for i in tour_id:
    tour.append(" ".join(i))

submission= pd.DataFrame(list(zip(bikers, tour)),columns =['biker_id', 'tour_id'])
submission.to_csv('CE18B063_CE18B016_2.csv',index=False)


# In[ ]:





# In[ ]:




