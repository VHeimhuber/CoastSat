# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:38:46 2019

@author: Valentin Heimhuber

Code for postprocessing the remote sensing results for ICOLL entrances
"""

#load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn import tree
import sklearn
import graphviz 
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from coastsat import SDS_tools #SDS_download,
import geopandas as gpd
#sklearn.cross_validation.cross_val_score
#to make graphviz binaries work on windows- one needs to append the location of the executables to the system path (only once)
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\bin\\'
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\bin\\dot.exe'

ALPHA_figs = 0 
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)


#Choose validation data type
val_type = 'Manual'
Version = 'V2'           

# name of the site
sitename = 'CATHIE'

# filepath where data will be stored
filepath_data = os.path.join('H:/WRL_Projects/Estuary_sat_data')

# create a subfolder to store the .jpg images showing the detection
out_path = os.path.join(filepath_data, 'data',sitename,  'results_' + Version)
if not os.path.exists(out_path):
    os.makedirs(out_path)

#load ICOLL entrance RS processed data time series from CSV
path = os.path.join(filepath_data, 'data', sitename, 'results_V2',  sitename + '_S2_entrance_stats.csv')  #you need to adjust the directory here

#files=glob.glob(path)
result_df = pd.read_csv(path, index_col=[0],  header=0) #index_col=0,
dates = pd.to_datetime(result_df.index, format='%Y-%m-%d-%H-%M-%S')
result_df.index = pd.to_datetime(result_df.index, format='%Y-%m-%d-%H-%M-%S').normalize()
#result_df = result_df.iloc[:,:-4] 

#load and refine node vs mesh shapefile to store the results of the statistics
Allsites = gpd.read_file(os.path.join(os.getcwd(), 'Sites', 'All_sites.shp')).iloc[:,-4:]
Site_shps = Allsites.loc[(Allsites.Sitename==sitename)]
BBX_coords = []
for b in Site_shps.loc[(Site_shps.layer=='ocean_seed')].geometry.boundary:
    coords = np.dstack(b.coords.xy).tolist()
    BBX_coords.append(*coords) 
    
#get tide data for each image
TideCoords = BBX_coords[0][0] 
tide = SDS_tools.compute_tide_dates(TideCoords, dates)
result_df['tide'] = (tide)

#load validation data
if val_type == 'OEH':
    #load ICOLL validation data from CSV
    path= os.path.join(filepath_data, 'validation', sitename + '_val_data.csv')  #you need to adjust the directory here
    files=glob.glob(path)
    Val_df= pd.read_csv(files[0],  header=0) #index_col=0,
    #do manual subsetting and create a daily series of open or closed
    Val_OC_df = Val_df.iloc[0:11,0:2]
    OC_df = pd.DataFrame()
    for i in range(0,len(Val_OC_df.iloc[:,0])):
        print(i)
        open_dates =pd.date_range(start=Val_OC_df.iloc[i][0], end=Val_OC_df.iloc[i][1] , freq='D')
        print(Val_OC_df.iloc[i][0], Val_OC_df.iloc[i][1] )
        x = pd.DataFrame(data=['open']*len(open_dates), index= open_dates, columns=['known_state'])
        OC_df = OC_df.append(x)
    #remote duplicates and create a daily series of open vs. closed
    OC_df = OC_df[~OC_df.index.duplicated()]
    OC_df = OC_df.asfreq(freq='D', fill_value='closed')
else:
    filepath = glob.glob(os.path.join(filepath_data, 'validation','Han')  + '/*'+ sitename + '*')
    Val_df = pd.read_csv(filepath[0], index_col=[0],  header=0) #index_col=0,
    Val_df = Val_df.iloc[:,-1] 
    Val_df.index = pd.to_datetime(Val_df.index, format='%Y-%m-%d-%H-%M-%S').normalize()

#merge data frames

result_df_full = result_df.join(Val_df)
result_df = result_df.join(Val_df, how='inner')

#replace open and closed with 1 and -1
mapping = {'open': 1, 'closed': -1}
result_df = result_df.replace({'OTSU_ndwi_ful': mapping, 'OTSU_ndwi_ent': mapping, 'observed_state':mapping, 'OTSU_mndwi_ent':mapping, 'CFD_0_mndwi_open':mapping, 'NN_CFD':mapping})
result_df_full = result_df_full.replace({'OTSU_ndwi_ful': mapping, 'OTSU_ndwi_ent': mapping, 'observed_state':mapping, 'OTSU_mndwi_ent':mapping, 'CFD_0_mndwi_open':mapping, 'NN_CFD':mapping})

result_df_full.to_csv(os.path.join(out_path, sitename + '_full_ICOLL_XY_dataset.csv'))

################################
#Fit the decision tree
################################

#Prepare the data and fit the regression tree
Y = result_df.iloc[:,-1]  #.values
X = result_df.iloc[:,1:-1]  #result_df.filter(regex= 'tol') #result_df.iloc[:,-4:-1]  #.values
feature_names = X.columns

depth = []
for i in range(1,20):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=clf, X=X, y=Y, cv=7, scoring='accuracy')
    #All scorer objects follow the convention that higher return values are better than lower return values. 
    #https://scikit-learn.org/stable/modules/model_evaluation.html
    depth.append((i,scores.mean()))

fig = plt.figure(figsize=(10,10))
ax=plt.subplot(1,1,1)
#plt.figure()
ax.plot([i[0] for i in depth], [i[1] for i in depth])
ax.axis("tight")
#ax.xlabel("Tree Depth")
#ax.ylabel("Accuracy")
opt_depth = depth[np.argmax([i[1] for i in depth])]
ax.plot(opt_depth[0] ,np.amax([i[1] for i in depth]),"or")
fig.tight_layout()
fig.savefig(os.path.join(out_path, sitename + '_decision_tree_depth_optimizer.pdf') , dpi=150)

#fit the optimum depth decision tree and output graphically
clf = tree.DecisionTreeClassifier(max_depth=opt_depth[0])
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=0)
clf = clf.fit(X=x_train, y=y_train)
#plot the decision tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=X.columns,  
                     #class_names=result_df.columns[-1],
                     class_names = ['closed', 'open'],  
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render(os.path.join(out_path, sitename + '_decision_tree_optDepth')) 

#use tree to predict each timestep
result_df_full['Opt_Tree_predicted'] = clf.predict(result_df_full.iloc[:,1:-1].values)


#fit a 4 layer depth decision tree and output graphically
clf = tree.DecisionTreeClassifier(max_depth=4) #opt_depth[0])
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=0)
clf = clf.fit(X=x_train, y=y_train)
#plot the decision tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=X.columns,  
                     #class_names=result_df.columns[-1],
                     class_names = ['closed', 'open'],  
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render(os.path.join(out_path, sitename + '_decision_tree_4 layers')) 

#use tree to predict each timestep and save result df as csv
result_df_full['4lay_Tree_predicted'] = clf.predict(result_df_full.iloc[:,1:-2].values)
result_df_full.to_csv(os.path.join(out_path, sitename + '_decision_tree_predicted_results.csv'))

#plot the distribution of the predictors in the form of a violin plot
fig = plt.figure(figsize=(20,10))
plt.violinplot(X.T)
#plt.xticks(range(1,X.shape[1]+1), X.columns, rotation='vertical')
plt.xticks(range(1,30), X.columns, rotation='vertical')
plt.ylim(-1,3)
fig.tight_layout()
#plt.show()
fig.savefig(os.path.join(out_path, sitename + '_predictor_distribution.pdf'))
plt.close(fig) 
plt.close()

################################


################################
####plot SWE extent time series from CSV
p = pd.Timedelta(30, unit='s')
#replace open and closed with 1 and -1
mapping = {11:2}
result_df_full = result_df_full.replace({'NIR_RG_tol': mapping, 'NIR_RG_tol_eo': mapping})

for decade in [1985, 2000]:
    #decade = 1997
    result_df_decadal = result_df_full.loc[(result_df_full.index > str(decade) + '-01-01') & (result_df_full.index < str(decade+20) + '-01-01')]
    df = result_df_decadal
    
    fig = plt.figure(figsize=(25,15))
    ax=plt.subplot(3,1,1)
    result_df_decadal['SWE_NN'].plot(kind='line', color= 'slateblue', x=result_df_decadal.index, stacked=True, ax=ax, ylim=(np.min(result_df_full['SWE_NN']),np.max(result_df_full['SWE_NN'])))
    #df['SWE'].plot()
    for (observed_state, _), g in df.groupby(['observed_state', df.observed_state.ne(-1).cumsum()]):
        if  observed_state == -1:
            start = g.index.min() - p
            end = g.index.max() + p
            plt.axvspan(start, end, color='grey', alpha=0.2)
    ax.xaxis.grid(False)
    ax.legend(loc='upper left')
    #plt.text(result_df_decadal.index[5], np.min(result_df_decadal['SWE']), "grey = closed", size=16, ha="center", va="center")
    plt.title(sitename + ' Surface Water Extent in square km')         
        
    ax=plt.subplot(3,1,2)
#    #result_df_decadal['Tree_predicted'].interpolate(method='linear').plot(kind='line', x=result_df_decadal.index,  ax=ax)
#    ax.plot(result_df_decadal.index, result_df_decadal['Tree_predicted'], label="predicted")
#    ax.plot(result_df_decadal.index, result_df_decadal['observed_state'], label="known")
#    ax.plot(result_df_decadal.index, result_df_decadal['OTSU_ndwi_ful'], label="OTSU",linestyle='--')
#    ax.xaxis.grid(False)
#    ax.legend()
#    plt.title(sitename + ' Open vs closed') 

    #result_df_decadal['Tree_predicted'].interpolate(method='linear').plot(kind='line', x=result_df_decadal.index,  ax=ax)
    #ax.plot(result_df_decadal.index, result_df_decadal['NIR_RG_tol'], label="NIR region growing tolerance needed for floodfill to pass entrance")
    result_df_decadal['fff_tol_NIR'].plot(kind='line',  color='tomato', x=result_df_decadal.index, ylim=(np.min(result_df_full['fff_tol_NIR']),np.max(result_df_full['fff_tol_NIR'])), stacked=True, ax=ax)
    result_df_decadal['fff_tol_SWIR'].plot(kind='line', color='mediumpurple', x=result_df_decadal.index,ylim=(np.min(result_df_full['fff_tol_SWIR']),np.max(result_df_full['fff_tol_SWIR'])), stacked=True, ax=ax)
    result_df_decadal['fff_tol_mndwi'].plot(kind='line', color='blue', x=result_df_decadal.index,ylim=(np.min(result_df_full['fff_tol_mndwi']),np.max(result_df_full['fff_tol_mndwi'])), stacked=True, ax=ax)
    ax.xaxis.grid(False)
    for (observed_state, _), g in df.groupby(['observed_state', df.observed_state.ne(-1).cumsum()]):
        if  observed_state == -1:
            start = g.index.min() - p
            end = g.index.max() + p
            plt.axvspan(start, end, color='grey', alpha=0.2)
    ax.legend(loc='upper left')
    #plt.text(result_df_decadal.index[5], np.min(result_df_decadal['NIR_RG_tol']), "grey = closed", size=16, ha="center", va="center")
    plt.title(sitename + ' NIR region growing tolerance needed for floodfill to pass entrance - eo is from inside the ICOLL to ocean') 
    
    ax=plt.subplot(3,1,3)
    #result_df_decadal['Tree_predicted'].interpolate(method='linear').plot(kind='line', x=result_df_decadal.index,  ax=ax)
    result_df_decadal['Green_by_red'].plot(kind='line', color='lightgreen',x=result_df_decadal.index,  ax=ax, ylim=(np.min(result_df_full['Green_by_red']),np.max(result_df_full['Green_by_red'])))
    result_df_decadal['Blue_by_red'].plot(kind='line',color='turquoise',  x=result_df_decadal.index,  ax=ax, ylim=(np.min(result_df_full['Blue_by_red']),np.max(result_df_full['Blue_by_red'])))
    #ax.plot(result_df_decadal.index, result_df_decadal['Green_by_red'], label="predicted")
    #ax.plot(result_df_decadal.index, result_df_decadal['Blue_by_red'], label="known")
    ax.xaxis.grid(False)
    for (observed_state, _), g in df.groupby(['observed_state', df.observed_state.ne(-1).cumsum()]):
        if  observed_state == -1:
            start = g.index.min() - p
            end = g.index.max() + p
            plt.axvspan(start, end, color='grey', alpha=0.2)
    ax.legend(loc='upper left')
    plt.title(sitename + ' Water Colour Indices')
    plt.text(result_df_full.index[5], np.min(result_df_full['Green_by_red']), "grey = closed", size=16, ha="center", va="center")

    fig.tight_layout()
    plt.rcParams['savefig.jpeg_quality'] = 100
    fig.savefig(os.path.join(out_path, sitename + '_'+ str(decade) + '_SWE_open_vs_closed.pdf') , dpi=150)
    plt.close()


#next steps: Cound percentage open vs. closed per year/decade
#use region growing form outside to in and from in to out! 
#refine the estuarine entrance area even further and limit the area for region growing on NIR and SWIR
#Include the SWIR2 band in the analysis also
#Plot the histograms also
################################


















## Make most of the ticklabels empty so the labels don't get too crowded
#ticklabels = ['']*len(df_ts.index)
## Every 12th ticklabel includes the year
#ticklabels[::12] = [item.strftime('%Y') for item in df_ts.index[::12]]
#ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
#plt.gcf().autofmt_xdate() 

ax=plt.subplot(4,1,3)
df_ts['sum'].plot(kind='bar', x=df_ts.index,color='grey', stacked=True, ax=ax)

ax.plot(activity, dog, label="dog")
ax.plot(activity, cat, label="cat")

ax.xaxis.grid(False)
plt.title(sitename + ' sum of NDWI values below 0.2 along the transect') 
# Make most of the ticklabels empty so the labels don't get too crowded
ticklabels = ['']*len(df_ts.index)














SWE_df.index = pd.to_datetime(SWE_df.index)
SWE_df.index = SWE_df.index.to_period('D')
SWE_df.columns = ['Satellite', 'Resolution', 'SWE', 'Error']

#load rainfall time series from SILO
from coastsat import SDS_climdata_fcts as fct
from coastsat import SDS_SILO_Rainfall as sil
SILO_Clim_var = ['daily_rain'] 
Base_period_start = str(SWE_df.index[0])       #Start of interval for base period of climate variability
Base_period_end = str(SWE_df.index[-1])
Case_Study_Name = 'CASESTUDY2' 
Casestudy2_csv_path =  'C:/Users/z5025317/OneDrive - UNSW/WRL_Postdoc_Manual_Backup/WRL_Postdoc/Projects/Paper#1/Data/NARCLIM_Site_CSVs/' + Case_Study_Name + '/' + Case_Study_Name + '_NARCLIM_Point_Sites.csv'
with open(Casestudy2_csv_path, mode='r') as infile:
    reader = csv.reader(infile)
    next(reader, None) 
    with open('coors_new.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        mydict = dict((rows[0],[rows[1],rows[2]]) for rows in reader)
silo_df = sil.pointdata(SILO_Clim_var, 'Okl9EDxgS2uzjLWtVNIBM5YqwvVcCxOmpd3nCzJh', Base_period_start.replace("-", ""), Base_period_end.replace("-", ""), 
                            None, mydict['NADGEE'][0], mydict['NADGEE'][1], False, None)
Present_day_df = silo_df.iloc[:,[2]]
Present_day_df_monthly = Present_day_df.resample('M').sum()         
 








##load the NDWI cross section transects           
pdf1=pd.DataFrame(XS)  
#save the data to csv
out_path = output_directory + sitename +'/NDWI_XS_LS5_XS2.csv'
pdf1.to_csv(out_path) 

#create a df with the sum of the absolutes of the negative NDWI values along the transect
df2 = pdf1
df2 = pd.DataFrame(np.where( df2 >=-0.2, 0, np.abs(df2)))
df3 = df2.sum(axis=0, skipna=True)
df3.index =pd.to_datetime(pdf1.columns.tolist())
df3.index = df3.index.to_period('D')
df3 = pd.DataFrame(df3)
df3.columns = ['sum']

#merge into a single data frame
R13_df = pd.concat([df3, SWE_df], axis=1)
df_ts = R13_df


############################tino's stuff
#-- Plot...+
############################tino's stuff
import matplotlib
matplotlib.style.use('ggplot')
png_out_path = output_directory + sitename +'/NDWI_Allimages3.png'
fig = plt.figure(figsize=(25,18))

####plot SWE extent time series from CSV
ax=plt.subplot(4,1,2)
df_ts['SWE'].interpolate(method='linear').plot(kind='line', x=df_ts.index, stacked=True, ax=ax)
ax.xaxis.grid(False)
plt.title(sitename + ' Surface Water Extent in square km') 
## Make most of the ticklabels empty so the labels don't get too crowded
#ticklabels = ['']*len(df_ts.index)
## Every 12th ticklabel includes the year
#ticklabels[::12] = [item.strftime('%Y') for item in df_ts.index[::12]]
#ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
#plt.gcf().autofmt_xdate() 

ax=plt.subplot(4,1,3)
df_ts['sum'].plot(kind='bar', x=df_ts.index,color='grey', stacked=True, ax=ax)
ax.xaxis.grid(False)
plt.title(sitename + ' sum of NDWI values below 0.2 along the transect') 
# Make most of the ticklabels empty so the labels don't get too crowded
ticklabels = ['']*len(df_ts.index)
# Every 4th ticklable shows the month and day
#ticklabels[::6] = [item.strftime('%b %d') for item in df_ts.index[::4]]
# Every 12th ticklabel includes the year
ticklabels[::12] = [item.strftime('%Y') for item in df_ts.index[::12]]
ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
plt.gcf().autofmt_xdate()   

ax=plt.subplot(4,1,1)
plt.title(sitename + ' NDWI transects') 
pdf1.plot(cmap='viridis', ax=ax, legend=False)
plt.ylim(-1,1)
ax.xaxis.grid(False)
plt.axhline(y=0, xmin=-1, xmax=1, color='r', linestyle='--', lw=1, alpha=0.5) 
#fig.tight_layout()

#rainfall:
ax=plt.subplot(4,1,4)
df_ts2 = Present_day_df_monthly
df_ts2['daily_rain'].plot(kind='bar', x=df_ts2.index,color='blue', stacked=True, ax=ax)
ax.xaxis.grid(False)
plt.title(sitename + ' monthly rainfall over the lagoon') 
# Make most of the ticklabels empty so the labels don't get too crowded
ticklabels = ['']*len(df_ts2.index)
# Every 4th ticklable shows the month and day
#ticklabels[::6] = [item.strftime('%b %d') for item in df_ts2.index[::4]]
# Every 12th ticklabel includes the year
ticklabels[::12] = [item.strftime('%Y') for item in df_ts2.index[::12]]
ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
plt.gcf().autofmt_xdate()


fig.savefig(png_out_path)
plt.close()
    

