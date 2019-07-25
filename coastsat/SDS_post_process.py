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
import pandas as pd
from sklearn import tree
import sklearn
import graphviz 
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#sklearn.cross_validation.cross_val_score

#to make graphviz binaries work on windows- one needs to append the location of the executables to the system path (only once)
#os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\bin\\'
#os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\bin\\dot.exe'


# name of the site
sitename = 'DURRAS'

# create a subfolder to store the .jpg images showing the detection
out_path = os.path.join(os.getcwd(), 'data',sitename,  'results')
        

#load ICOLL entrance RS processed data time series from CSV
path= os.path.join(os.getcwd(), 'data', sitename, 'results',  sitename + '__entrance_stats.csv')  #you need to adjust the directory here
files=glob.glob(path)
result_df = pd.read_csv(files[0], index_col=[0],  header=0) #index_col=0,
result_df.index = pd.to_datetime(result_df.index, format='%Y-%m-%d-%H-%M-%S').normalize()

#load ICOLL validation data from CSV
path= os.path.join(os.getcwd(), 'validation', sitename + '_val_data.csv')  #you need to adjust the directory here
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
result_df = result_df.join(OC_df)

mapping = {'open': 1, 'closed': -1}
result_df = result_df.replace({'OTSU_ndwi_ful': mapping, 'OTSU_ndwi_ent': mapping, 'known_state':mapping})


################################
#Fit the decision tree
################################

#Prepare the data and fit the regression tree
Y = result_df.iloc[:,-1]  #.values
X = result_df.iloc[:,-6:-1]  #.values

depth = []
for i in range(1,20):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=clf, X=X, y=Y, cv=7, scoring='accuracy')
    #All scorer objects follow the convention that higher return values are better than lower return values. 
    #https://scikit-learn.org/stable/modules/model_evaluation.html
    depth.append((i,scores.mean()))

plt.figure()
plt.plot([i[0] for i in depth], [i[1] for i in depth])
plt.axis("tight")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
opt_depth = depth[np.argmax([i[1] for i in depth])]
plt.plot(opt_depth[0] ,np.amax([i[1] for i in depth]),"or")


#fit the optimum depth decision tree and output graphically
clf = tree.DecisionTreeClassifier(max_depth=opt_depth[0])
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=0)
clf = clf.fit(X=x_train, y=y_train)

#plot the decision tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=result_df.columns[-6:-1] ,  
                     #class_names=result_df.columns[-1],
                     class_names = ['closed', 'open'],  
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render(os.path.join(out_path, sitename + '_decision_tree_optDepth9')) 

result_df['Tree_predicted'] = clf.predict(X.values)
################################




################################
####plot SWE extent time series from CSV
fig = plt.figure(figsize=(25,15))
ax=plt.subplot(2,1,1)
result_df['SWE'].plot(kind='line', x=result_df.index, stacked=True, ax=ax)
ax.xaxis.grid(False)
plt.title(sitename + ' Surface Water Extent in square km') 

ax=plt.subplot(2,1,2)
#result_df['Tree_predicted'].interpolate(method='linear').plot(kind='line', x=result_df.index,  ax=ax)
ax.plot(result_df.index, result_df['Tree_predicted'], label="predicted")
ax.plot(result_df.index, result_df['known_state'], label="known")
ax.plot(result_df.index, result_df['OTSU_ndwi_ful'], label="OTSU",linestyle='--')
ax.xaxis.grid(False)
ax.legend()
plt.title(sitename + ' Open vs closed') 

fig.tight_layout()
plt.rcParams['savefig.jpeg_quality'] = 100
fig.savefig(os.path.join(out_path, sitename + '_SWE_open_vs_closed.pdf') , dpi=150)
plt.close()
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
    

