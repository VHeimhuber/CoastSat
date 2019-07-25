# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:32:01 2018

@author: z5025317
"""

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import glob


def compare_images(im1, im2):
    """plots 2 images next to each other, sharing the axis"""
    plt.figure()
    ax1 = plt.subplot(121)
    plt.imshow(im1, cmap='gray')
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    plt.imshow(im2, cmap='gray')
    plt.show()  
    
def reject_outliers(data, m=2):
    "rejects outliers in a numpy array"
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def duplicates_dict(lst):
    "return duplicates and indices"
    # nested function
    def duplicates(lst, item):
            return [i for i, x in enumerate(lst) if x == item]
        
    return dict((x, duplicates(lst, x)) for x in set(lst) if lst.count(x) > 1)

def datenum2datetime(datenum):
    "convert datenum to datetime"
    #takes in datenum and outputs python datetime
    time = [datetime.fromordinal(int(dn)) + timedelta(days=float(dn)%1) - timedelta(days = 366) for dn in datenum]
    return time


def select_min_med_max_dif_model(NARCLIM_df):
    #Select the 3 most representative models (min med and max difference betwen far future and present)
    Fdf_1900_2080_sorted = NARCLIM_df.reindex_axis(sorted(NARCLIM_df.columns), axis=1)
    Fdf_1900_2080_sorted_means = pd.DataFrame(Fdf_1900_2080_sorted.mean())
    df = Fdf_1900_2080_sorted_means
    #add a simple increasing integer index 
    df = df.reset_index()
    df= df[df.index % 3 != 1]
    df['C'] = df[0].diff()
    df = df.reset_index()
    df= df[df.index % 2 != 0]
    #get max difference model (difference between far future and prsent day)
    a = df[df.index == df['C'].argmax(skipna=True)]
    Max_dif_mod_name = a.iloc[0]['index']
    #get min difference model
    a = df[df.index == df['C'].argmin(skipna=True)]
    Min_dif_mod_name = a.iloc[0]['index']
    #get the model which difference is closest to the median difference
    df['D'] = abs(df['C']- df['C'].median())
    a = df[df.index == df['D'].argmin(skipna=True)]
    Med_dif_mod_name = a.iloc[0]['index']
    #data frame with min med and max difference model
    df2 = NARCLIM_df.filter(regex= Min_dif_mod_name[:-5] + '|' +  Med_dif_mod_name[:-5] + '|' +  Max_dif_mod_name[:-5] )
    dfall = df2.reindex_axis(sorted(df2.columns), axis=1)
    #data frame with individual models
    dfmin = NARCLIM_df.filter(regex= Min_dif_mod_name[:-5])
    dfmax = NARCLIM_df.filter(regex= Max_dif_mod_name[:-5])
    dfmed = NARCLIM_df.filter(regex= Max_dif_mod_name[:-5])
    return dfall , dfmin, dfmed, dfmax, Min_dif_mod_name,Med_dif_mod_name, Max_dif_mod_name


def calculate_deltas_NF_FF2(Annual_df, Seasonal_df, Stats, Perc_vs_Abs):
    """calculates the "deltas" between nearfuture and present day for annual or seasonal climate data in pandas TS format"""
    times = ['annual', 'DJF', 'MAM', 'JJA','SON']
    delta_all_df = pd.DataFrame()
    for temp in times:
        if temp == 'annual':
            Mean_df = Annual_df.mean()
            Column_names = ['near', 'far']
        if temp == 'DJF':
            Mean_df = Seasonal_df[Seasonal_df.index.quarter==1].mean()
            Column_names = ['DJF_near', 'DJF_far']
        if temp == 'MAM':
            Mean_df = Seasonal_df[Seasonal_df.index.quarter==2].mean()
            Column_names = ['MAM_near', 'MAM_far']
        if temp == 'JJA':
            Mean_df = Seasonal_df[Seasonal_df.index.quarter==3].mean()
            Column_names = ['JJA_near', 'JJA_far']
        if temp == 'SON':
            Mean_df = Seasonal_df[Seasonal_df.index.quarter==4].mean()
            Column_names = ['SON_near', 'SON_far']
        if(Stats[:4] =='days'):
            models = list(Seasonal_df.mean().index.get_level_values(0))
        else:
            models = list(Seasonal_df.mean().index)
        newmodel = []
        for each in models:
            newmodel.append(each[:-5])
        unique_models = set(newmodel)
        # calculate diff for each unique model
        delta_NF_ensemble = []
        delta_FF_ensemble = []
        for unique_model in unique_models:
            dfdiff = Mean_df.filter(regex= unique_model)
            type(dfdiff)
            if Perc_vs_Abs == 'absolute':
                delta_NF = dfdiff[1] - dfdiff[0]
                delta_NF_ensemble.append(delta_NF)
                delta_FF = dfdiff[2] - dfdiff[0]
                delta_FF_ensemble.append(delta_FF)
            if Perc_vs_Abs == 'percent':
                delta_NF = ((dfdiff[1] - dfdiff[0])/dfdiff[0])*100
                delta_NF_ensemble.append(delta_NF)
                delta_FF = ((dfdiff[2] - dfdiff[0])/dfdiff[0])*100
                delta_FF_ensemble.append(delta_FF)
    
        delta_df1=pd.DataFrame(delta_NF_ensemble, index=unique_models)
        delta_df2=pd.DataFrame(delta_FF_ensemble, index=unique_models)
        delta_df= pd.concat([delta_df1, delta_df2], axis=1)
    
        #rename columns
        delta_df.columns = Column_names
        #add a row with medians and 10 and 90th percentiles
        delta_df.loc['10th'] = pd.Series({Column_names[0]:np.percentile(delta_df[Column_names[0]], 10), Column_names[1]:np.percentile(delta_df[Column_names[1]], 10)})
        delta_df.loc['median'] = pd.Series({Column_names[0]:np.percentile(delta_df[Column_names[0]], 50), Column_names[1]:np.percentile(delta_df[Column_names[1]], 50)})
        delta_df.loc['90th'] = pd.Series({Column_names[0]:np.percentile(delta_df[Column_names[0]], 90), Column_names[1]:np.percentile(delta_df[Column_names[1]], 90)})
        #append df to overall df
        delta_all_df = pd.concat([delta_all_df, delta_df], axis=1)
        if(Stats[:4] =='days'):
            delta_all_df  = delta_all_df .astype(int).fillna(0.0)
    return delta_all_df

def calculate_deltas_monthly(Monthly_df, Stats, Perc_vs_Abs):  
    """calculates the "deltas" between nearfuture and present day for annual or seasonal climate data in pandas TS format"""   
    delta_all_df = pd.DataFrame()
    for i in range(1, 13, 1):
        Mean_df = Monthly_df[Monthly_df.index.month==i].mean()
        Column_names = [str(i)+'_near', str(i)+'_far']
        if(Stats[:4] =='days'):
            models = list(Monthly_df.mean().index.get_level_values(0))
        else:
            models = list(Monthly_df.mean().index)
        newmodel = []
        for each in models:
            newmodel.append(each[:-5])
        unique_models = set(newmodel)
        # calculate diff for each unique model
        delta_NF_ensemble = []
        delta_FF_ensemble = []
        for unique_model in unique_models:
            dfdiff = Mean_df.filter(regex= unique_model)
            type(dfdiff)
            if Perc_vs_Abs == 'absolute':
                delta_NF = dfdiff[1] - dfdiff[0]
                delta_NF_ensemble.append(delta_NF)
                delta_FF = dfdiff[2] - dfdiff[0]
                delta_FF_ensemble.append(delta_FF)
            if Perc_vs_Abs == 'percent':
                delta_NF = ((dfdiff[1] - dfdiff[0])/dfdiff[0])*100
                delta_NF_ensemble.append(delta_NF)
                delta_FF = ((dfdiff[2] - dfdiff[0])/dfdiff[0])*100
                delta_FF_ensemble.append(delta_FF)
    
        delta_df1=pd.DataFrame(delta_NF_ensemble, index=unique_models)
        delta_df2=pd.DataFrame(delta_FF_ensemble, index=unique_models)
        delta_df= pd.concat([delta_df1, delta_df2], axis=1)
    
        #rename columns
        delta_df.columns = Column_names
        #add a row with medians and 10 and 90th percentiles
        delta_df.loc['10th'] = pd.Series({Column_names[0]:np.percentile(delta_df[Column_names[0]], 10), Column_names[1]:np.percentile(delta_df[Column_names[1]], 10)})
        delta_df.loc['median'] = pd.Series({Column_names[0]:np.percentile(delta_df[Column_names[0]], 50), Column_names[1]:np.percentile(delta_df[Column_names[1]], 50)})
        delta_df.loc['90th'] = pd.Series({Column_names[0]:np.percentile(delta_df[Column_names[0]], 90), Column_names[1]:np.percentile(delta_df[Column_names[1]], 90)})
        #append df to overall df
        delta_all_df = pd.concat([delta_all_df, delta_df], axis=1)
        if(Stats[:4] =='days'):
            delta_all_df  = delta_all_df .astype(int).fillna(0.0)
    return delta_all_df
    
def import_present_day_climdata_csv(Estuary, Clim_var_type):
    """
    this funciton imports the present day climate data used for 
    characterizing the present day climate varibility
   
    If DataSource == 'Station', individual weather station data is used. 
    If DataSource == 'SILO' , SILO time series is used using the estuary centerpoint as reference locatoin for 
    selection of the grid cell
    """ 
    #load present day climate data for the same variable
    if Clim_var_type == 'evspsblmean': #ET time series that we have is not in the same format as the other variables, hence the different treatment
        Present_day_Var_CSV = glob.glob('./Data/Wheather_Station_Data/**/' + Estuary + '_' +  'ET' + '*csv')
        Present_day_df = pd.read_csv(Present_day_Var_CSV[0])
        Dates = pd.to_datetime(Present_day_df.Date)
        Present_day_df.index = Dates
        Present_day_df = Present_day_df.iloc[:,1]
        Present_day_df = Present_day_df.replace(r'\s+', np.nan, regex=True)
        Present_day_df = pd.to_numeric(Present_day_df)
        Present_day_df.index = Dates
        [minplotDelta, maxplotDelta]=[50,50]
    #for tasmean, observed min and max T need to be converted into mean T
    elif Clim_var_type == 'tasmean':
        Present_day_Var_CSV = glob.glob('./Data/Wheather_Station_Data/**/' + Estuary + '_MaxT*csv')
        Present_day_df = pd.read_csv(Present_day_Var_CSV[0])
        Dates = pd.to_datetime(Present_day_df.Year*10000+Present_day_df.Month*100+Present_day_df.Day,format='%Y%m%d')
        Present_day_df.index = Dates
        Present_day_MaxT_df = Present_day_df.iloc[:,5]
        Present_day_Var_CSV = glob.glob('./Data/Wheather_Station_Data/**/' + Estuary + '_MinT*csv')
        Present_day_df = pd.read_csv(Present_day_Var_CSV[0])
        Dates = pd.to_datetime(Present_day_df.Year*10000+Present_day_df.Month*100+Present_day_df.Day,format='%Y%m%d')
        Present_day_df.index = Dates
        Present_day_MinT_df = Present_day_df.iloc[:,5]
        Present_day_df = (Present_day_MaxT_df + Present_day_MinT_df)/2
        [minplotDelta, maxplotDelta]=[1,2]
    elif Clim_var_type == 'tasmax':
        Present_day_Var_CSV = glob.glob('./Data/Wheather_Station_Data/**/' + Estuary + '_MaxT*csv')
        Present_day_df = pd.read_csv(Present_day_Var_CSV[0])
        Dates = pd.to_datetime(Present_day_df.Year*10000+Present_day_df.Month*100+Present_day_df.Day,format='%Y%m%d')
        Present_day_df.index = Dates
        Present_day_df = Present_day_df.iloc[:,5]
        [minplotDelta, maxplotDelta]=[1,2]  
    elif Clim_var_type == 'wssmean' or  Clim_var_type == 'wss1Hmaxtstep':
        Present_day_Var_CSV = glob.glob('./Data/Wheather_Station_Data/**/Terrigal_Wind.csv')
        Present_day_df = pd.read_csv(Present_day_Var_CSV[0])
        Present_day_df.index = Present_day_df[['Year', 'Month', 'Day', 'Hour']].apply(lambda s : datetime(*s),axis = 1)
        Present_day_df = Present_day_df.filter(regex= 'm/s') 
        Present_day_df = Present_day_df.replace(r'\s+', np.nan, regex=True)
        Present_day_df['Wind speed measured in m/s'] = Present_day_df['Wind speed measured in m/s'].convert_objects(convert_numeric=True)
        [minplotDelta, maxplotDelta]=[1, 1.5]  
    elif Clim_var_type == 'sstmean':
        Estuary_Folder = glob.glob('./Data/NARCLIM_Site_CSVs/CASESTUDY2/' + Estuary + '*' )
        Present_day_Var_CSV = glob.glob(Estuary_Folder[0] + '/' + Clim_var_type + '_NNRP*')
        Present_day_df = pd.read_csv(Present_day_Var_CSV[0], parse_dates=True, index_col=0)
        Present_day_df = Present_day_df.filter(regex= 'NNRP_R1_1950') 
        Present_day_df['NNRP_R1_1950'] = Present_day_df['NNRP_R1_1950'].convert_objects(convert_numeric=True)
        [minplotDelta, maxplotDelta]=[1, 1]
    else:   
        Present_day_Var_CSV = glob.glob('./Data/Wheather_Station_Data/**/' + Estuary + '_' +  'Rainfall' + '*csv')
        Present_day_df = pd.read_csv(Present_day_Var_CSV[0])
        Dates = pd.to_datetime(Present_day_df.Year*10000+Present_day_df.Month*100+Present_day_df.Day,format='%Y%m%d')
        Present_day_df.index = Dates
        Present_day_df = Present_day_df.iloc[:,5]
        [minplotDelta, maxplotDelta]=[50,100]    
    return Present_day_df, minplotDelta, maxplotDelta


def quant_quant_scaling(Full_df, quantiles, Plotbool):
    """
    calculates the % "deltas" for each quantile in line with Climate Change in Australia recommendations provided here:
    https://www.climatechangeinaustralia.gov.au/en/support-and-guidance/using-climate-projections/application-ready-data/scaling-methods/    
    """
    Periods = ['1990', '2020', '2060']
    quantiles_srings = [str(x) for x in quantiles][:-1]
    models = list(Full_df.resample('A').max().mean().index)
    newmodel = []
    for each in models:
        newmodel.append(each[:-5])
    unique_models = list(set(newmodel))
    #create empty df and loop through models and periods to derive the % change factors for all quantiles and models
    Quant_diff_df_outer = pd.DataFrame()
    for unique_model in unique_models:
        dfdiff = Full_df.filter(regex= unique_model)
        Quant_diff_df = pd.DataFrame() 
        for period in Periods:
            x=dfdiff.filter(regex= period).dropna().values
            x.sort(axis=0)
            df=pd.DataFrame(x)
            df.columns = ['A']
            cut_df = pd.DataFrame(df.groupby(pd.qcut(df.rank(method='first').A, quantiles))['A'].mean().values)
            cut_df.columns = [period]
            Quant_diff_df = pd.concat([Quant_diff_df, cut_df], axis=1, join='outer')  
        if Plotbool:
            Quant_diff_df.plot(x=Quant_diff_df.index, y=Periods, kind="bar", title = unique_model)
        Quant_diff_df['NF_%change_'+ unique_model] =  (Quant_diff_df['2020'] - Quant_diff_df['1990'])/Quant_diff_df['1990']*100
        Quant_diff_df['FF_%change_'+ unique_model] =  (Quant_diff_df['2060'] - Quant_diff_df['1990'])/Quant_diff_df['1990']*100
        Quant_diff_df = Quant_diff_df.replace([np.inf, -np.inf], np.nan)
        Quant_diff_df = Quant_diff_df.iloc[:,[3,4]].fillna(0)
        Quant_diff_df_outer = pd.concat([Quant_diff_df_outer, Quant_diff_df], axis=1, join='outer')
    Quant_diff_df_outer.index = quantiles_srings 
    if Plotbool:
        Quant_diff_df_outer.plot(x=Quant_diff_df_outer.index, y=Quant_diff_df_outer.columns, kind="bar", legend = False, title='% intensity change per quantile')
    #add new cols and rows with summary statistics
    Quantile_change_NF_df = Quant_diff_df_outer.filter(regex= 'NF')
    Quantile_change_FF_df = Quant_diff_df_outer.filter(regex= 'FF')
    Quantile_change_stats_df = pd.DataFrame()
    for loop_df in [Quantile_change_NF_df, Quantile_change_FF_df]:
        Sum95_df = pd.DataFrame(loop_df.iloc[-5:,:].sum()).transpose()
        Sum95_df.index = ['Sum>95perc']
        Sum_df = pd.DataFrame(loop_df.sum()).transpose()
        Sum_df.index = ['Sum']
        Med_df = pd.DataFrame(loop_df.median()).transpose()
        Med_df.index = ['Median']
        loop_df = loop_df.append(Sum_df)
        loop_df = loop_df.append(Sum95_df)
        loop_df = loop_df.append(Med_df)
        loop_df['Median'] = loop_df.median(axis=1)
        loop_df['Maximum'] = loop_df.max(axis=1)
        loop_df['Minimum'] = loop_df.min(axis=1)
        Quantile_change_stats_df = pd.concat([Quantile_change_stats_df, loop_df], axis=1, join='outer')
    return Quantile_change_stats_df
        













