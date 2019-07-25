# -*- coding: utf-8 -*-

import requests
import pandas as pd
from io import StringIO


def pointdata(variables,
              api_key,
              start,
              finish,
              station=None,
              lat=None,
              lon=None,
              units=True,
              output=None):
    """Request point data from SILO.

    Args:
        variables: list of variable code strings (see variable info below)
        api_key:   SILO api key
        start:     start date (yyyymmdd, earliest date is '18890101')
        finish:    finish date (yyyymmdd, latest date is yesterday)
        station:   weather station ID (or use 'lat' and 'lon')
        lat:       latitude (between -44° and -10°, in increments of 0.05°)
        lon:       longitude (between 112° and 154°, in increments of 0.05°)
        units:     include units in dataframe column names
        output:    name of file to export data (csv format)

    Returns:
        Dataframe containing climate data time series


    API tutorial:
    https://silo.longpaddock.qld.gov.au/api-documentation/tutorial

    API documentation:
    https://silo.longpaddock.qld.gov.au/api-documentation/reference

    API key registration:
    https://silo.longpaddock.qld.gov.au/register

    Variable info:

    Code                 Units    Name
    ---------------------------------------------------------------------------
    daily_rain           mm       Daily rainfall
    monthly_rain         mm       Monthly rainfall
    max_temp             Celsius  Maximum temperature
    min_temp             Celsius  Minimum temperature
    vp                   hPa      Vapour pressure
    vp_deficit           hPa      Vapour pressure deficit
    evap_pan             mm       Evaporation - Class A pan
    evap_syn             mm       Evaporation - synthetic estimate
    evap_comb            mm       Evaporation - combination (synthetic estimate
                                  pre-1970, class A pan 1970 onwards)
    evap_morton_lake     mm       Evaporation - Morton's shallow lake evap.
    radiation            MJm-2    Solar radiation - total incoming downward
                                  shortwave radiation on a horizontal surface
    rh_tmax              %        Relative humidity at the time of max. temp.
    rh_tmin              %        Relative humidity at the time of min. temp.
    et_short_crop        mm       Evapotranspiration - FAO56 short crop
    et_tall_crop         mm       Evapotranspiration - ASCE tall crop
    et_morton_actual     mm       Evapotranspiration - Morton's areal actual
                                  evapotranspiration
    et_morton_potential  mm       Evapotranspiration - Morton's potential
                                  evapotranspiration
    et_morton_wet        mm       Evapotranspiration - Morton's wet-environment
                                  areal evapotranspiration over land
    mslp                 hPa      Mean sea level pressure

    """

    unit_defs = {
        'daily_rain': 'mm',
        'monthly_rain': 'mm',
        'max_temp': 'Celsius',
        'min_temp': 'Celsius',
        'vp': 'hPa',
        'vp_deficit': 'hPa',
        'evap_pan': 'mm',
        'evap_syn': 'mm',
        'evap_comb': 'mm',
        'evap_morton_lake': 'mm',
        'radiation': 'MJm-2',
        'rh_tmax': '%',
        'rh_tmin': '%',
        'et_short_crop': 'mm',
        'et_tall_crop': 'mm',
        'et_morton_actual': 'mm',
        'et_morton_potential': 'mm',
        'et_morton_wet': 'mm',
        'mslp': 'hPa',
    }

    # Validate inputs
    if (type(lat) and type(lon)) == type(station):
        raise ValueError(
            "'lat' and 'lon' must be provided if 'station' is not specified.")

    params = {
        'apikey': api_key,
        'format': 'csv',
        'start': start,
        'finish': finish,
        'variables': ','.join(variables)
    }

    if station is not None:
        # Get observations from specified weather station
        params['station'] = station
    else:
        # Get observations from specified coordinates
        params['lat'] = lat
        params['lon'] = lon

    base_url = 'https://siloapi.longpaddock.qld.gov.au/pointdata'
    r = requests.get(base_url, params=params)
    text = r.content.decode()
    df = pd.read_csv(StringIO(text), parse_dates=["date"])
    #df = pd.read_csv(StringIO(text))
    df = df.set_index("date")

    # Add units to columns names
    if units:
        labels = {}
        for key, val in unit_defs.items():
            labels[key] = '{}_{}'.format(key, val)
        df = df.rename(columns=labels)

    # Write to csv
    if output:
        df.to_csv(output)

    return df
