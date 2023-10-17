from pystac_client import Client
import pystac
import stackstac
import requests
import boto3
from rasterio.session import AWSSession
import rasterio
import rioxarray as rio
import os
import xarray as xr
import numpy as np
import planetary_computer
import pandas as pd
from pandas.tseries.offsets import DateOffset
from datetime import datetime, timezone
from xrspatial import slope

def get_lpdaac_creds():
    
    temp_creds_req = requests.get('https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials').json()
    session = boto3.Session(aws_access_key_id=temp_creds_req['accessKeyId'], 
                        aws_secret_access_key=temp_creds_req['secretAccessKey'],
                        aws_session_token=temp_creds_req['sessionToken'],
                        region_name='us-west-2')

    rio_env = rasterio.Env(AWSSession(session),
                  GDAL_DISABLE_READDIR_ON_OPEN='TRUE',
                  GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
                  GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'))
    rio_env.__enter__()
    
    return temp_creds_req

def get_sentinel_stack(geometry, start, end,
                       cloud_cover_threshold=None,resolution=None,
                       epsg=6933):
    
    if isinstance(geometry, list):
        bbox = geometry
        
    elif isinstance(geometry, gpd.GeoDataFrame):
        bbox = geometry.to_crs('EPSG:4326').total_bounds
    else: 
        print('Please provide either a list of Lat/Lon bounding box coordinates or a GeoDataFrame as the geometry input.')
        return
    
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = Client.open(URL)
    items = catalog.search(collections=["sentinel-2-l2a"],
                           bbox=bbox,
                           datetime=start+'/'+end).item_collection()
    
    print(f'Total matches for search: {len(items)}')
    
    if len(items)==0:
        return
    
    if resolution:
        data = stackstac.stack(items, epsg=epsg, resolution=resolution)
    
    else:
        data = stackstac.stack(items, epsg=epsg)
    
    if cloud_cover_threshold:
        data = data[data["eo:cloud_cover"] < cloud_cover_threshold]
        if data.shape[0]==0:
            print('No imagery available for set cloud cover threshold. Returning None...')
            return None
        
        else:
            print('Final number of cloud-filtered scenes:',data.shape[0])  
    
    return data

def get_hls_stack(gdf, start, end, creds=False,
                  collections=['HLSL30.v2.0', 'HLSS30.v2.0'],
                  cloud_cover_threshold=None,resolution=90,
                  epsg=6933,mask_type='cloud'):
    
    '''Returns time series of HLS imagery for an arbitrary GeoDataFrame object.'''
    
    if creds:
        get_lpdaac_creds()
        
    gdf_copy = gdf.copy().to_crs('EPSG:4326')
    bbox = gdf_copy.total_bounds # need to provide lat/lon for catalog search
    
    STAC_URL = 'https://cmr.earthdata.nasa.gov/stac'
    catalog = Client.open(f"{STAC_URL}/LPCLOUD")
    
    search = catalog.search(collections = collections,
                            bbox = bbox,
                            datetime = start+'/'+end) 
 
    print('Total matches for search:',search.matched())
    
    item_list = list(search.items())
    
    ls_ids = [item for item in item_list if 'L30' in item.id] # Landsat IDs
    s_ids = [item for item in item_list if 'S30' in item.id] # Sentinel IDs

    ls_item_collection = pystac.ItemCollection(items=ls_ids)
    s_item_collection = pystac.ItemCollection(items=s_ids)

    print('Number of available Landsat scenes:', len(ls_item_collection))
    print('Number of available Sentinel scenes:', len(s_item_collection))
    
    ordered_band_names = ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2',"Fmask"]
            
    if len(ls_ids):
        ls_data = stackstac.stack(ls_item_collection, 
                                  assets=["B04", "B03", "B02","B05","B06","B07","Fmask"], 
                                  epsg=epsg, resolution=resolution) 
        ls_data["band"] = ordered_band_names # rename based on wavelength

        
    if len(s_ids): 
        s_data = stackstac.stack(s_item_collection, 
                                 assets=["B04", "B03", "B02","B8A","B11","B12","Fmask"], 
                                 epsg=epsg, resolution=resolution)
        s_data["band"] = ordered_band_names 
        
    if (len(ls_ids)>0) & (len(s_ids)>0): # if data from both sources are present... combine!
        data = xr.concat((ls_data, s_data), dim='time').sortby("time")
    
    else: # Just get the data that are present
        if len(ls_ids):
            data = ls_data
            print('Returning Landsat scenes only...')
        elif len(s_ids): 
            data = s_data
            print('Returning Sentinel scenes only...')
        else:
            print('No scenes available!')
            return None

    data = data[data['time']>np.datetime64(start)]
    
    # do some clipping to the extent of the GeoDataFrame
    if epsg==4326:
        bbox_adj = bbox
    else:
        gdf_copy_new_epsg = gdf.to_crs('EPSG:'+str(epsg)) # convert to crs of the returned imagery
        bbox_adj = gdf_copy_new_epsg.total_bounds
    
    # NOTE: sometimes the y dimension is flipped, so this clips based on the given y orientation
    if data['y'][0] < data['y'][-1]:
        data = data.sel(x=slice(bbox_adj[0],bbox_adj[2]),
                        y=slice(bbox_adj[1],bbox_adj[3])) # order: ymin, ymax
        
    if data['y'][0] > data['y'][-1]:
        data = data.sel(x=slice(bbox_adj[0],bbox_adj[2]),
                        y=slice(bbox_adj[3],bbox_adj[1])) # order: ymax, ymin
        
        
    if cloud_cover_threshold:
        data = data[data["eo:cloud_cover"] < cloud_cover_threshold]
        if data.shape[0]==0:
            print('No imagery available for set cloud cover threshold. Returning None...')
            return None
        
        else:
            print('Final number of cloud-filtered scenes:',data.shape[0])        
        
    if mask_type==None:
        return data
    
    # if mask is specified, define bitmask
    elif mask_type.lower()=='cloud':
        # Make a bitmask---when we `bitwise-and` it with the data, it leaves just the 4 bits we care about
        mask_bitfields = [0, 1, 2, 3, 4]  # cloud, adjacent to cloud shadow, shadow, snow
    
    elif mask_type.lower()=='all':
        mask_bitfields = [0, 1, 2, 3, 4, 5]  # cloud, adjacent to cloud shadow, shadow, snow
    
    # apply it
    bitmask = 0
    for field in mask_bitfields:
        bitmask |= 1 << field

    bin(bitmask)
       
    data_qa = data.sel(band="Fmask").astype("uint16")
    data_bad = data_qa & bitmask  # just look at those 4 bits
    data = data.where(data_bad == 0)  # mask pixels where any one of those bits are set
    
    return data

def calc_dnbr(gdf, fire_start, fire_end, 
              pre_offset=1,pre_offset_range=2,post_offset=2,
              gdf_id=None,cloud_cover_threshold=None,
              epsg=6933,resolution=90,
              creds=None,mask_type='cloud',
              composites=True,logging=False,log_path=None,
              write_path=None):
    
    '''Application-specific use case of get_hls_stack().
       Because API credentials are needed for access, they 
       can either be passed through or generated.
       Credentials are then returned for future use.'''
    
    if creds==None:
        creds = get_lpdaac_creds()
        exp = pd.to_datetime(creds['expiration'])
        
    else:
        creds = creds # not necessary but helps with intuitive flow of function
        exp = pd.to_datetime(creds['expiration'])
        
    
    bounds = gdf.to_crs('EPSG:4326').total_bounds
    minx, miny, maxx, maxy = bounds[0], bounds[1], bounds[2], bounds[3]

    pre_start = (pd.to_datetime(fire_start) - DateOffset(months=pre_offset+pre_offset_range)).strftime('%Y-%m')
    pre_end = (pd.to_datetime(fire_start) - DateOffset(months=pre_offset)).strftime('%Y-%m')

    post_start = (pd.to_datetime(fire_end) + DateOffset(days=1)).strftime('%Y-%m-%d')
    post_end = (pd.to_datetime(fire_end) + DateOffset(months=post_offset)).strftime('%Y-%m-%d')
    
    now = pd.to_datetime(datetime.now(timezone.utc))
    
    time_diff = (exp - now).total_seconds() / 60.0 # expressed as minutes
    
    if time_diff < 5: # less than 5 min until creds expire!
        
        print('Reaquiring temporary creds...')
        creds = get_lpdaac_creds()
        exp = pd.to_datetime(creds['expiration'])
    
    pre_fire_data = get_hls_stack(gdf,pre_start,pre_end,
                                  creds=False,epsg=epsg,resolution=resolution,
                                  cloud_cover_threshold=cloud_cover_threshold,
                                  mask_type=mask_type)
    
    post_fire_data = get_hls_stack(gdf, post_start, 
                                   post_end, creds=False,
                                   cloud_cover_threshold=cloud_cover_threshold,
                                   mask_type=mask_type)
    
    if (pre_fire_data is None) or (post_data is None):
        dnbr = None
        return dnbr, creds
    
    ######## Composite Generation, Logging, And Saving #########
    
    if composites:
        
        assert gdf_id is not None,"No ID provided for file writing. Please specify a column for gdf_id"
        
        id_attribute = gdf[gdf_id].unique()[0]
        pre_fire_rgb_path = f"pre_fire_composite_{id_attribute}_{pre_start}_{pre_end}_{cloud_cover_threshold}.tif"
        post_fire_rgb_path = f"post_fire_composite_{id_attribute}_{post_start}_{post_end}_{cloud_cover_threshold}.tif"
        
        if write_path:
            pre_fire_rgb_path = os.path.join(write_path,pre_fire_rgb_path)
            post_fire_rgb_path = os.path.join(write_path,post_fire_rgb_path)

        pre_fire_rgb = pre_fire_data.sel(band=['Red', 'Green', 'Blue']).median("time", keep_attrs=True).compute()
        post_fire_rgb = post_fire_data.sel(band=['Red', 'Green', 'Blue']).median("time", keep_attrs=True).compute()
        
        pre_fire_visibility = float(pre_fire_rgb.notnull().sum() / (pre_fire_rgb.shape[0]*pre_fire_rgb.shape[1]))
        post_fire_visibility = float(post_fire_rgb.notnull().sum() / (post_fire_rgb.shape[0]*post_fire_rgb.shape[1]))

        pre_fire_rgb.rio.to_raster(pre_fire_rgb_path)
        post_fire_rgb.rio.to_raster(post_fire_rgb_path)
        
        if logging:
            
            logging_data = {'Attribute': id_attribute,
                            'FireStart': fire_start,
                            'FireEnd': fire_end,
                            'PreOffset': pre_offset,
                            'PreOffsetRange': pre_offset_range,
                            'PreFireStart': pre_start,
                            'PreFireEnd': pre_end,
                            'PostOffset': post_offset,
                            'PostStart': post_start,
                            'PostEnd': post_end,
                            'CloudCoverThreshold': cloud_cover_threshold,
                            'MaskType': mask_type,
                            'EstPreFireViz': pre_fire_visibility,
                            'EstPostFireViz': post_fire_visibility,
                            'geometry': gdf.geometry}
            
            logging_df = pd.DataFrame.from_dict(logging_data)
            
            if log_path:
                
                if os.path.exists(log_path):
                    print('Appending composite metadata to existing logging dataframe')
                    existing_logging_df = pd.read_csv(log_path,header=True,index=True)
                    new_logging_df = pd.concat([existing_logging_df, logging_df])
                else:
                    new_logging_df = logging_df
                
                new_logging_df.to_csv(log_path,header=True,index=True)

    ###################################################

    pre_nir = pre_fire_data.sel(band='NIR').median("time", keep_attrs=True)
    pre_swir1 = pre_fire_data.sel(band='SWIR1').median("time", keep_attrs=True)
    pre_nbr = (pre_nir - pre_swir1)/((pre_nir + pre_swir1) + 1e-10)
    
    # pull our composited bands for nbr calculation
    post_nir = post_data.sel(band='NIR').median("time", keep_attrs=True)
    post_swir1 = post_data.sel(band='SWIR1').median("time", keep_attrs=True)
    post_nbr = (post_nir - post_swir1)/((post_nir + post_swir1) + 1e-10)
    
    dnbr = pre_nbr - post_nbr
    dnbr.rio.write_crs('EPSG:'+str(epsg),inplace=True)

    if composites:
        return dnbr.compute(), pre_fire_rgb, post_fire_rgb, creds # return creds and all composites to reuse!    
    
    return dnbr.compute(), creds # return creds to reuse!

def pc_catalog():
    
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/",
                          modifier=planetary_computer.sign_inplace)
    
    return catalog

def get_elevation(gdf,catalog=None,epsg=6933,resolution=30,creds=False):
    
    gdf_copy = gdf.copy().to_crs('EPSG:4326')
    bbox = gdf_copy.total_bounds
    
    if catalog:
        catalog = catalog
    
    else:
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/",
                          modifier=planetary_computer.sign_inplace)

    search = catalog.search(collections=["nasadem"], bbox=bbox, limit=1)
    nasadem_item = next(search.items())
    
    elevation_raster = stackstac.stack([nasadem_item.to_dict()],
                                       epsg=6933,
                                       resampling=rasterio.enums.Resampling.bilinear,
                                       chunksize=2048,
                                       resolution=30).squeeze()
    
    gdf_elevation_epsg = gdf.copy().to_crs('EPSG:'+str(epsg))
    bbox_adj = gdf_elevation_epsg.total_bounds
    
    if elevation_raster['y'][0] < elevation_raster['y'][-1]:
        elevation_raster = elevation_raster.sel(x=slice(bbox_adj[0],bbox_adj[2]),
                                                y=slice(bbox_adj[1],bbox_adj[3]))
    
    if elevation_raster['y'][0] > elevation_raster['y'][-1]:
        elevation_raster = elevation_raster.sel(x=slice(bbox_adj[0],bbox_adj[2]),
                                                y=slice(bbox_adj[3],bbox_adj[1]))
    #print(elevation_raster.shape)
    return elevation_raster


    
    
    