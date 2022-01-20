"""A collection of functions for accessing data on Microsoft's Planetary computer"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence, Tuple, Union

from loguru import logger
import numpy as np
import pandas as pd

from PIL import Image
import planetary_computer as pc
from pystac_client import Client
import shapely.geometry
import rasterio
import rioxarray

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

def get_closest_item(
    items: Sequence[dict],
    polygon: shapely.geometry.polygon.Polygon,
    timestamp: datetime,
) -> Optional[dict]:
    """
    Returns the item with maximum overlap and closest timestamp.

    Args:
        items (Sequence[dict]): items returned by a PySTAC catalog search
        polygon (shapely.geometry.polygon.Poylgon): polygon of the latitude/longitude
            coordinates for the original chip to be matched
        timestamp (datetime.timestamp): timestamp for the original chip to be matched

    Returns:
        pystac.item.Item: PySTAC item with the maximum geographic overlap to the
            original chip to be matched. If multiple items have equal overlap, the
            item with the closest timestamp is returned.
    """
    # Convert consumable iterators to lists
    items = list(items)

    # Compute overlap between each query result and the geotiff polygon
    overlaps = [
        shapely.geometry.shape(item.geometry).intersection(polygon).area / polygon.area
        for item in items
    ]
    max_overlap = max(overlaps)
    items_overlaps = [
        (item, overlap)
        for item, overlap in zip(items, overlaps)
        if overlap == max_overlap
    ]

    # If one item has higher overlap than the rest, return it
    if len(items_overlaps) == 1:
        return items_overlaps[0][0]

    # If multiple items have equally high overlap, return the one with the closest timestamp
    min_timedelta = timedelta.max
    best_item = None
    for item, overlap in items_overlaps:
        item_timedelta = abs(item.datetime.astimezone(timestamp.tzinfo) - timestamp)
        if item_timedelta < min_timedelta:
            min_timedelta = item_timedelta
            best_item = item

    return best_item


def get_all_items(
    items: Sequence[dict],
    polygon: shapely.geometry.polygon.Polygon,
    timestamp: datetime,
) -> Optional[dict]:
    """
    Returns all items with overlap and their timestamps.

    Args:
        items (Sequence[dict]): items returned by a PySTAC catalog search
        polygon (shapely.geometry.polygon.Poylgon): polygon of the latitude/longitude
            coordinates for the original chip to be matched
        timestamp (datetime.timestamp): timestamp for the original chip to be matched

    Returns:
        pystac.item.Item: PySTAC item with the maximum geographic overlap to the
            original chip to be matched. If multiple items have equal overlap, the
            item with the closest timestamp is returned.
    """
    # Convert consumable iterators to lists
    items = list(items)
    
    # Compute overlap between each query result and the geotiff polygon
    overlaps = [
        shapely.geometry.shape(item.geometry).intersection(polygon).area / polygon.area
        for item in items
    ]
    max_overlap = max(overlaps)

    # keep only items that cover the whole region
    # can be extended to patch together only partial covers, but for now this should work well enough
    items = [item for item, overlap in zip(items, overlaps) if overlap == max_overlap]

    return items


def check_projection(geotiff, item, verbose=True):
    """Ensure that original chip and PySTAC item have the same coordinate projection"""
    
    if geotiff.meta["crs"] == item.properties["proj:epsg"]:
        if verbose:
            logger.debug(
                f"""GeoTIFF and STAC item have same CRS {geotiff.meta["crs"]}"""
            )
        bounds = geotiff.bounds
    else:
        if verbose:
            logger.debug(
                f"""Transforming from {geotiff.meta["crs"]} """
                f"""to {item.properties["proj:epsg"]}"""
            )
        bounds = rasterio.warp.transform_bounds(
            geotiff.meta["crs"],
            item.properties["proj:epsg"],
            geotiff.bounds.left,
            geotiff.bounds.bottom,
            geotiff.bounds.right,
            geotiff.bounds.top,
        )

    return bounds


def query_bands(
    geotiff: rasterio.io.DatasetReader,
    timestamp: Union[datetime, pd.Timestamp, str],
    asset_keys: Sequence[str],
    collection: str = "sentinel-2-l2a",
    query_range_minutes: int = 120,
    output_shape: Optional[Tuple[int, int]] = None,
    verbose: Optional[bool] = True,
    want_closest: bool = True,
    max_item_limit: int = 500,
    max_cloud_cover: int = 100,
    max_cloud_shadow_cover: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Queries the Planetary Computer STAC API for additional imagery that
    corresponds to the same spatial extent as a provided GeoTIFF.

    Args:
        geotiff (rasterio.io.DatasetReader): A rasterio GeoTIFF
        timestamp (datetime or str): Timestamp for GeoTIFF acquisition used
            in the STAC API query to find the corresponding scene
        asset_keys (sequence of str): A sequence (list, tuple, set, etc.) of
            keys specifying the desired STAC assets to return
        query_range_minutes (int): Duration of the time range for the STAC API
            query. You can increase this if the query does not return any results.
        output_shape (tuple of ints, optional): If provided, reshape the output
            to this (height, width)
        verbose (bool, Optional): Whether to print logging messages. Defaults to True
        want_closest (bool): return closest image to query time (True) or all images in range (False). Default True
    Returns:
        dict {str: np.ndarray}: A dictionary where each key is an asset name, and each value
            is the array of values for that asset from the PySTAC item that most closely
            matches the original chip's location and time
    """
    # Convert bounds to regular lat/long coordinates
    left, bottom, right, top = rasterio.warp.transform_bounds(
        geotiff.meta["crs"],
        4326,  # code for the lat-lon coordinate system
        *geotiff.bounds,
    )

    # Get a polygon for the area to search
    area_of_interest = shapely.geometry.shape(
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [left, bottom],
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ]
            ],
        }
    )

    # Get the timestamp range to search
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()

    range_start = timestamp - timedelta(minutes=query_range_minutes // 2)
    range_end = timestamp + timedelta(minutes=query_range_minutes // 2)
    time_range = (
        f"{range_start.strftime(DATETIME_FORMAT)}/{range_end.strftime(DATETIME_FORMAT)}"
    )

    # Search the catalog
    if want_closest:
        query = None
    else:
        query = {"eo:cloud_cover": {"lt": max_cloud_cover},
                "s2:cloud_shadow_percentage": {"lt": max_cloud_shadow_cover}}

    search = catalog.search(
        collections=[collection],
        intersects=area_of_interest,
        datetime=time_range,
        limit=max_item_limit,
        query=query,
    )

    if want_closest:
        # Filter to the best-matching item
        items = list(search.get_items())
        items = [get_closest_item(items, area_of_interest, timestamp)]
        dtime = np.array( [ abs( (item.datetime - timestamp).total_seconds() ) for item in items])
        properties = [item.properties for item in items]
        
    else:
        items = get_all_items(search.get_items(), area_of_interest, timestamp)
        print("Found {:d} items matching search parameters".format(len(items)))
              
        # keep only closest <max_item_limit> items to original chip timestamp
        dtime = np.array( [ abs( (item.datetime - timestamp).total_seconds() ) for item in items])
        dtime = dtime % (60*60*24*365) # land should be roughly similar each year at a given time 
        
        dtime_sort = np.argsort(dtime)
        
        dtime = dtime[dtime_sort][:max_item_limit]
        properties = [items[i].properties for i in dtime_sort][:max_item_limit]
        items = [items[i] for i in dtime_sort][:max_item_limit]
        #        print([properties[i]['eo:cloud_cover'] for i in range(len(items))])
        #         print(dtime_sort, items[dtime_sort])
        #         # items = [items[i] for item in items
        
        #         dtime_max = dtime[np.argsort(dtime)][min(len(dtime)-1, max_item_limit - 1)]
        #         dm = dtime <= dtime_max
        
        #         items = [item for i, item in enumerate(items) if dm[i] == True]
        #         dtime = [dtime for i, dtime in enumerate(dtime) if dm[i] == True]
    if len(items) == 0:
        raise ValueError(
            "Query returned no results. Check that the bounding box is correct "
            "or try increasing the query time range."
        )

        

    assets = {}
    for it, item in enumerate(items):
        bounds = check_projection(geotiff, item, verbose)

        # Load the matching PySTAC asset
        for asset_key in asset_keys:

            try:
                asset = np.array(
                    rioxarray.open_rasterio(pc.sign(item.assets[asset_key].href))
                    .rio.clip_box(*bounds)
                    .load()
                    .transpose("y", "x", "band")
                )
                
            except:
                print("No data in bounds for item, asset_key = ", item, asset_key)
                asset = np.full((512, 512), 0, dtype=np.uint8)
                    
            # Reshape to singe-band image and resize if needed
            asset = Image.fromarray(asset.squeeze())
            if output_shape:
                asset = asset.resize(output_shape)
            asset = np.array(asset)
            if len(items) == 1:
                assets[asset_key] = asset
                assets[asset_key + "_time"] = str(item.datetime)
                assets[asset_key + "_dtime"] = dtime[it]
                assets[asset_key + "_properties"] = properties[it]

            else:
                if it == 0:
                    assets[asset_key] = []
                    assets[asset_key + "_time"] = []
                    assets[asset_key + "_dtime"] = []
                    assets[asset_key + "_properties"] = []

                assets[asset_key].append(asset)
                assets[asset_key + "_time"].append(str(item.datetime))
                assets[asset_key + "_dtime"].append(dtime[it])
                assets[asset_key + "_properties"].append(properties[it])

    return assets, items
