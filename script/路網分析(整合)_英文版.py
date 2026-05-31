# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import folium
import numpy as np
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
from IPython.display import display
from shapely.geometry import LineString, MultiLineString
from branca.element import MacroElement, Template

output_dir = Path("../output")
output_dir.mkdir(exist_ok=True)

# Scenario and accessibility settings
scenario_names = ["Q1point1", "Q10", "Q25", "Q50", "Q100"]
max_accept_time_medical = 60
max_accept_time_shelter = 60

twd97_crs = "EPSG:3826"  # TWD97 / TM2
wgs84 = "EPSG:4326"  # WGS84


def format_coord(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.5f}"

lu_gdf = gpd.read_file("../data/11_港口溪流域/01_土地利用/2007_LU.shp")

# Set CRS if missing, then normalize to EPSG:3826 for centroid and network analysis.
if lu_gdf.crs is None:
    lu_gdf = lu_gdf.set_crs(twd97_crs)

lu_3826 = lu_gdf.to_crs(twd97_crs)

lu_3826["LCODE_C1"] = lu_3826["LCODE_C1"].astype(str).str.zfill(2)
lu_3826["LCODE_C2"] = lu_3826["LCODE_C2"].astype(str).str.zfill(4)
lu_3826["LCODE_C3"] = lu_3826["LCODE_C3"].astype(str).str.zfill(6)
print(lu_3826[["LCODE_C1", "LCODE_C2", "LCODE_C3"]].head(5))

#%% 1.1 Add land-use labels

c1_map = {
    "01": "Agricultural Land",
    "02": "Forestry Land",
    "03": "Transportation Land",
    "04": "Water Conservancy Land",
    "05": "Building Land",
    "06": "Public Use Land",
    "07": "Recreation Land",
    "08": "Mining and Salt Land",
    "09": "Other Land Use",
}

c2_map = {
    "0502": "Residential",
    "0603": "Medical Care",
}

c3_map = {
    "050201": "Pure Residential",
    "050202": "Mixed Residential",
    "050203": "Rural Residential",
    "050204": "Other Residential",
    "060300": "Medical Care",
}

lu_3826["C1_NAME"] = lu_3826["LCODE_C1"].map(c1_map)
lu_3826["C2_NAME"] = lu_3826["LCODE_C2"].map(c2_map)
lu_3826["C3_NAME"] = lu_3826["LCODE_C3"].map(c3_map)

#%% 2. Extract residential and medical land-use polygons

residential_3826 = lu_3826[lu_3826["LCODE_C2"] == "0502"].copy()
medical_3826 = lu_3826[lu_3826["LCODE_C2"] == "0603"].copy()

print("Residential land parcel count:", len(residential_3826))
print("Medical land parcel count:", len(medical_3826))

residential_3826.head(3)

#%% 2.1 Read shelters CSV

shelters_csv_path = Path("../data/避難收容處所_清理後.csv")

if shelters_csv_path.exists():
    shelters_csv = pd.read_csv(shelters_csv_path, encoding="utf-8-sig")
    print("Total shelter records:", len(shelters_csv))

    if "座標有效性" in shelters_csv.columns:
        valid_shelters = shelters_csv[
            shelters_csv["座標有效性"] == "有效"
        ].copy()
    else:
        print('Warning: "座標有效性" column not found, using all records')
        valid_shelters = shelters_csv.copy()

    print("Valid shelter record count:", len(valid_shelters))
else:
    raise FileNotFoundError(f"Shelter CSV not found: {shelters_csv_path}")

valid_shelters.head(1)

#%% 2.2 Convert shelters to GeoDataFrame

shelters_wgs84 = gpd.GeoDataFrame(
    valid_shelters,
    geometry=gpd.points_from_xy(
        valid_shelters["經度"],
        valid_shelters["緯度"]
    ),
    crs=wgs84
)
shelters_3826 = shelters_wgs84.to_crs(twd97_crs)

print("Shelter CRS:", shelters_3826.crs)

#%% 2.3 Keep shelters within study area

study_area_3826 = gpd.GeoDataFrame(
    geometry=[lu_3826.geometry.union_all()],
    crs=lu_3826.crs
)

shelters_in_study_3826 = gpd.sjoin(
    shelters_3826,
    study_area_3826,
    how="inner",
    predicate="within"
).copy()

if "index_right" in shelters_in_study_3826.columns:
    shelters_in_study_3826 = shelters_in_study_3826.drop(columns="index_right")

print("Shelters within study area:", len(shelters_in_study_3826))

#%% 2.4 Keep shelters within 1 km buffer

study_area_buffer_3826 = study_area_3826.copy()
study_area_buffer_3826["geometry"] = study_area_buffer_3826.geometry.buffer(1000)

shelters_in_buffer_3826 = gpd.sjoin(
    shelters_3826,
    study_area_buffer_3826,
    how="inner",
    predicate="within"
).copy()

if "index_right" in shelters_in_buffer_3826.columns:
    shelters_in_buffer_3826 = shelters_in_buffer_3826.drop(columns="index_right")

print("Shelters within 1 km buffer of study area:", len(shelters_in_buffer_3826))

#%% 3. Convert polygons to centroids

residential_origins_3826 = residential_3826.copy()
residential_origins_3826["geometry"] = residential_origins_3826.geometry.centroid
residential_origins_3826["origin_id"] = range(len(residential_origins_3826))

medical_dest_3826 = medical_3826.copy()
medical_dest_3826["geometry"] = medical_dest_3826.geometry.centroid
medical_dest_3826["medical_id"] = range(len(medical_dest_3826))

print("Residential origin count:", len(residential_origins_3826))
print("Medical destination count:", len(medical_dest_3826))
print("Shelter destination count:", len(shelters_in_buffer_3826))

shelters_in_buffer_3826.head(1)

#%% 4. Build road network

road_boundary_3826 = study_area_3826.copy()
road_boundary_3826["geometry"] = road_boundary_3826.geometry.buffer(500)

road_boundary_wgs84 = road_boundary_3826.to_crs(wgs84)
road_boundary_polygon = road_boundary_wgs84.geometry.iloc[0]

print("Road network download extent CRS:", road_boundary_wgs84.crs)

#%% 4.1 Download road network from OSM

G_4326 = ox.graph_from_polygon(
    road_boundary_polygon,
    network_type="all",
    simplify=True,
    retain_all=True,
    truncate_by_edge=True
)

print("OSM road network download completed")

#%% 4.2 Project road network to EPSG:3826

G_3826 = ox.project_graph(G_4326, to_crs=twd97_crs)
nodes_3826, edges_3826 = ox.graph_to_gdfs(G_3826)

print("Road network node count:", len(nodes_3826))
print("Road network edge count:", len(edges_3826))
print("Road network CRS:", edges_3826.crs)

def clean_highway(x):
    if isinstance(x, list):
        return x[0]
    else:
        return x

edges_3826["highway_simple"] = edges_3826["highway"].apply(clean_highway)

print(edges_3826["highway_simple"].unique())

#%% 5. 給定預設速度

# 預設速度
#%% 給 all 路網設定速度

default_speed = {
    "primary": 50,
    "primary_link": 40,
    "secondary": 50,
    "secondary_link": 40,
    "tertiary": 40,
    "residential": 30,
    "unclassified": 30,

    # 小型道路或出入道路
    "service": 20,
    "track": 15,

    # 步行類道路
    "path": 5,
    "footway": 5,
    "steps": 2,
}


# G_3826單位為公尺
# u      這段道路的起點節點
# v      這段道路的終點節點
# k      edge key，用來區分同一組節點之間的多條路
# data   這段道路的屬性資料
for u, v, k, data in G_3826.edges(keys=True, data=True):
    highway_type = clean_highway(data.get("highway", None))
    
    # 不確定這邊要預設多少
    speed_kmh = default_speed.get(highway_type, 10)
    length_m = data.get("length", 0)
    travel_time_min = length_m / 1000 / speed_kmh * 60

    data["highway_type"] = highway_type
    data["speed_kmh"] = speed_kmh
    data["travel_time_min"] = travel_time_min

#%% 5.1 重新生成新的路網與節點，包含限速與行車時間等

nodes_3826, edges_3826 = ox.graph_to_gdfs(G_3826)

display(edges_3826.head(2))
display(edges_3826[[
    "highway_type",
    "speed_kmh",
    "length",
    "travel_time_min"
]].head(5))

#%% 6. 繪製互動式地圖，檢查目前建立的圖層

# ------------------------------------------------------------
# 1. 轉成 WGS84，給 Folium 使用
# ------------------------------------------------------------
study_area_wgs84 = study_area_3826.to_crs(wgs84)
road_boundary_wgs84 = road_boundary_3826.to_crs(wgs84)

residential_wgs84 = residential_3826.to_crs(wgs84)
residential_origins_wgs84 = residential_origins_3826.to_crs(wgs84)

medical_wgs84 = medical_3826.to_crs(wgs84)
medical_dest_wgs84 = medical_dest_3826.to_crs(wgs84)

shelters_wgs84 = shelters_in_buffer_3826.to_crs(wgs84)

nodes_wgs84 = nodes_3826.to_crs(wgs84)
edges_wgs84 = edges_3826.to_crs(wgs84)

# ------------------------------------------------------------
# 2. 建立底圖
# ------------------------------------------------------------
center = study_area_wgs84.geometry.union_all().centroid

m = folium.Map(
    location=[center.y, center.x],
    zoom_start=12,
    tiles="OpenStreetMap"
)
# ------------------------------------------------------------
# 3. 流域範圍與道路網抓取範圍
# ------------------------------------------------------------

# folium.GeoJson(
#     study_area_wgs84,
#     name="流域範圍",
#     style_function=lambda x: {
#         "fillColor": "none",
#         "color": "black",
#         "weight": 3,
#         "fillOpacity": 0,
#         "opacity": 1
#     }
# ).add_to(m)

folium.GeoJson(
    road_boundary_wgs84,
    name="Study Area",
    style_function=lambda x: {
        "fillColor": "none",
        "color": "purple",
        "weight": 2,
        "fillOpacity": 0,
        "opacity": 1,
        "dashArray": "5, 5"
    }
).add_to(m)


# ------------------------------------------------------------
# 4. 住宅與醫療場所 polygon
# ------------------------------------------------------------

folium.GeoJson(
    residential_wgs84,
    name="Residential Land 0502",
    style_function=lambda x: {
        "fillColor": "yellow",
        "color": "orange",
        "weight": 1,
        "fillOpacity": 0.45,
        "opacity": 0.9
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["LCODE_C2", "LCODE_C3"],
        aliases=["Level 2 Code", "Level 3 Code"]
    )
).add_to(m)

folium.GeoJson(
    medical_wgs84,
    name="Medical Land Use 0603",
    style_function=lambda x: {
        "fillColor": "blue",
        "color": "blue",
        "weight": 1,
        "fillOpacity": 0.35,
        "opacity": 0.9
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["LCODE_C2", "LCODE_C3"],
        aliases=["Level 2 Code", "Level 3 Code"]
    )
).add_to(m)


# ------------------------------------------------------------
# 5. 路網圖層
# ------------------------------------------------------------

folium.GeoJson(
    edges_wgs84,
    name="Road Network",
    style_function=lambda x: {
        "color": "gray",
        "weight": 3,
        "opacity": 0.6
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[
            col for col in ["highway_type", "speed_kmh", "length", "travel_time_min"]
            if col in edges_wgs84.columns
        ],
        aliases=[
            "Road Type", "Speed (km/h)", "Length (m)", "Travel Time (min)"
        ]
    )
).add_to(m)


# ------------------------------------------------------------
# 6. 建立可開關的點位圖層
# ------------------------------------------------------------

road_node_layer = folium.FeatureGroup(
    name="Road Network Nodes",
    show=False
)

residential_origin_layer = folium.FeatureGroup(
    name="Residential Origins (Centroids)",
    show=True
)

medical_dest_layer = folium.FeatureGroup(
    name="Medical Destinations (Centroids)",
    show=True
)

shelter_layer = folium.FeatureGroup(
    name="Shelters",
    show=True
)


# ------------------------------------------------------------
# 7. 加入路網節點
# ------------------------------------------------------------

for idx, row in nodes_wgs84.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=2,
        color="black",
        fill=True,
        fill_color="black",
        fill_opacity=0.7,
        popup=f"Node ID: {idx}"
    ).add_to(road_node_layer)


# ------------------------------------------------------------
# 8. 加入住宅起點 centroid
# ------------------------------------------------------------

# 將每一個住宅用地 centroid 加入住宅起點圖層
for idx, row in residential_origins_wgs84.iterrows():

    popup_text = (
        f"Residential Origin ID: {row.get('origin_id', '')}<br>"
        f"Land Use Level 1: {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
        f"Land Use Level 2: {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
        f"Land Use Level 3: {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}"
    )

    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.85,
        popup=folium.Popup(popup_text, max_width=350)
    ).add_to(residential_origin_layer)


# ------------------------------------------------------------
# 9. 加入醫療目的地 centroid
# ------------------------------------------------------------

# 將每一個醫療保健用地 centroid 加入醫療目的地圖層
for idx, row in medical_dest_wgs84.iterrows():

    popup_text = (
        f"Medical Destination ID: {row.get('medical_id', '')}<br>"
        f"Land Use Level 1: {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
        f"Land Use Level 2: {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
        f"Land Use Level 3: {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}"
    )

    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=6,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.9,
        popup=folium.Popup(popup_text, max_width=350)
    ).add_to(medical_dest_layer)


# ------------------------------------------------------------
# 10. 加入避難所
# ------------------------------------------------------------

# 將每一個避難所加入避難所圖層
for idx, row in shelters_wgs84.iterrows():

    popup_text = (
        f"Shelter Name: {row.get('避難收容處所名稱', '')}<br>"
        f"County and Township: {row.get('縣市及鄉鎮市區', '')}<br>"
        f"Village: {row.get('村里', '')}<br>"
        f"Address: {row.get('避難收容處所地址', '')}<br>"
        f"Planned Service Villages: {row.get('預計收容村里', '')}<br>"
        f"Planned Capacity: {row.get('預計收容人數', '')}<br>"
        f"Applicable Disaster Type: {row.get('適用災害類別', '')}<br>"
        f"Manager: {row.get('管理人姓名', '')}<br>"
        f"Manager Phone: {row.get('管理人電話', '')}<br>"
        f"Longitude: {format_coord(row.get('經度', np.nan))}<br>"
        f"Latitude: {format_coord(row.get('緯度', np.nan))}"
    )

    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=6,
        color="green",
        fill=True,
        fill_color="green",
        fill_opacity=0.9,
        popup=folium.Popup(popup_text, max_width=350)
    ).add_to(shelter_layer)


# ------------------------------------------------------------
# 11. 將點位圖層加入地圖
# ------------------------------------------------------------

road_node_layer.add_to(m)
residential_origin_layer.add_to(m)
medical_dest_layer.add_to(m)
shelter_layer.add_to(m)


# ------------------------------------------------------------
# 12. 圖層控制與顯示
# ------------------------------------------------------------

folium.LayerControl(collapsed=False).add_to(m)

m

#%% 13. 儲存互動式地圖為 HTML

html_path = output_dir / "road_network_analysis_result.html"

m.save(html_path)

print("Interactive map saved to:", html_path)

paths = {
    "Q1.1": "../data/Geo_RA/Q1point1_depth_max.GangKoudem.2022dem.tif",
    "Q10":  "../data/Geo_RA/Q10_depth_max.GangKoudem.2022dem.tif",
    "Q25":  "../data/Geo_RA/Q25_depth_max.GangKoudem.2022dem.tif",
    "Q50":  "../data/Geo_RA/Q50_depth_max.GangKoudem.2022dem.tif",
    "Q100": "../data/Geo_RA/Q100_depth_max.GangKoudem.2022dem.tif",
}

for name, path in paths.items():
    with rasterio.open(path) as src:
        print(name)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Shape:", src.height, src.width)
        print("Bounds:", src.bounds)
        print("Nodata:", src.nodata)
        print()

import rasterio
import numpy as np

tif_path ="../data/Geo_RA/Q1point1_depth_max.GangKoudem.2022dem.tif"

with rasterio.open(tif_path) as src:
    arr = src.read(1).astype(float)
    nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    # 如果 0 代表沒有淹水，可以另外看 > 0 的範圍
    arr_positive = arr[arr > 0]

    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Nodata:", nodata)
    print("All valid values min:", np.nanmin(arr))
    print("All valid values max:", np.nanmax(arr))
    print("All valid values mean:", np.nanmean(arr))
    print("All valid values median:", np.nanmedian(arr))

    print("Number of pixels greater than 0:", arr_positive.size)

    if arr_positive.size > 0:
        print("Flood depth > 0 min:", np.nanmin(arr_positive))
        print("Flood depth > 0 max:", np.nanmax(arr_positive))
        print("Flood depth > 0 mean:", np.nanmean(arr_positive))
        print("Flood depth > 0 median:", np.nanmedian(arr_positive))
        print("95th percentile:", np.nanpercentile(arr_positive, 95))
        print("99th percentile:", np.nanpercentile(arr_positive, 99))

#%% 8. 選擇分析用路網

G_analysis = G_3826
nodes_analysis_3826 = nodes_3826
edges_analysis_3826 = edges_3826

#%% 9. 將住宅、醫療、避難所 snap 到分析用路網，「每個點對應到哪個路網節點」

residential_origins_3826["nearest_node"] = ox.distance.nearest_nodes(
    G_analysis,
    X=residential_origins_3826.geometry.x,
    Y=residential_origins_3826.geometry.y
)

medical_dest_3826["nearest_node"] = ox.distance.nearest_nodes(
    G_analysis,
    X=medical_dest_3826.geometry.x,
    Y=medical_dest_3826.geometry.y
)

shelters_in_buffer_3826["nearest_node"] = ox.distance.nearest_nodes(
    G_analysis,
    X=shelters_in_buffer_3826.geometry.x,
    Y=shelters_in_buffer_3826.geometry.y
)

print("Residential, medical, and shelter points have been snapped to the analysis road network")

#%% 10. 直接由住宅點往目的地計算最快路徑

def get_edge_min_attr(G, u, v, attr):
    edge_data = G.get_edge_data(u, v)

    if edge_data is None:
        return np.nan

    values = [data[attr] for k, data in edge_data.items() if attr in data]

    if len(values) == 0:
        return np.nan

    return min(values)


def calc_route_cost(G, route, attr):
    if route is None or len(route) < 2:
        return np.nan

    total = 0

    for u, v in zip(route[:-1], route[1:]):
        value = get_edge_min_attr(G, u, v, attr)
        if pd.isna(value):
            return np.nan
        total += value

    return total


def find_nearest_destination_direct(
    G,
    origins_gdf,
    destinations_gdf,
    origin_node_col="nearest_node",
    dest_node_col="nearest_node",
    dest_id_col=None,
    prefix="dest"
):
    origins_gdf = origins_gdf.copy()
    destinations_gdf = destinations_gdf.copy()

    destination_nodes = destinations_gdf[dest_node_col].dropna().unique().tolist()

    if dest_id_col is not None:
        dest_node_to_id = (
            destinations_gdf
            .drop_duplicates(subset=dest_node_col)
            .set_index(dest_node_col)[dest_id_col]
            .to_dict()
        )
    else:
        dest_node_to_id = {node: node for node in destination_nodes}

    nearest_dest_ids = []
    nearest_distances = []
    nearest_times = []
    nearest_routes = []

    for origin_node in origins_gdf[origin_node_col]:
        try:
            # 計算從住宅節點出發，到所有可達節點的最短通行時間
            distance_dict, path_dict = nx.single_source_dijkstra(
                G,
                source=origin_node,
                weight="travel_time_min"
            )

            reachable_dest_nodes = [
                node for node in destination_nodes
                if node in distance_dict
            ]

            if len(reachable_dest_nodes) == 0:
                nearest_dest_ids.append(np.nan)
                nearest_distances.append(np.nan)
                nearest_times.append(np.nan)
                nearest_routes.append(None)
                continue

            # 找出通行時間最短的目的地節點
            nearest_dest_node = min(
                reachable_dest_nodes,
                key=lambda node: distance_dict[node]
            )

            route = path_dict[nearest_dest_node]

            nearest_dest_ids.append(dest_node_to_id.get(nearest_dest_node, nearest_dest_node))
            nearest_distances.append(calc_route_cost(G, route, "length"))
            nearest_times.append(calc_route_cost(G, route, "travel_time_min"))
            nearest_routes.append(route)

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            nearest_dest_ids.append(np.nan)
            nearest_distances.append(np.nan)
            nearest_times.append(np.nan)
            nearest_routes.append(None)

    origins_gdf[f"nearest_{prefix}_id"] = nearest_dest_ids
    origins_gdf[f"distance_to_{prefix}_m"] = nearest_distances
    origins_gdf[f"time_to_{prefix}_min"] = nearest_times
    origins_gdf[f"route_to_{prefix}"] = nearest_routes

    return origins_gdf

#%% 11. 計算住宅到最近醫療場所

residential_origins_3826 = find_nearest_destination_direct(
    G=G_analysis,
    origins_gdf=residential_origins_3826,
    destinations_gdf=medical_dest_3826,
    origin_node_col="nearest_node",
    dest_node_col="nearest_node",
    dest_id_col="medical_id",
    prefix="medical"
)

#%% 12. 計算住宅到最近避難所

shelters_in_buffer_3826 = shelters_in_buffer_3826.copy()
shelters_in_buffer_3826["shelter_id"] = range(len(shelters_in_buffer_3826))

residential_origins_3826 = find_nearest_destination_direct(
    G=G_analysis,
    origins_gdf=residential_origins_3826,
    destinations_gdf=shelters_in_buffer_3826,
    origin_node_col="nearest_node",
    dest_node_col="nearest_node",
    dest_id_col="shelter_id",
    prefix="shelter"
)

#%% 13. 檢查可及性結果

print("Number of residential points unable to reach medical facilities:")
print(residential_origins_3826["time_to_medical_min"].isna().sum())

print("Number of residential points unable to reach shelters:")
print(residential_origins_3826["time_to_shelter_min"].isna().sum())

print("Medical travel distance statistics:")
display(residential_origins_3826["distance_to_medical_m"].describe())

print("Medical travel time statistics:")
display(residential_origins_3826["time_to_medical_min"].describe())

print("Shelter travel distance statistics:")
display(residential_origins_3826["distance_to_shelter_m"].describe())

print("Shelter travel time statistics:")
display(residential_origins_3826["time_to_shelter_min"].describe())

#%% 14. 繪製整合版互動式地圖：淹水 + 路網 + 住宅最近醫療/避難所資訊

import folium
import rasterio
import numpy as np
from rasterio.warp import transform_bounds
from matplotlib import cm, colors


def add_flood_raster_to_folium(
    m,
    tif_path,
    layer_name="Flood Depth",
    vmin=0,
    vmax=1,
    opacity=0.80,
    show=False,
    add_colorbar=False,
    return_layer=False
):
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

        if nodata is not None:
            arr[arr == nodata] = np.nan

        bounds_wgs84 = transform_bounds(
            src.crs,
            "EPSG:4326",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
            densify_pts=21
        )

    west, south, east, north = bounds_wgs84
    folium_bounds = [[south, west], [north, east]]

    valid_values = arr[np.isfinite(arr)]
    valid_max = valid_values.max() if valid_values.size > 0 else vmax
    effective_vmax = max(vmax, valid_max, 0.500001)

    # 與 get_flood_time_factor 使用相同的分級邏輯
    flood_bins = [0.0, 0.05, 0.10, 0.25, 0.50, effective_vmax]
    flood_colors = [
        "#fff5f0",
        "#fcbba1",
        "#fc9272",
        "#ef3b2c",
        "#99000d",
    ]

    cmap = colors.ListedColormap(flood_colors)
    norm = colors.BoundaryNorm(flood_bins, cmap.N, clip=True)
    rgba = cmap(norm(arr))
    rgba[np.isnan(arr), 3] = 0
    rgba_img = (rgba * 255).astype(np.uint8)

    overlay = folium.raster_layers.ImageOverlay(
        image=rgba_img,
        bounds=folium_bounds,
        name=layer_name,
        opacity=opacity,
        interactive=True,
        cross_origin=False,
        zindex=1,
        show=show
    )
    overlay.add_to(m)

    if return_layer:
        return overlay
    return m


def add_flood_legend_to_folium(m, flood_layer, title="Flood Potential (m)", legend_id="scenario-flood-legend"):
    legend_items = [
        ("#fff5f0", "0 - 0.05 m"),
        ("#fcbba1", "0.05 - 0.10 m"),
        ("#fc9272", "0.10 - 0.25 m"),
        ("#ef3b2c", "0.25 - 0.50 m"),
        ("#99000d", "> 0.50 m"),
    ]

    items_html = "".join(
        [
            (
                "<div style='display:flex; align-items:center; margin-bottom:6px;'>"
                f"<span style='display:inline-block; width:18px; height:12px; background:{color}; "
                "border:1px solid #666; margin-right:8px;'></span>"
                f"<span>{label}</span>"
                "</div>"
            )
            for color, label in legend_items
        ]
    )

    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div id="{legend_id}" style="
        display:none;
        position: fixed;
        bottom: 40px;
        left: 40px;
        width: 210px;
        z-index: 9999;
        background: white;
        border: 2px solid #666;
        border-radius: 6px;
        padding: 12px 12px 10px 12px;
        font-size: 13px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    ">
        <div style="font-weight:700; margin-bottom:10px;">{title}</div>
        {items_html}
    </div>
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var map = {m.get_name()};
        var floodLayer = {flood_layer.get_name()};
        var legend = document.getElementById("{legend_id}");

        function syncLegend() {{
            if (!legend) return;
            legend.style.display = map.hasLayer(floodLayer) ? "block" : "none";
        }}

        syncLegend();
        map.on("overlayadd", function(e) {{
            if (e.layer === floodLayer) {{
                syncLegend();
            }}
        }});
        map.on("overlayremove", function(e) {{
            if (e.layer === floodLayer) {{
                syncLegend();
            }}
        }});
    }});
    </script>
    {{% endmacro %}}
    """

    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)
    return m


def sample_line_points(line, sample_dist=10):
    if line.length == 0:
        return [line.interpolate(0)]

    distances = np.arange(0, line.length, sample_dist)
    points = [line.interpolate(d) for d in distances]
    points.append(line.interpolate(line.length))
    return points


def get_edge_geometry(G, u, v, data):
    geom = data.get("geometry", None)
    if geom is not None and not geom.is_empty:
        return geom

    x1 = G.nodes[u]["x"]
    y1 = G.nodes[u]["y"]
    x2 = G.nodes[v]["x"]
    y2 = G.nodes[v]["y"]
    return LineString([(x1, y1), (x2, y2)])


def get_edge_max_flood_depth(edge_geom, src, sample_dist=10):
    if edge_geom is None or edge_geom.is_empty:
        return 0.0

    if isinstance(edge_geom, LineString):
        lines = [edge_geom]
    elif isinstance(edge_geom, MultiLineString):
        lines = list(edge_geom.geoms)
    else:
        return 0.0

    sample_points = []
    for line in lines:
        sample_points.extend(sample_line_points(line, sample_dist=sample_dist))

    coords = [(pt.x, pt.y) for pt in sample_points]
    sampled_values = list(src.sample(coords))

    flood_values = []
    for v in sampled_values:
        val = float(v[0])

        if src.nodata is not None and val == src.nodata:
            continue
        if np.isnan(val):
            continue
        if val > 0:
            flood_values.append(val)

    if len(flood_values) == 0:
        return 0.0

    return max(flood_values)


def get_flood_time_factor(max_depth):
    if np.isnan(max_depth) or max_depth == 0:
        return 1.0
    elif 0 < max_depth <= 0.05:
        return 1.5
    elif 0.05 < max_depth <= 0.10:
        return 2.0
    elif 0.10 < max_depth <= 0.25:
        return 4.0
    elif 0.25 < max_depth <= 0.50:
        return 10.0
    else:
        return None


def build_flood_scenario_graphs(
    G_base,
    flood_tif_path,
    scenario_name,
    sample_dist=10,
    base_time_col="travel_time_min"
):
    G_full = G_base.copy()
    blocked_edges = []

    depth_col = f"{scenario_name}_max_flood_depth_m"
    factor_col = f"{scenario_name}_time_factor"
    blocked_col = f"{scenario_name}_is_blocked"
    scenario_time_col = f"{scenario_name}_travel_time_min"

    with rasterio.open(flood_tif_path) as src:
        for u, v, k, data in G_full.edges(keys=True, data=True):
            edge_geom = get_edge_geometry(G_full, u, v, data)
            max_depth = get_edge_max_flood_depth(edge_geom, src, sample_dist=sample_dist)
            time_factor = get_flood_time_factor(max_depth)

            data[depth_col] = max_depth
            data[factor_col] = time_factor
            data[blocked_col] = time_factor is None

            if time_factor is None:
                data[scenario_time_col] = np.nan
                blocked_edges.append((u, v, k))
            else:
                base_time = data.get(base_time_col, np.nan)
                data[scenario_time_col] = base_time * time_factor

    G_routing = G_full.copy()
    for u, v, k in blocked_edges:
        if G_routing.has_edge(u, v, k):
            G_routing.remove_edge(u, v, k)

    _, edges_full_3826 = ox.graph_to_gdfs(G_full)
    return G_full, G_routing, edges_full_3826


def get_edge_min_attr(G, u, v, attr):
    edge_data = G.get_edge_data(u, v)

    if edge_data is None:
        return np.nan

    values = [data[attr] for k, data in edge_data.items() if attr in data]

    if len(values) == 0:
        return np.nan

    return min(values)


def calc_route_cost(G, route, attr):
    if route is None or len(route) < 2:
        return np.nan

    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        value = get_edge_min_attr(G, u, v, attr)
        if pd.isna(value):
            return np.nan
        total += value

    return total


def find_nearest_destination_in_scenario(
    G,
    origins_gdf,
    destinations_gdf,
    scenario_name,
    weight_col,
    origin_node_col="nearest_node",
    dest_node_col="nearest_node",
    dest_id_col=None,
    prefix="dest"
):
    origins_result = origins_gdf.copy()
    destinations_copy = destinations_gdf.copy()

    destination_nodes = destinations_copy[dest_node_col].dropna().unique().tolist()

    if dest_id_col is not None:
        dest_node_to_id = (
            destinations_copy
            .drop_duplicates(subset=dest_node_col)
            .set_index(dest_node_col)[dest_id_col]
            .to_dict()
        )
    else:
        dest_node_to_id = {node: node for node in destination_nodes}

    nearest_dest_ids = []
    nearest_distances = []
    nearest_times = []
    nearest_routes = []

    for origin_node in origins_result[origin_node_col]:
        try:
            distance_dict, path_dict = nx.single_source_dijkstra(
                G,
                source=origin_node,
                weight=weight_col
            )

            reachable_dest_nodes = [
                node for node in destination_nodes
                if node in distance_dict
            ]

            if len(reachable_dest_nodes) == 0:
                nearest_dest_ids.append(np.nan)
                nearest_distances.append(np.nan)
                nearest_times.append(np.nan)
                nearest_routes.append(None)
                continue

            nearest_dest_node = min(
                reachable_dest_nodes,
                key=lambda node: distance_dict[node]
            )
            route = path_dict[nearest_dest_node]

            nearest_dest_ids.append(dest_node_to_id.get(nearest_dest_node, nearest_dest_node))
            nearest_distances.append(calc_route_cost(G, route, "length"))
            nearest_times.append(calc_route_cost(G, route, weight_col))
            nearest_routes.append(route)

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            nearest_dest_ids.append(np.nan)
            nearest_distances.append(np.nan)
            nearest_times.append(np.nan)
            nearest_routes.append(None)

    origins_result[f"nearest_{prefix}_id_{scenario_name}"] = nearest_dest_ids
    origins_result[f"distance_to_{prefix}_m_{scenario_name}"] = nearest_distances
    origins_result[f"time_to_{prefix}_min_{scenario_name}"] = nearest_times
    origins_result[f"route_to_{prefix}_{scenario_name}"] = nearest_routes

    return origins_result


def calc_accessibility_score(travel_time, max_acceptable_time):
    if pd.isna(travel_time):
        return 0.0
    if pd.isna(max_acceptable_time) or max_acceptable_time <= 0:
        return 0.0

    score = 1 - (travel_time / max_acceptable_time)
    return max(0.0, score)


def add_residential_popup_layers(
    m,
    residential_result_wgs84,
    medical_lookup,
    shelter_lookup,
    scenario_name,
    medical_time_col,
    shelter_time_col,
    medical_id_col,
    shelter_id_col,
    medical_score_scenario_col,
    shelter_score_scenario_col
):
    residential_origin_layer = folium.FeatureGroup(name="Residential Origins (Centroids)", show=True)

    for _, row in residential_result_wgs84.iterrows():
        res_lon = row.geometry.x
        res_lat = row.geometry.y
        res_lon_str = format_coord(res_lon)
        res_lat_str = format_coord(res_lat)

        med_score_normal = row.get("medical_access_score_normal", 0.0)
        shel_score_normal = row.get("shelter_access_score_normal", 0.0)
        med_score_scenario = row.get(medical_score_scenario_col, 0.0)
        shel_score_scenario = row.get(shelter_score_scenario_col, 0.0)

        medical_text_normal = (
            "Normal Scenario Nearest Medical: Unreachable<br>"
            f"Normal Scenario Medical Accessibility Score: {med_score_normal:.2f}"
        )
        shelter_text_normal = (
            "Normal Scenario Nearest Shelter: Unreachable<br>"
            f"Normal Scenario Shelter Accessibility Score: {shel_score_normal:.2f}"
        )
        medical_text_scenario = (
            f"{scenario_name} Scenario Nearest Medical: Unreachable<br>"
            f"{scenario_name} Scenario Medical Accessibility Score: {med_score_scenario:.2f}"
        )
        shelter_text_scenario = (
            f"{scenario_name} Scenario Nearest Shelter: Unreachable<br>"
            f"{scenario_name} Scenario Shelter Accessibility Score: {shel_score_scenario:.2f}"
        )

        if pd.notna(row.get("nearest_medical_id", np.nan)):
            medical_row = medical_lookup.loc[row["nearest_medical_id"]]
            medical_text_normal = (
                f"Normal Scenario Nearest Medical ID: {row['nearest_medical_id']}<br>"
                f"Longitude: {format_coord(medical_row.geometry.x)}<br>"
                f"Latitude: {format_coord(medical_row.geometry.y)}<br>"
                f"Fastest Passable Time: {row['time_to_medical_min']:.2f} min<br>"
                f"Normal Scenario Medical Accessibility Score: {med_score_normal:.2f}"
            )

        if pd.notna(row.get("nearest_shelter_id", np.nan)):
            shelter_row = shelter_lookup.loc[row["nearest_shelter_id"]]
            shelter_text_normal = (
                f"Normal Scenario Nearest Shelter ID: {row['nearest_shelter_id']}<br>"
                f"Longitude: {format_coord(shelter_row.geometry.x)}<br>"
                f"Latitude: {format_coord(shelter_row.geometry.y)}<br>"
                f"Fastest Passable Time: {row['time_to_shelter_min']:.2f} min<br>"
                f"Normal Scenario Shelter Accessibility Score: {shel_score_normal:.2f}"
            )

        if pd.notna(row.get(medical_id_col, np.nan)):
            medical_row_s = medical_lookup.loc[row[medical_id_col]]
            medical_text_scenario = (
                f"{scenario_name} Scenario Nearest Medical ID: {row[medical_id_col]}<br>"
                f"Longitude: {format_coord(medical_row_s.geometry.x)}<br>"
                f"Latitude: {format_coord(medical_row_s.geometry.y)}<br>"
                f"Fastest Passable Time: {row[medical_time_col]:.2f} min<br>"
                f"{scenario_name} Scenario Medical Accessibility Score: {med_score_scenario:.2f}"
            )

        if pd.notna(row.get(shelter_id_col, np.nan)):
            shelter_row_s = shelter_lookup.loc[row[shelter_id_col]]
            shelter_text_scenario = (
                f"{scenario_name} Scenario Nearest Shelter ID: {row[shelter_id_col]}<br>"
                f"Longitude: {format_coord(shelter_row_s.geometry.x)}<br>"
                f"Latitude: {format_coord(shelter_row_s.geometry.y)}<br>"
                f"Fastest Passable Time: {row[shelter_time_col]:.2f} min<br>"
                f"{scenario_name} Scenario Shelter Accessibility Score: {shel_score_scenario:.2f}"
            )

        popup_text = (
            f"Residential ID: {row.get('origin_id', '')}<br>"
            f"Land Use Level 1: {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
            f"Land Use Level 2: {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
            f"Land Use Level 3: {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}<br>"
            f"Longitude: {res_lon_str}<br>"
            f"Latitude: {res_lat_str}<br><br>"
            f"{medical_text_normal}<br><br>"
            f"{medical_text_scenario}<br><br>"
            f"{shelter_text_normal}<br><br>"
            f"{shelter_text_scenario}"
        )

        folium.CircleMarker(
            location=[res_lat, res_lon],
            radius=4,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.85,
            popup=folium.Popup(popup_text, max_width=430)
        ).add_to(residential_origin_layer)

    residential_origin_layer.add_to(m)

def build_scenario_map(
    scenario_name,
    flood_tif_path,
    edges_scenario_3826,
    residential_scenario_3826,
    medical_scenario_3826,
    shelter_scenario_3826,
    depth_col,
    factor_col,
    blocked_col,
    scenario_time_col,
    medical_time_col,
    shelter_time_col,
    medical_id_col,
    shelter_id_col,
    medical_score_scenario_col,
    shelter_score_scenario_col
):
    study_area_wgs84 = study_area_3826.to_crs(wgs84)
    road_boundary_wgs84 = road_boundary_3826.to_crs(wgs84)
    residential_wgs84 = residential_3826.to_crs(wgs84)
    residential_result_wgs84 = residential_scenario_3826.to_crs(wgs84)
    medical_wgs84 = medical_3826.to_crs(wgs84)
    medical_dest_wgs84 = medical_scenario_3826.to_crs(wgs84)
    shelters_result_wgs84 = shelter_scenario_3826.to_crs(wgs84)
    nodes_wgs84 = nodes_3826.to_crs(wgs84)
    edges_wgs84 = edges_3826.to_crs(wgs84)

    edges_scenario_wgs84 = edges_scenario_3826.to_crs(wgs84)
    edges_scenario_passable_wgs84 = edges_scenario_wgs84[~edges_scenario_wgs84[blocked_col]].copy()
    edges_scenario_blocked_wgs84 = edges_scenario_wgs84[edges_scenario_wgs84[blocked_col]].copy()

    medical_lookup = medical_dest_wgs84.set_index("medical_id")
    shelter_lookup = shelters_result_wgs84.set_index("shelter_id")

    center = study_area_wgs84.geometry.union_all().centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="OpenStreetMap")

    flood_overlay = add_flood_raster_to_folium(
        m=m,
        tif_path=flood_tif_path,
        layer_name=f"{scenario_name} Flood Potential Map",
        vmin=0,
        vmax=1,
        opacity=0.8,
        show=True,
        add_colorbar=False,
        return_layer=True
    )
    m = add_flood_legend_to_folium(
        m,
        flood_overlay,
        title=f"Flood Potential (m)<br>Scenario: {scenario_name}"
    )

    folium.GeoJson(
        road_boundary_wgs84,
        name="Study Area",
        style_function=lambda x: {
            "fillColor": "none",
            "color": "purple",
            "weight": 2,
            "fillOpacity": 0,
            "opacity": 1,
            "dashArray": "5, 5"
        }
    ).add_to(m)

    folium.GeoJson(
        residential_wgs84,
        name="Residential Land 0502",
        style_function=lambda x: {
            "fillColor": "yellow",
            "color": "orange",
            "weight": 1,
            "fillOpacity": 0.45,
            "opacity": 0.9
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["LCODE_C2", "LCODE_C3"],
            aliases=["Land Use Level 2 Code", "Land Use Level 3 Code"]
        )
    ).add_to(m)

    folium.GeoJson(
        medical_wgs84,
        name="Medical Land Use 0603",
        style_function=lambda x: {
            "fillColor": "blue",
            "color": "blue",
            "weight": 1,
            "fillOpacity": 0.35,
            "opacity": 0.9
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["LCODE_C2", "LCODE_C3"],
            aliases=["Land Use Level 2 Code", "Land Use Level 3 Code"]
        )
    ).add_to(m)

    folium.GeoJson(
        edges_wgs84,
        name="Road Network",
        style_function=lambda x: {
            "color": "gray",
            "weight": 2,
            "opacity": 0.45
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[col for col in ["highway_type", "speed_kmh", "length", "travel_time_min"] if col in edges_wgs84.columns],
            aliases=["Road Type", "Speed Limit (km/h)", "Length (m)", "Baseline Fastest Passable Time (min)"]
        )
    ).add_to(m)

    folium.GeoJson(
        edges_scenario_passable_wgs84,
        name=f"{scenario_name} Scenario Passable Roads",
        style_function=lambda x: {
            "color": "orange",
            "weight": 3,
            "opacity": 0.85
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                col for col in [
                    "highway_type", "length", "travel_time_min",
                    depth_col, factor_col, scenario_time_col
                ] if col in edges_scenario_passable_wgs84.columns
            ],
            aliases=[
                "Road Type", "Road Length (m)", "Baseline Fastest Passable Time (min)",
                "Flood Depth (m)", "Travel Time Factor", f"{scenario_name} Scenario Fastest Passable Time (min)"
            ]
        )
    ).add_to(m)

    folium.GeoJson(
        edges_scenario_blocked_wgs84,
        name=f"{scenario_name} Scenario Blocked Roads",
        style_function=lambda x: {
            "color": "red",
            "weight": 4,
            "opacity": 1.0
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                col for col in [
                    "highway_type", "length", "travel_time_min",
                    depth_col, blocked_col
                ] if col in edges_scenario_blocked_wgs84.columns
            ],
            aliases=[
                "Road Type", "Road Length (m)", "Baseline Fastest Passable Time (min)",
                "Flood Depth (m)", "Blocked"
            ]
        )
    ).add_to(m)

    road_node_layer = folium.FeatureGroup(name="Road Network Nodes", show=False)
    medical_dest_layer = folium.FeatureGroup(name="Medical Destinations (Centroids)", show=True)
    shelter_layer = folium.FeatureGroup(name="Shelters", show=True)

    for idx, row in nodes_wgs84.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2,
            color="black",
            fill=True,
            fill_color="black",
            fill_opacity=0.7,
            popup=f"Node ID: {idx}"
        ).add_to(road_node_layer)

    add_residential_popup_layers(
        m=m,
        residential_result_wgs84=residential_result_wgs84,
        medical_lookup=medical_lookup,
        shelter_lookup=shelter_lookup,
        scenario_name=scenario_name,
        medical_time_col=medical_time_col,
        shelter_time_col=shelter_time_col,
        medical_id_col=medical_id_col,
        shelter_id_col=shelter_id_col,
        medical_score_scenario_col=medical_score_scenario_col,
        shelter_score_scenario_col=shelter_score_scenario_col
    )

    for _, row in medical_dest_wgs84.iterrows():
        popup_text = (
            f"Medical Destination ID: {row.get('medical_id', '')}<br>"
            f"Land Use Level 1: {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
            f"Land Use Level 2: {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
            f"Land Use Level 3: {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}<br>"
            f"Longitude: {format_coord(row.geometry.x)}<br>"
            f"Latitude: {format_coord(row.geometry.y)}"
        )
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.9,
            popup=folium.Popup(popup_text, max_width=350)
        ).add_to(medical_dest_layer)

    shelter_popup_fields = [
        ("避難收容處所名稱", "Shelter Name"),
        ("避難收容處所地址", "Shelter Address"),
        ("村里", "Village"),
        ("縣市及鄉鎮市區", "County and Township"),
        ("預計收容人數", "Planned Capacity"),
        ("適用災害類別", "Applicable Disaster Type"),
        ("管理人姓名", "Manager Name"),
        ("管理人電話", "Manager Phone"),
    ]

    for _, row in shelters_result_wgs84.iterrows():
        popup_lines = [f"Shelter ID: {row.get('shelter_id', '')}"]
        for field_name, field_label in shelter_popup_fields:
            if field_name in row.index:
                popup_lines.append(f"{field_label}: {row.get(field_name, '')}")
        popup_lines.append(f"Longitude: {format_coord(row.geometry.x)}")
        popup_lines.append(f"Latitude: {format_coord(row.geometry.y)}")
        popup_text = "<br>".join(popup_lines)

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.9,
            popup=folium.Popup(popup_text, max_width=380)
        ).add_to(shelter_layer)

    road_node_layer.add_to(m)
    medical_dest_layer.add_to(m)
    shelter_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

residential_base_3826 = residential_origins_3826.copy()
residential_base_3826["medical_access_score_normal"] = residential_base_3826["time_to_medical_min"].apply(
    lambda x: calc_accessibility_score(x, max_accept_time_medical)
)
residential_base_3826["shelter_access_score_normal"] = residential_base_3826["time_to_shelter_min"].apply(
    lambda x: calc_accessibility_score(x, max_accept_time_shelter)
)

scenario_csv_frames = []
q100_edges_3826 = None


def build_comparison_dataframe(residential_base_gdf, scenario_frames):
    comparison = residential_base_gdf[[
        "origin_id",
        "LCODE_C1",
        "LCODE_C2",
        "LCODE_C3",
        "C1_NAME",
        "C2_NAME",
        "C3_NAME",
        "time_to_medical_min",
        "time_to_shelter_min",
        "medical_access_score_normal",
        "shelter_access_score_normal",
        "geometry"
    ]].copy()

    comparison["x_3826"] = comparison.geometry.x
    comparison["y_3826"] = comparison.geometry.y
    comparison = comparison.drop(columns="geometry")

    for scenario_df in scenario_frames:
        comparison = comparison.merge(scenario_df, on="origin_id", how="left")

    return comparison

for scenario_name in scenario_names:
    flood_tif_path = f"../data/Geo_RA/{scenario_name}_depth_max.GangKoudem.2022dem.tif"
    print(f"\n===== Start Processing Scenario: {scenario_name} =====")

    G_scenario_full, G_scenario_routing, edges_scenario_3826 = build_flood_scenario_graphs(
        G_base=G_3826,
        flood_tif_path=flood_tif_path,
        scenario_name=scenario_name,
        sample_dist=10,
        base_time_col="travel_time_min"
    )

    depth_col = f"{scenario_name}_max_flood_depth_m"
    factor_col = f"{scenario_name}_time_factor"
    blocked_col = f"{scenario_name}_is_blocked"
    scenario_time_col = f"{scenario_name}_travel_time_min"

    print(f"{scenario_name} scenario road network flood analysis and passability assessment completed")
    print("Total road segments:", len(edges_scenario_3826))
    print("Blocked road segments:", int(edges_scenario_3826[blocked_col].sum()))
    print("Passable road segments:", int((~edges_scenario_3826[blocked_col]).sum()))

    if scenario_name == "Q100":
        q100_edges_3826 = edges_scenario_3826.copy()

    residential_scenario_3826 = residential_base_3826.copy()
    medical_scenario_3826 = medical_dest_3826.copy()
    shelter_scenario_3826 = shelters_in_buffer_3826.copy()

    if "shelter_id" not in shelter_scenario_3826.columns:
        shelter_scenario_3826["shelter_id"] = range(len(shelter_scenario_3826))

    residential_scenario_3826["nearest_node"] = ox.distance.nearest_nodes(
        G_scenario_routing,
        X=residential_scenario_3826.geometry.x,
        Y=residential_scenario_3826.geometry.y
    )
    medical_scenario_3826["nearest_node"] = ox.distance.nearest_nodes(
        G_scenario_routing,
        X=medical_scenario_3826.geometry.x,
        Y=medical_scenario_3826.geometry.y
    )
    shelter_scenario_3826["nearest_node"] = ox.distance.nearest_nodes(
        G_scenario_routing,
        X=shelter_scenario_3826.geometry.x,
        Y=shelter_scenario_3826.geometry.y
    )

    residential_scenario_3826 = find_nearest_destination_in_scenario(
        G=G_scenario_routing,
        origins_gdf=residential_scenario_3826,
        destinations_gdf=medical_scenario_3826,
        scenario_name=scenario_name,
        weight_col=scenario_time_col,
        origin_node_col="nearest_node",
        dest_node_col="nearest_node",
        dest_id_col="medical_id",
        prefix="medical"
    )
    residential_scenario_3826 = find_nearest_destination_in_scenario(
        G=G_scenario_routing,
        origins_gdf=residential_scenario_3826,
        destinations_gdf=shelter_scenario_3826,
        scenario_name=scenario_name,
        weight_col=scenario_time_col,
        origin_node_col="nearest_node",
        dest_node_col="nearest_node",
        dest_id_col="shelter_id",
        prefix="shelter"
    )

    medical_time_col = f"time_to_medical_min_{scenario_name}"
    shelter_time_col = f"time_to_shelter_min_{scenario_name}"
    medical_id_col = f"nearest_medical_id_{scenario_name}"
    shelter_id_col = f"nearest_shelter_id_{scenario_name}"
    medical_score_scenario_col = f"medical_access_score_{scenario_name}"
    shelter_score_scenario_col = f"shelter_access_score_{scenario_name}"

    residential_scenario_3826[medical_score_scenario_col] = residential_scenario_3826[medical_time_col].apply(
        lambda x: calc_accessibility_score(x, max_accept_time_medical)
    )
    residential_scenario_3826[shelter_score_scenario_col] = residential_scenario_3826[shelter_time_col].apply(
        lambda x: calc_accessibility_score(x, max_accept_time_shelter)
    )

    print(f"{scenario_name} scenario residential accessibility score calculation completed")
    print("Residential points unable to reach medical destinations:", residential_scenario_3826[medical_time_col].isna().sum())
    print("Residential points unable to reach shelters:", residential_scenario_3826[shelter_time_col].isna().sum())

    m = build_scenario_map(
        scenario_name=scenario_name,
        flood_tif_path=flood_tif_path,
        edges_scenario_3826=edges_scenario_3826,
        residential_scenario_3826=residential_scenario_3826,
        medical_scenario_3826=medical_scenario_3826,
        shelter_scenario_3826=shelter_scenario_3826,
        depth_col=depth_col,
        factor_col=factor_col,
        blocked_col=blocked_col,
        scenario_time_col=scenario_time_col,
        medical_time_col=medical_time_col,
        shelter_time_col=shelter_time_col,
        medical_id_col=medical_id_col,
        shelter_id_col=shelter_id_col,
        medical_score_scenario_col=medical_score_scenario_col,
        shelter_score_scenario_col=shelter_score_scenario_col
    )

    html_path = output_dir / f"{scenario_name}_road_network_analysis_interactive_map.html"
    m.save(html_path)
    print("Exported interactive road network analysis map:", html_path.resolve())

    scenario_cols = [
        "origin_id",
        medical_id_col,
        medical_time_col,
        medical_score_scenario_col,
        shelter_id_col,
        shelter_time_col,
        shelter_score_scenario_col
    ]
    scenario_csv_frames.append(residential_scenario_3826[scenario_cols].copy())


comparison_df = build_comparison_dataframe(residential_base_3826, scenario_csv_frames)

csv_path = output_dir / "residential_accessibility_comparison.csv"
comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("Exported residential accessibility comparison CSV:", csv_path.resolve())

#%% 讀取人口數量分布圖層
population_gpkg_path = Path("../data/人口數量分布.gpkg")
population_gdf = gpd.read_file(population_gpkg_path)

if population_gdf.crs is None:
    population_gdf = population_gdf.set_crs(wgs84)
else:
    population_gdf = population_gdf.to_crs(wgs84)

population_col_candidates = [
    col for col in population_gdf.columns
    if "人口" in col and "比例" not in col and "面積" not in col
]

if not population_col_candidates:
    raise KeyError("No usable population field was found. Please verify the field names in the population distribution layer.")

population_value_col = population_col_candidates[0]
population_gdf[population_value_col] = pd.to_numeric(
    population_gdf[population_value_col],
    errors="coerce"
)
population_gdf = population_gdf.dropna(subset=[population_value_col]).copy()

preferred_population_cols = [
    "114年12月行政區人口統計 村里_屏東縣_人口數",
    "114年12月行政區人口統計_村里_屏東縣_人口數",
]

population_gdf = gpd.read_file(population_gpkg_path)
if population_gdf.crs is None:
    population_gdf = population_gdf.set_crs(wgs84)
else:
    population_gdf = population_gdf.to_crs(wgs84)

selected_population_col = next(
    (col for col in preferred_population_cols if col in population_gdf.columns),
    None
)

if selected_population_col is None:
    selected_population_col = next(
        (
            col for col in population_gdf.columns
            if "114年12月行政區人口統計" in str(col) and "人口數" in str(col)
        ),
        None
    )

if selected_population_col is None:
    available_columns = ", ".join(map(str, population_gdf.columns))
    raise KeyError(
        "The specified population field for exposure could not be found. "
        "Expected a field including \"114年12月行政區人口統計 村里_屏東縣_人口數\". "
        f"Available fields are: {available_columns}"
    )

population_value_col = selected_population_col
population_gdf[population_value_col] = pd.to_numeric(
    population_gdf[population_value_col],
    errors="coerce"
)
population_gdf = population_gdf.dropna(subset=[population_value_col]).copy()
print(f"Exposure uses population field: {population_value_col}")

if population_gdf.empty:
    raise ValueError("The population distribution layer does not contain usable numeric values.")

population_center = (
    population_gdf.to_crs(twd97_crs)
    .dissolve()
    .centroid
    .to_crs(wgs84)
    .iloc[0]
)

population_min = float(population_gdf[population_value_col].min())
population_max = float(population_gdf[population_value_col].max())

if population_min == population_max:
    population_max = population_min + 1

population_gdf["exposure_norm"] = (
    (population_gdf[population_value_col] - population_min)
    / (population_max - population_min)
)
population_gdf["exposure_display"] = population_gdf["exposure_norm"].apply(
    lambda value: f"{value:.2f}" if pd.notna(value) else None
)

#%% 建立暴露度分級地圖
population_quantiles = np.quantile(
    population_gdf[population_value_col].to_numpy(),
    np.linspace(0, 1, 6)
)

def round_to_preferred_population_break(value):
    return int(5 * round(float(value) / 5))

population_bins = np.array(
    [round_to_preferred_population_break(value) for value in population_quantiles],
    dtype=float
)

for i in range(1, len(population_bins)):
    if population_bins[i] <= population_bins[i - 1]:
        population_bins[i] = population_bins[i - 1] + 5

if population_bins[0] > population_min:
    population_bins[0] = int(5 * np.floor(population_min / 5))

if population_bins[-1] < population_max:
    population_bins[-1] = int(5 * np.ceil(population_max / 5))

population_colors = [
    "#fff5eb",
    "#fdd0a2",
    "#fdae6b",
    "#fd8d3c",
    "#7f2704",
]

population_map_discrete = folium.Map(
    location=[population_center.y, population_center.x],
    zoom_start=12,
    tiles="CartoDB positron"
)

def population_style_function_discrete(feature):
    population_value = feature["properties"].get(population_value_col)
    population_norm = feature["properties"].get("exposure_norm")
    if population_value is None or pd.isna(population_value) or population_norm is None or pd.isna(population_norm):
        return {
            "fillColor": "transparent",
            "color": "#666666",
            "weight": 0.8,
            "fillOpacity": 0.0,
        }

    color_idx = np.digitize([population_value], population_bins[1:-1], right=False)[0]
    return {
        "fillColor": population_colors[color_idx],
        "color": "#666666",
        "weight": 0.8,
        "fillOpacity": 0.7,
    }

population_legend_items = []
for idx, color in enumerate(population_colors):
    lower = int(population_bins[idx])
    upper = int(population_bins[idx + 1])
    lower_norm = (lower - population_min) / (population_max - population_min)
    upper_norm = (upper - population_min) / (population_max - population_min)
    lower_norm = min(max(lower_norm, 0.0), 1.0)
    upper_norm = min(max(upper_norm, 0.0), 1.0)
    label = f"{lower_norm:.2f} - {upper_norm:.2f} ({lower:,} - {upper:,})"
    population_legend_items.append((color, label))

population_items_html = "".join(
    [
        (
            "<div style='display:flex; align-items:center; margin-bottom:6px;'>"
            f"<span style='display:inline-block; width:18px; height:12px; background:{color}; "
            "border:1px solid #666; margin-right:8px;'></span>"
            f"<span>{label}</span>"
            "</div>"
        )
        for color, label in population_legend_items
    ]
)

population_legend_html = f"""
<div id="exposure-legend" style="
    display:none;
    position: fixed;
    bottom: 235px;
    left: 40px;
    width: 220px;
    z-index: 9999;
    background: white;
    border: 2px solid #666;
    border-radius: 6px;
    padding: 12px 12px 10px 12px;
    font-size: 13px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
">
    <div style="font-weight:700; margin-bottom:10px;">Exposure (Total Population)</div>
    {population_items_html}
</div>
"""

population_popup = folium.GeoJsonPopup(
    fields=["COUNTYNAME", "TOWNNAME", "VILLNAME", population_value_col, "exposure_display"],
    aliases=["County", "Township", "Village", "Total Population", "Exposure Index"],
    localize=True,
    labels=True,
    style="background-color: white;"
)

population_tooltip = folium.GeoJsonTooltip(
    fields=["COUNTYNAME", "TOWNNAME", "VILLNAME", population_value_col, "exposure_display"],
    aliases=["County", "Township", "Village", "Total Population", "Exposure Index"],
    localize=True,
    sticky=True
)

if "comparison_df" not in locals():
    comparison_df = build_comparison_dataframe(residential_base_3826, scenario_csv_frames)

required_q100_cols = ["medical_access_score_Q100", "shelter_access_score_Q100"]
missing_q100_cols = [col for col in required_q100_cols if col not in comparison_df.columns]
if missing_q100_cols:
    raise KeyError(
        "comparison_df is missing Q100 scenario fields: "
        + ", ".join(missing_q100_cols)
        + ". Please run the full scenario workflow above first."
    )

vulnerability_points_3826 = gpd.GeoDataFrame(
    comparison_df.copy(),
    geometry=gpd.points_from_xy(comparison_df["x_3826"], comparison_df["y_3826"]),
    crs=twd97_crs
)
vulnerability_points_wgs84 = vulnerability_points_3826.to_crs(wgs84)
vulnerability_points_wgs84["vulnerability_score_Q100"] = (
    1 - (
        vulnerability_points_wgs84["medical_access_score_Q100"].fillna(0)
        + vulnerability_points_wgs84["shelter_access_score_Q100"].fillna(0)
    ) / 2
).clip(lower=0.0, upper=1.0)

vulnerability_join = gpd.sjoin(
    vulnerability_points_wgs84[["vulnerability_score_Q100", "geometry"]],
    population_gdf[["geometry"]],
    how="left",
    predicate="intersects"
)

vulnerability_stats = (
    vulnerability_join.dropna(subset=["index_right"])
    .groupby("index_right")
    .agg(
        vulnerability_avg_Q100=("vulnerability_score_Q100", "mean"),
        residential_point_count=("vulnerability_score_Q100", "size")
    )
)

population_gdf["vulnerability_avg_Q100"] = population_gdf.index.map(
    vulnerability_stats["vulnerability_avg_Q100"]
)
population_gdf["residential_point_count"] = population_gdf.index.map(
    vulnerability_stats["residential_point_count"]
)

vulnerability_bins = np.linspace(0, 1, 6)
vulnerability_colors = [
    "#f7fcf0",
    "#ccebc5",
    "#7bccc4",
    "#2b8cbe",
    "#084081",
]

if q100_edges_3826 is None:
    raise ValueError("Q100 road network analysis results were not found, so the hazard layer cannot be created.")

population_gdf_3826 = population_gdf.to_crs(twd97_crs).copy()
population_gdf_3826["polygon_id"] = population_gdf_3826.index
q100_blocked_col = "Q100_is_blocked"
q100_edges_for_hazard = q100_edges_3826[["geometry", q100_blocked_col]].copy()
q100_edges_for_hazard = q100_edges_for_hazard[q100_edges_for_hazard.geometry.notna()].copy()

total_road_segments = gpd.overlay(
    q100_edges_for_hazard,
    population_gdf_3826[["polygon_id", "geometry"]],
    how="intersection",
    keep_geom_type=False
)
total_road_segments["segment_length_m"] = total_road_segments.geometry.length

blocked_road_segments = gpd.overlay(
    q100_edges_for_hazard[q100_edges_for_hazard[q100_blocked_col]].copy(),
    population_gdf_3826[["polygon_id", "geometry"]],
    how="intersection",
    keep_geom_type=False
)
blocked_road_segments["segment_length_m"] = blocked_road_segments.geometry.length

total_road_length_by_polygon = total_road_segments.groupby("polygon_id")["segment_length_m"].sum()
blocked_road_length_by_polygon = blocked_road_segments.groupby("polygon_id")["segment_length_m"].sum()

population_gdf["total_road_length_m_Q100"] = population_gdf.index.map(total_road_length_by_polygon)
population_gdf["blocked_road_length_m_Q100"] = (
    population_gdf.index.map(blocked_road_length_by_polygon).fillna(0.0)
)
population_gdf["hazard_ratio_Q100"] = np.where(
    population_gdf["total_road_length_m_Q100"].fillna(0) > 0,
    population_gdf["blocked_road_length_m_Q100"] / population_gdf["total_road_length_m_Q100"],
    np.nan
)

valid_hazard_ratio = population_gdf["hazard_ratio_Q100"].dropna()
if valid_hazard_ratio.empty:
    hazard_ratio_min = 0.0
    hazard_ratio_max = 1.0
else:
    hazard_ratio_min = float(valid_hazard_ratio.min())
    hazard_ratio_max = float(valid_hazard_ratio.max())

if hazard_ratio_min == hazard_ratio_max:
    hazard_ratio_max = hazard_ratio_min + 0.000001

population_gdf["hazard_norm_Q100"] = np.where(
    population_gdf["hazard_ratio_Q100"].notna(),
    (population_gdf["hazard_ratio_Q100"] - hazard_ratio_min) / (hazard_ratio_max - hazard_ratio_min),
    np.nan
)
population_gdf["hazard_display_Q100"] = population_gdf.apply(
    lambda row: (
        f"{row['hazard_norm_Q100']:.2f} ({row['hazard_ratio_Q100'] * 100:.1f}% roads blocked)"
        if pd.notna(row["hazard_norm_Q100"]) and pd.notna(row["hazard_ratio_Q100"])
        else None
    ),
    axis=1
)
population_gdf["hazard_ratio_display_Q100"] = population_gdf["hazard_ratio_Q100"].apply(
    lambda value: f"{value * 100:.1f}% roads blocked" if pd.notna(value) else None
)

hazard_ratio_bins = np.linspace(0, 1, 6)
hazard_ratio_colors = [
    "#ffffcc",
    "#ffeda0",
    "#feb24c",
    "#f03b20",
    "#bd0026",
]


def hazard_ratio_style_function(feature):
    hazard_ratio_value = feature["properties"].get("hazard_norm_Q100")
    if hazard_ratio_value is None or pd.isna(hazard_ratio_value):
        return {
            "fillColor": "transparent",
            "color": "#666666",
            "weight": 0.8,
            "fillOpacity": 0.0,
        }

    color_idx = np.digitize([hazard_ratio_value], hazard_ratio_bins[1:-1], right=False)[0]
    return {
        "fillColor": hazard_ratio_colors[color_idx],
        "color": "#404040",
        "weight": 0.8,
        "fillOpacity": 0.75,
    }


hazard_ratio_legend_items = []
for idx, color in enumerate(hazard_ratio_colors):
    lower = hazard_ratio_bins[idx]
    upper = hazard_ratio_bins[idx + 1]
    label = f"{lower:.2f} - {upper:.2f}"
    hazard_ratio_legend_items.append((color, label))

hazard_ratio_items_html = "".join(
    [
        (
            "<div style='display:flex; align-items:center; margin-bottom:6px;'>"
            f"<span style='display:inline-block; width:18px; height:12px; background:{color}; "
            "border:1px solid #666; margin-right:8px;'></span>"
            f"<span>{label}</span>"
            "</div>"
        )
        for color, label in hazard_ratio_legend_items
    ]
)

hazard_q100_tif_path = Path("../data/Geo_RA/Q100_depth_max.GangKoudem.2022dem.tif")
flood_potential_layer = add_flood_raster_to_folium(
    population_map_discrete,
    tif_path=hazard_q100_tif_path,
    layer_name="Flood Potential Map - Q100",
    opacity=0.80,
    show=True,
    return_layer=True
)

flood_potential_legend_html = """
<div id="flood-potential-legend" style="
    display:none;
    position: fixed;
    bottom: 30px;
    right: 40px;
    width: 220px;
    z-index: 9999;
    background: white;
    border: 2px solid #666;
    border-radius: 6px;
    padding: 12px 12px 10px 12px;
    font-size: 13px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
">
    <div style="font-weight:700; margin-bottom:10px;">Flood Potential Map - Q100</div>
    <div style='display:flex; align-items:center; margin-bottom:6px;'>
        <span style='display:inline-block; width:18px; height:12px; background:#fff5f0; border:1px solid #666; margin-right:8px;'></span>
        <span>0 - 0.05 m</span>
    </div>
    <div style='display:flex; align-items:center; margin-bottom:6px;'>
        <span style='display:inline-block; width:18px; height:12px; background:#fcbba1; border:1px solid #666; margin-right:8px;'></span>
        <span>0.05 - 0.10 m</span>
    </div>
    <div style='display:flex; align-items:center; margin-bottom:6px;'>
        <span style='display:inline-block; width:18px; height:12px; background:#fc9272; border:1px solid #666; margin-right:8px;'></span>
        <span>0.10 - 0.25 m</span>
    </div>
    <div style='display:flex; align-items:center; margin-bottom:6px;'>
        <span style='display:inline-block; width:18px; height:12px; background:#ef3b2c; border:1px solid #666; margin-right:8px;'></span>
        <span>0.25 - 0.50 m</span>
    </div>
    <div style='display:flex; align-items:center; margin-bottom:6px;'>
        <span style='display:inline-block; width:18px; height:12px; background:#99000d; border:1px solid #666; margin-right:8px;'></span>
        <span>> 0.50 m</span>
    </div>
</div>
"""

hazard_popup = folium.GeoJsonPopup(
    fields=["COUNTYNAME", "TOWNNAME", "VILLNAME", "blocked_road_length_m_Q100", "total_road_length_m_Q100", "hazard_display_Q100", "hazard_ratio_display_Q100"],
    aliases=["County", "Township", "Village", "Blocked Road Length (m)", "Total Road Length (m)", "Hazard Index", "Flooded Road Ratio"],
    localize=True,
    labels=True,
    style="background-color: white;"
)

hazard_tooltip = folium.GeoJsonTooltip(
    fields=["COUNTYNAME", "TOWNNAME", "VILLNAME", "blocked_road_length_m_Q100", "total_road_length_m_Q100", "hazard_display_Q100", "hazard_ratio_display_Q100"],
    aliases=["County", "Township", "Village", "Blocked Road Length (m)", "Total Road Length (m)", "Hazard Index", "Flooded Road Ratio"],
    localize=True,
    sticky=True
)

hazard_layer = folium.GeoJson(
    data=population_gdf,
    name="Hazard (Road Flooding)",
    style_function=hazard_ratio_style_function,
    popup=hazard_popup,
    tooltip=hazard_tooltip,
    highlight_function=lambda _: {
        "weight": 2.0,
        "color": "#252525",
        "fillOpacity": 0.85,
    },
)
hazard_layer.add_to(population_map_discrete)

hazard_legend_html = f"""
<div id="hazard-legend" style="
    display:none;
    position: fixed;
    bottom: 440px;
    left: 40px;
    width: 220px;
    z-index: 9999;
    background: white;
    border: 2px solid #666;
    border-radius: 6px;
    padding: 12px 12px 10px 12px;
    font-size: 13px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
">
    <div style="font-weight:700; margin-bottom:10px;">Hazard (Road Flooding)</div>
    {hazard_ratio_items_html}
</div>
"""

exposure_layer = folium.GeoJson(
    data=population_gdf,
    name="Exposure (Total Population)",
    style_function=population_style_function_discrete,
    popup=population_popup,
    tooltip=population_tooltip,
    highlight_function=lambda _: {
        "weight": 2.0,
        "color": "#252525",
        "fillOpacity": 0.85,
    },
)
exposure_layer.add_to(population_map_discrete)

def vulnerability_style_function(feature):
    vulnerability_value = feature["properties"].get("vulnerability_avg_Q100")
    if vulnerability_value is None or pd.isna(vulnerability_value):
        return {
            "fillColor": "#bdbdbd",
            "color": "#666666",
            "weight": 0.8,
            "fillOpacity": 0.75,
        }

    color_idx = np.digitize([vulnerability_value], vulnerability_bins[1:-1], right=False)[0]
    return {
        "fillColor": vulnerability_colors[color_idx],
        "color": "#404040",
        "weight": 0.8,
        "fillOpacity": 0.75,
    }

vulnerability_legend_items = []
for idx, color in enumerate(vulnerability_colors):
    lower = vulnerability_bins[idx]
    upper = vulnerability_bins[idx + 1]
    label = f"{lower:.2f} - {upper:.2f}"
    vulnerability_legend_items.append((color, label))

vulnerability_items_html = "".join(
    [
        (
            "<div style='display:flex; align-items:center; margin-bottom:6px;'>"
            f"<span style='display:inline-block; width:18px; height:12px; background:{color}; "
            "border:1px solid #666; margin-right:8px;'></span>"
            f"<span>{label}</span>"
            "</div>"
        )
        for color, label in vulnerability_legend_items
    ]
)
vulnerability_items_html += (
    "<div style='display:flex; align-items:center; margin-bottom:6px;'>"
    "<span style='display:inline-block; width:18px; height:12px; background:#bdbdbd; "
    "border:1px solid #666; margin-right:8px;'></span>"
    "<span>No Residential Points</span>"
    "</div>"
)

vulnerability_popup = folium.GeoJsonPopup(
    fields=["COUNTYNAME", "TOWNNAME", "VILLNAME", "vulnerability_avg_Q100", "residential_point_count"],
    aliases=["County", "Township", "Village", "Average Vulnerability under Q100", "Residential Point Count"],
    localize=True,
    labels=True,
    style="background-color: white;"
)

vulnerability_tooltip = folium.GeoJsonTooltip(
    fields=["COUNTYNAME", "TOWNNAME", "VILLNAME", "vulnerability_avg_Q100", "residential_point_count"],
    aliases=["County", "Township", "Village", "Average Vulnerability under Q100", "Residential Point Count"],
    localize=True,
    sticky=True
)

vulnerability_layer = folium.GeoJson(
    data=population_gdf,
    name="Vulnerability (Medical and Shelter Accessibility - Q100)",
    style_function=vulnerability_style_function,
    popup=vulnerability_popup,
    tooltip=vulnerability_tooltip,
    highlight_function=lambda _: {
        "weight": 2.0,
        "color": "#252525",
        "fillOpacity": 0.85,
    },
)
vulnerability_layer.add_to(population_map_discrete)

population_gdf["risk_Q100"] = np.where(
    population_gdf[["hazard_norm_Q100", "exposure_norm", "vulnerability_avg_Q100"]].notna().all(axis=1),
    (
        population_gdf["hazard_norm_Q100"]
        + population_gdf["exposure_norm"]
        + population_gdf["vulnerability_avg_Q100"]
    ) / 3,
    np.nan
)

risk_bins = np.linspace(0, 1, 6)
risk_colors = [
    "#ffffcc",
    "#fed976",
    "#feb24c",
    "#fd8d3c",
    "#bd0026",
]


def risk_style_function(feature):
    risk_value = feature["properties"].get("risk_Q100")
    if risk_value is None or pd.isna(risk_value):
        return {
            "fillColor": "#bdbdbd",
            "color": "#666666",
            "weight": 0.8,
            "fillOpacity": 0.75,
        }

    color_idx = np.digitize([risk_value], risk_bins[1:-1], right=False)[0]
    return {
        "fillColor": risk_colors[color_idx],
        "color": "#404040",
        "weight": 0.8,
        "fillOpacity": 0.75,
    }


risk_legend_items = []
for idx, color in enumerate(risk_colors):
    lower = risk_bins[idx]
    upper = risk_bins[idx + 1]
    label = f"{lower:.2f} - {upper:.2f}"
    risk_legend_items.append((color, label))

risk_items_html = "".join(
    [
        (
            "<div style='display:flex; align-items:center; margin-bottom:6px;'>"
            f"<span style='display:inline-block; width:18px; height:12px; background:{color}; "
            "border:1px solid #666; margin-right:8px;'></span>"
            f"<span>{label}</span>"
            "</div>"
        )
        for color, label in risk_legend_items
    ]
)
risk_items_html += (
    "<div style='display:flex; align-items:center; margin-bottom:6px;'>"
    "<span style='display:inline-block; width:18px; height:12px; background:#bdbdbd; "
    "border:1px solid #666; margin-right:8px;'></span>"
    "<span>No Data</span>"
    "</div>"
)

risk_popup = folium.GeoJsonPopup(
    fields=["COUNTYNAME", "TOWNNAME", "VILLNAME", "hazard_norm_Q100", "exposure_norm", "vulnerability_avg_Q100", "risk_Q100"],
    aliases=["County", "Township", "Village", "Hazard (Road Flooding)", "Exposure (Total Population)", "Vulnerability (Medical and Shelter Accessibility - Q100)", "Risk"],
    localize=True,
    labels=True,
    style="background-color: white; min-width: 460px; white-space: normal;"
)

risk_tooltip = folium.GeoJsonTooltip(
    fields=["COUNTYNAME", "TOWNNAME", "VILLNAME", "hazard_norm_Q100", "exposure_norm", "vulnerability_avg_Q100", "risk_Q100"],
    aliases=["County", "Township", "Village", "Hazard (Road Flooding)", "Exposure (Total Population)", "Vulnerability (Medical and Shelter Accessibility - Q100)", "Risk"],
    localize=True,
    sticky=True
)

risk_layer = folium.GeoJson(
    data=population_gdf,
    name="Risk (Vulnerability, Hazard, Exposure)",
    style_function=risk_style_function,
    popup=risk_popup,
    tooltip=risk_tooltip,
    highlight_function=lambda _: {
        "weight": 2.0,
        "color": "#252525",
        "fillOpacity": 0.85,
    },
)
risk_layer.add_to(population_map_discrete)

risk_legend_html = f"""
<div id="risk-legend" style="
    display:none;
    position: fixed;
    bottom: 235px;
    right: 40px;
    width: 220px;
    z-index: 9999;
    background: white;
    border: 2px solid #666;
    border-radius: 6px;
    padding: 12px 12px 10px 12px;
    font-size: 13px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
">
    <div style="font-weight:700; margin-bottom:10px;">Risk (Vulnerability, Hazard, Exposure)</div>
    {risk_items_html}
</div>
"""

vulnerability_legend_html = f"""
<div id="vulnerability-legend" style="
    display:none;
    position: fixed;
    bottom: 30px;
    left: 40px;
    width: 220px;
    z-index: 9999;
    background: white;
    border: 2px solid #666;
    border-radius: 6px;
    padding: 12px 12px 10px 12px;
    font-size: 13px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
">
    <div style="font-weight:700; margin-bottom:10px;">Vulnerability (Medical and Shelter Accessibility - Q100)</div>
    {vulnerability_items_html}
</div>
"""

legend_control_html = f"""
{{% macro html(this, kwargs) %}}
{flood_potential_legend_html}
{risk_legend_html}
{hazard_legend_html}
{vulnerability_legend_html}
{population_legend_html}
<style>
    .leaflet-popup-content table {{
        width: auto !important;
    }}
    .leaflet-popup-content th,
    .leaflet-popup-content td {{
        white-space: nowrap !important;
        vertical-align: top;
    }}
    .leaflet-popup-content th {{
        padding-right: 12px;
    }}
</style>
<script>
document.addEventListener("DOMContentLoaded", function() {{
    var map = {population_map_discrete.get_name()};
    var floodPotentialLegend = document.getElementById("flood-potential-legend");
    var riskLegend = document.getElementById("risk-legend");
    var hazardLegend = document.getElementById("hazard-legend");
    var vulnerabilityLegend = document.getElementById("vulnerability-legend");
    var exposureLegend = document.getElementById("exposure-legend");
    var floodPotentialLayer = {flood_potential_layer.get_name()};
    var riskLayer = {risk_layer.get_name()};
    var hazardLayer = {hazard_layer.get_name()};
    var exposureLayer = {exposure_layer.get_name()};
    var vulnerabilityLayer = {vulnerability_layer.get_name()};

    function updateLegendLayout() {{
        var gap = 18;
        var leftBottomStart = 12;
        var rightBottomStart = 30;

        var leftLegends = [vulnerabilityLegend, exposureLegend, hazardLegend];
        var leftBottom = leftBottomStart;
        leftLegends.forEach(function(legend) {{
            if (legend && legend.style.display !== "none") {{
                legend.style.left = "40px";
                legend.style.right = "";
                legend.style.bottom = leftBottom + "px";
                leftBottom += legend.offsetHeight + gap;
            }}
        }});

        var rightLegends = [floodPotentialLegend, riskLegend];
        var rightBottom = rightBottomStart;
        rightLegends.forEach(function(legend) {{
            if (legend && legend.style.display !== "none") {{
                legend.style.right = "40px";
                legend.style.left = "";
                legend.style.bottom = rightBottom + "px";
                rightBottom += legend.offsetHeight + gap;
            }}
        }});
    }}

    function updateLegendByLayer(layer, visible) {{
        if (layer === floodPotentialLayer && floodPotentialLegend) {{
            floodPotentialLegend.style.display = visible ? "block" : "none";
        }}
        if (layer === riskLayer && riskLegend) {{
            riskLegend.style.display = visible ? "block" : "none";
        }}
        if (layer === hazardLayer && hazardLegend) {{
            hazardLegend.style.display = visible ? "block" : "none";
        }}
        if (layer === vulnerabilityLayer && vulnerabilityLegend) {{
            vulnerabilityLegend.style.display = visible ? "block" : "none";
        }}
        if (layer === exposureLayer && exposureLegend) {{
            exposureLegend.style.display = visible ? "block" : "none";
        }}
        updateLegendLayout();
    }}

    updateLegendByLayer(floodPotentialLayer, map.hasLayer(floodPotentialLayer));
    updateLegendByLayer(riskLayer, map.hasLayer(riskLayer));
    updateLegendByLayer(hazardLayer, map.hasLayer(hazardLayer));
    updateLegendByLayer(vulnerabilityLayer, map.hasLayer(vulnerabilityLayer));
    updateLegendByLayer(exposureLayer, map.hasLayer(exposureLayer));
    updateLegendLayout();

    map.on("overlayadd", function(e) {{
        if (e.layer) {{
            updateLegendByLayer(e.layer, true);
        }}
    }});

    map.on("overlayremove", function(e) {{
        if (e.layer) {{
            updateLegendByLayer(e.layer, false);
        }}
    }});
}});
</script>
{{% endmacro %}}
"""

legend_macro = MacroElement()
legend_macro._template = Template(legend_control_html)
population_map_discrete.get_root().add_child(legend_macro)
folium.LayerControl(collapsed=False).add_to(population_map_discrete)

population_html_path = output_dir / "exposure_total_population.html"
population_map_discrete.save(population_html_path)
print("Exported interactive exposure / hazard / vulnerability / risk map:", population_html_path.resolve())



