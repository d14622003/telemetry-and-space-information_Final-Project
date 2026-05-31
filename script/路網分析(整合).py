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
max_accept_time_medical = 20
max_accept_time_shelter = 20

twd97_crs = "EPSG:3826"  # TWD97 / TM2
wgs84 = "EPSG:4326"  # WGS84

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
    "01": "農業使用土地",
    "02": "林業使用土地",
    "03": "交通使用土地",
    "04": "水利使用土地",
    "05": "建築使用土地",
    "06": "公共使用土地",
    "07": "遊憩使用土地",
    "08": "礦鹽使用土地",
    "09": "其他使用土地",
}

c2_map = {
    "0502": "住宅",
    "0603": "醫療保健",
}

c3_map = {
    "050201": "純住宅",
    "050202": "混合住宅",
    "050203": "鄉村住宅",
    "050204": "其他住宅",
    "060300": "醫療保健",
}

lu_3826["C1_NAME"] = lu_3826["LCODE_C1"].map(c1_map)
lu_3826["C2_NAME"] = lu_3826["LCODE_C2"].map(c2_map)
lu_3826["C3_NAME"] = lu_3826["LCODE_C3"].map(c3_map)

#%% 2. Extract residential and medical land-use polygons

residential_3826 = lu_3826[lu_3826["LCODE_C2"] == "0502"].copy()
medical_3826 = lu_3826[lu_3826["LCODE_C2"] == "0603"].copy()

print("住宅用地筆數：", len(residential_3826))
print("醫療保健用地筆數：", len(medical_3826))

residential_3826.head(3)

#%% 2.1 Read shelters CSV

shelters_csv_path = Path("../data/避難收容處所_清理後.csv")

if shelters_csv_path.exists():
    shelters_csv = pd.read_csv(shelters_csv_path, encoding="utf-8-sig")
    print("避難收容處所總筆數：", len(shelters_csv))

    if "座標有效性" in shelters_csv.columns:
        valid_shelters = shelters_csv[
            shelters_csv["座標有效性"] == "有效"
        ].copy()
    else:
        print('Warning: "座標有效性" column not found, using all records')
        valid_shelters = shelters_csv.copy()

    print("有效避難收容處所筆數：", len(valid_shelters))
else:
    raise FileNotFoundError(f"找不到避難收容處所 CSV：{shelters_csv_path}")

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

print("避難所 CRS：", shelters_3826.crs)

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

print("研究區內避難所筆數：", len(shelters_in_study_3826))

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

print("研究區外擴 1 km 內避難所筆數：", len(shelters_in_buffer_3826))

#%% 3. Convert polygons to centroids

residential_origins_3826 = residential_3826.copy()
residential_origins_3826["geometry"] = residential_origins_3826.geometry.centroid
residential_origins_3826["origin_id"] = range(len(residential_origins_3826))

medical_dest_3826 = medical_3826.copy()
medical_dest_3826["geometry"] = medical_dest_3826.geometry.centroid
medical_dest_3826["medical_id"] = range(len(medical_dest_3826))

print("住宅起點筆數：", len(residential_origins_3826))
print("醫療目的地筆數：", len(medical_dest_3826))
print("避難所目的地筆數：", len(shelters_in_buffer_3826))

shelters_in_buffer_3826.head(1)

#%% 4. Build road network

road_boundary_3826 = study_area_3826.copy()
road_boundary_3826["geometry"] = road_boundary_3826.geometry.buffer(500)

road_boundary_wgs84 = road_boundary_3826.to_crs(wgs84)
road_boundary_polygon = road_boundary_wgs84.geometry.iloc[0]

print("路網抓取範圍 CRS：", road_boundary_wgs84.crs)

#%% 4.1 Download road network from OSM

G_4326 = ox.graph_from_polygon(
    road_boundary_polygon,
    network_type="all",
    simplify=True,
    retain_all=True,
    truncate_by_edge=True
)

print("OSM 路網下載完成")

#%% 4.2 Project road network to EPSG:3826

G_3826 = ox.project_graph(G_4326, to_crs=twd97_crs)
nodes_3826, edges_3826 = ox.graph_to_gdfs(G_3826)

print("路網節點數：", len(nodes_3826))
print("路網郊數：", len(edges_3826))
print("路網 CRS：", edges_3826.crs)

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
    name="研究區域",
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
    name="住宅用地 0502",
    style_function=lambda x: {
        "fillColor": "yellow",
        "color": "orange",
        "weight": 1,
        "fillOpacity": 0.45,
        "opacity": 0.9
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["LCODE_C2", "LCODE_C3"],
        aliases=["第二級代碼", "第三級代碼"]
    )
).add_to(m)

folium.GeoJson(
    medical_wgs84,
    name="醫療保健用地 0603",
    style_function=lambda x: {
        "fillColor": "blue",
        "color": "blue",
        "weight": 1,
        "fillOpacity": 0.35,
        "opacity": 0.9
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["LCODE_C2", "LCODE_C3"],
        aliases=["第二級代碼", "第三級代碼"]
    )
).add_to(m)


# ------------------------------------------------------------
# 5. 路網圖層
# ------------------------------------------------------------

folium.GeoJson(
    edges_wgs84,
    name="路網",
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
            "道路類型", "速度(km/h)", "長度(m)", "通行時間(min)"
        ]
    )
).add_to(m)


# ------------------------------------------------------------
# 6. 建立可開關的點位圖層
# ------------------------------------------------------------

road_node_layer = folium.FeatureGroup(
    name="路網節點",
    show=False
)

residential_origin_layer = folium.FeatureGroup(
    name="住宅起點 centroid",
    show=True
)

medical_dest_layer = folium.FeatureGroup(
    name="醫療目的地 centroid",
    show=True
)

shelter_layer = folium.FeatureGroup(
    name="避難所",
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
        f"住宅起點 ID: {row.get('origin_id', '')}<br>"
        f"第一級分類: {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
        f"第二級分類: {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
        f"第三級分類: {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}"
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
        f"醫療目的地 ID: {row.get('medical_id', '')}<br>"
        f"第一級分類: {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
        f"第二級分類: {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
        f"第三級分類: {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}"
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
        f"避難所名稱: {row.get('避難收容處所名稱', '')}<br>"
        f"縣市鄉鎮: {row.get('縣市及鄉鎮市區', '')}<br>"
        f"村里: {row.get('村里', '')}<br>"
        f"地址: {row.get('避難收容處所地址', '')}<br>"
        f"預計收容村里: {row.get('預計收容村里', '')}<br>"
        f"預計收容人數: {row.get('預計收容人數', '')}<br>"
        f"適用災害類別: {row.get('適用災害類別', '')}<br>"
        f"管理人: {row.get('管理人姓名', '')}<br>"
        f"管理人電話: {row.get('管理人電話', '')}<br>"
        f"經度: {row.get('經度', '')}<br>"
        f"緯度: {row.get('緯度', '')}"
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

html_path = output_dir / "路網分析結果.html"

m.save(html_path)

print("互動式地圖已儲存至：", html_path)

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
    print("全部有效值 min:", np.nanmin(arr))
    print("全部有效值 max:", np.nanmax(arr))
    print("全部有效值 mean:", np.nanmean(arr))
    print("全部有效值 median:", np.nanmedian(arr))

    print("大於 0 的像元數:", arr_positive.size)

    if arr_positive.size > 0:
        print("淹水深度 > 0 min:", np.nanmin(arr_positive))
        print("淹水深度 > 0 max:", np.nanmax(arr_positive))
        print("淹水深度 > 0 mean:", np.nanmean(arr_positive))
        print("淹水深度 > 0 median:", np.nanmedian(arr_positive))
        print("95百分位數:", np.nanpercentile(arr_positive, 95))
        print("99百分位數:", np.nanpercentile(arr_positive, 99))

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

print("住宅、醫療、避難所已重新 snap 到分析用路網")

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

print("無法到達醫療場所的住宅點數量:")
print(residential_origins_3826["time_to_medical_min"].isna().sum())

print("無法到達避難所的住宅點數量:")
print(residential_origins_3826["time_to_shelter_min"].isna().sum())

print("醫療行駛距離統計:")
display(residential_origins_3826["distance_to_medical_m"].describe())

print("醫療行駛時間統計:")
display(residential_origins_3826["time_to_medical_min"].describe())

print("避難所行駛距離統計:")
display(residential_origins_3826["distance_to_shelter_m"].describe())

print("避難所行駛時間統計:")
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
    layer_name="淹水深度",
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


def add_flood_legend_to_folium(m, title="淹水潛勢（公尺）"):
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
    <div style="
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
    residential_origin_layer = folium.FeatureGroup(name="住宅起點 centroid", show=True)

    for _, row in residential_result_wgs84.iterrows():
        res_lon = row.geometry.x
        res_lat = row.geometry.y

        med_score_normal = row.get("medical_access_score_normal", 0.0)
        shel_score_normal = row.get("shelter_access_score_normal", 0.0)
        med_score_scenario = row.get(medical_score_scenario_col, 0.0)
        shel_score_scenario = row.get(shelter_score_scenario_col, 0.0)

        medical_text_normal = (
            "一般情境最近醫療：無法到達<br>"
            f"一般情境醫療可及性分數：{med_score_normal:.2f}"
        )
        shelter_text_normal = (
            "一般情境最近避難所：無法到達<br>"
            f"一般情境避難可及性分數：{shel_score_normal:.2f}"
        )
        medical_text_scenario = (
            f"{scenario_name} 情境最近醫療：無法到達<br>"
            f"{scenario_name} 情境醫療可及性分數：{med_score_scenario:.2f}"
        )
        shelter_text_scenario = (
            f"{scenario_name} 情境最近避難所：無法到達<br>"
            f"{scenario_name} 情境避難可及性分數：{shel_score_scenario:.2f}"
        )

        if pd.notna(row.get("nearest_medical_id", np.nan)):
            medical_row = medical_lookup.loc[row["nearest_medical_id"]]
            medical_text_normal = (
                f"一般情境最近醫療 ID：{row['nearest_medical_id']}<br>"
                f"經度：{medical_row.geometry.x:.2f}<br>"
                f"緯度：{medical_row.geometry.y:.2f}<br>"
                f"最快通行時間：{row['time_to_medical_min']:.2f} 分鐘<br>"
                f"一般情境醫療可及性分數：{med_score_normal:.2f}"
            )

        if pd.notna(row.get("nearest_shelter_id", np.nan)):
            shelter_row = shelter_lookup.loc[row["nearest_shelter_id"]]
            shelter_text_normal = (
                f"一般情境最近避難所 ID：{row['nearest_shelter_id']}<br>"
                f"經度：{shelter_row.geometry.x:.2f}<br>"
                f"緯度：{shelter_row.geometry.y:.2f}<br>"
                f"最快通行時間：{row['time_to_shelter_min']:.2f} 分鐘<br>"
                f"一般情境避難可及性分數：{shel_score_normal:.2f}"
            )

        if pd.notna(row.get(medical_id_col, np.nan)):
            medical_row_s = medical_lookup.loc[row[medical_id_col]]
            medical_text_scenario = (
                f"{scenario_name} 情境最近醫療 ID：{row[medical_id_col]}<br>"
                f"經度：{medical_row_s.geometry.x:.2f}<br>"
                f"緯度：{medical_row_s.geometry.y:.2f}<br>"
                f"最快通行時間：{row[medical_time_col]:.2f} 分鐘<br>"
                f"{scenario_name} 情境醫療可及性分數：{med_score_scenario:.2f}"
            )

        if pd.notna(row.get(shelter_id_col, np.nan)):
            shelter_row_s = shelter_lookup.loc[row[shelter_id_col]]
            shelter_text_scenario = (
                f"{scenario_name} 情境最近避難所 ID：{row[shelter_id_col]}<br>"
                f"經度：{shelter_row_s.geometry.x:.2f}<br>"
                f"緯度：{shelter_row_s.geometry.y:.2f}<br>"
                f"最快通行時間：{row[shelter_time_col]:.2f} 分鐘<br>"
                f"{scenario_name} 情境避難可及性分數：{shel_score_scenario:.2f}"
            )

        popup_text = (
            f"住宅 ID：{row.get('origin_id', '')}<br>"
            f"土地利用第一級分類：{row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
            f"土地利用第二級分類：{row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
            f"土地利用第三級分類：{row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}<br>"
            f"經度：{res_lon:.2f}<br>"
            f"緯度：{res_lat:.2f}<br><br>"
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

    m = add_flood_raster_to_folium(
        m=m,
        tif_path=flood_tif_path,
        layer_name=f"{scenario_name} 淹水潛勢圖",
        vmin=0,
        vmax=1,
        opacity=0.8,
        show=True,
        add_colorbar=False
    )
    m = add_flood_legend_to_folium(
        m,
        title=f"淹水潛勢（公尺）<br>情境：{scenario_name}"
    )

    folium.GeoJson(
        road_boundary_wgs84,
        name="研究區域",
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
        name="住宅用地 0502",
        style_function=lambda x: {
            "fillColor": "yellow",
            "color": "orange",
            "weight": 1,
            "fillOpacity": 0.45,
            "opacity": 0.9
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["LCODE_C2", "LCODE_C3"],
            aliases=["土地利用中類代碼", "土地利用小類代碼"]
        )
    ).add_to(m)

    folium.GeoJson(
        medical_wgs84,
        name="醫療保健用地 0603",
        style_function=lambda x: {
            "fillColor": "blue",
            "color": "blue",
            "weight": 1,
            "fillOpacity": 0.35,
            "opacity": 0.9
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["LCODE_C2", "LCODE_C3"],
            aliases=["土地利用中類代碼", "土地利用小類代碼"]
        )
    ).add_to(m)

    folium.GeoJson(
        edges_wgs84,
        name="路網",
        style_function=lambda x: {
            "color": "gray",
            "weight": 2,
            "opacity": 0.45
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[col for col in ["highway_type", "speed_kmh", "length", "travel_time_min"] if col in edges_wgs84.columns],
            aliases=["道路類型", "速限(km/h)", "長度(m)", "基準最快通行時間(min)"]
        )
    ).add_to(m)

    folium.GeoJson(
        edges_scenario_passable_wgs84,
        name=f"{scenario_name} 情境可通行路段",
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
                "道路類型", "道路長度(m)", "基準最快通行時間(min)",
                "淹水深度(m)", "通行時間倍數", f"{scenario_name}情境最快通行時間(min)"
            ]
        )
    ).add_to(m)

    folium.GeoJson(
        edges_scenario_blocked_wgs84,
        name=f"{scenario_name} 情境阻斷路段",
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
                "道路類型", "道路長度(m)", "基準最快通行時間(min)",
                "淹水深度(m)", "是否阻斷"
            ]
        )
    ).add_to(m)

    road_node_layer = folium.FeatureGroup(name="路網節點", show=False)
    medical_dest_layer = folium.FeatureGroup(name="醫療目的地 centroid", show=True)
    shelter_layer = folium.FeatureGroup(name="避難所", show=True)

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
            f"醫療目的地 ID: {row.get('medical_id', '')}<br>"
            f"土地利用大類: {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
            f"土地利用中類: {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
            f"土地利用小類: {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}<br>"
            f"經度: {row.geometry.x:.2f}<br>"
            f"緯度: {row.geometry.y:.2f}"
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
        "避難收容處所名稱",
        "避難收容處所地址",
        "村里",
        "縣市及鄉鎮市區",
        "預計收容人數",
        "適用災害類別",
        "管理人姓名",
        "管理人電話",
    ]

    for _, row in shelters_result_wgs84.iterrows():
        popup_lines = [f"避難所 ID: {row.get('shelter_id', '')}"]
        for field in shelter_popup_fields:
            if field in row.index:
                popup_lines.append(f"{field}: {row.get(field, '')}")
        popup_lines.append(f"經度: {row.geometry.x:.2f}")
        popup_lines.append(f"緯度: {row.geometry.y:.2f}")
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
    print(f"\n===== 開始處理情境：{scenario_name} =====")

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

    print(f"{scenario_name} 情境路網已完成淹水分析與通阻判定")
    print("路段總數：", len(edges_scenario_3826))
    print("阻斷路段數：", int(edges_scenario_3826[blocked_col].sum()))
    print("可通行路段數：", int((~edges_scenario_3826[blocked_col]).sum()))

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

    print(f"{scenario_name} 情境住宅可及性分數計算完成")
    print("無法到達醫療目的地的住宅數：", residential_scenario_3826[medical_time_col].isna().sum())
    print("無法到達避難所的住宅數：", residential_scenario_3826[shelter_time_col].isna().sum())

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

    html_path = output_dir / f"{scenario_name}_路網分析互動地圖.html"
    m.save(html_path)
    print("已輸出路網分析互動地圖：", html_path.resolve())

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

csv_path = output_dir / "住宅可及性綜整比較.csv"
comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("已輸出住宅可及性比較 CSV：", csv_path.resolve())

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
    raise KeyError("找不到可用的人口數量欄位，請確認人口數量分布圖層的欄位名稱。")

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
        "找不到暴露度指定的人口欄位。"
        "預期欄位包含「114年12月行政區人口統計 村里_屏東縣_人口數」。"
        f"目前圖層欄位為：{available_columns}"
    )

population_value_col = selected_population_col
population_gdf[population_value_col] = pd.to_numeric(
    population_gdf[population_value_col],
    errors="coerce"
)
population_gdf = population_gdf.dropna(subset=[population_value_col]).copy()
print(f"暴露度使用人口欄位：{population_value_col}")

if population_gdf.empty:
    raise ValueError("人口數量分布圖層沒有可用的數值資料。")

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
    if population_value is None or pd.isna(population_value):
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
    label = f"{lower:,} - {upper:,}"
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
    <div style="font-weight:700; margin-bottom:10px;">暴露度(總人口數)</div>
    {population_items_html}
</div>
"""

population_popup = folium.GeoJsonPopup(
    fields=["COUNTYNAME", "TOWNNAME", "VILLNAME", population_value_col],
    aliases=["縣市", "鄉鎮市區", "村里", "總人口數"],
    localize=True,
    labels=True,
    style="background-color: white;"
)

population_tooltip = folium.GeoJsonTooltip(
    fields=["TOWNNAME", "VILLNAME", population_value_col],
    aliases=["鄉鎮市區", "村里", "總人口數"],
    localize=True,
    sticky=False
)

if "comparison_df" not in locals():
    comparison_df = build_comparison_dataframe(residential_base_3826, scenario_csv_frames)

required_q100_cols = ["medical_access_score_Q100", "shelter_access_score_Q100"]
missing_q100_cols = [col for col in required_q100_cols if col not in comparison_df.columns]
if missing_q100_cols:
    raise KeyError(
        "comparison_df 缺少 Q100 情境欄位："
        + ", ".join(missing_q100_cols)
        + "。請先完整執行前面的 scenario 計算流程。"
    )

vulnerability_points_3826 = gpd.GeoDataFrame(
    comparison_df.copy(),
    geometry=gpd.points_from_xy(comparison_df["x_3826"], comparison_df["y_3826"]),
    crs=twd97_crs
)
vulnerability_points_wgs84 = vulnerability_points_3826.to_crs(wgs84)
vulnerability_points_wgs84["vulnerability_score_Q100"] = (
    vulnerability_points_wgs84["medical_access_score_Q100"].fillna(0)
    + vulnerability_points_wgs84["shelter_access_score_Q100"].fillna(0)
) / 2

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

valid_vulnerability = population_gdf["vulnerability_avg_Q100"].dropna()
if valid_vulnerability.empty:
    vulnerability_min = 0.0
    vulnerability_max = 1.0
else:
    vulnerability_min = float(valid_vulnerability.min())
    vulnerability_max = float(valid_vulnerability.max())

if vulnerability_min == vulnerability_max:
    vulnerability_max = vulnerability_min + 0.000001

vulnerability_bins = np.linspace(vulnerability_min, vulnerability_max, 6)
vulnerability_colors = [
    "#f7fcf0",
    "#ccebc5",
    "#7bccc4",
    "#2b8cbe",
    "#084081",
]

hazard_q100_tif_path = Path("../data/Geo_RA/Q100_depth_max.GangKoudem.2022dem.tif")
hazard_layer = add_flood_raster_to_folium(
    population_map_discrete,
    tif_path=hazard_q100_tif_path,
    layer_name="危害度(淹水潛勢圖-Q100)",
    opacity=0.80,
    show=True,
    return_layer=True
)

hazard_legend_html = """
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
    <div style="font-weight:700; margin-bottom:10px;">危害度(淹水潛勢圖-Q100)</div>
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

exposure_layer = folium.GeoJson(
    data=population_gdf,
    name="暴露度(總人口數)",
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
    "<span>無住宅點</span>"
    "</div>"
)

vulnerability_popup = folium.GeoJsonPopup(
    fields=["COUNTYNAME", "TOWNNAME", "VILLNAME", "vulnerability_avg_Q100", "residential_point_count"],
    aliases=["縣市", "鄉鎮市區", "村里", "Q100 平均脆弱度", "住宅點數量"],
    localize=True,
    labels=True,
    style="background-color: white;"
)

vulnerability_tooltip = folium.GeoJsonTooltip(
    fields=["TOWNNAME", "VILLNAME", "vulnerability_avg_Q100"],
    aliases=["鄉鎮市區", "村里", "Q100 平均脆弱度"],
    localize=True,
    sticky=False
)

vulnerability_layer = folium.GeoJson(
    data=population_gdf,
    name="脆弱度(醫療、避難可及性)",
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
    <div style="font-weight:700; margin-bottom:10px;">脆弱度(醫療、避難可及性)</div>
    {vulnerability_items_html}
</div>
"""

legend_control_html = f"""
{{% macro html(this, kwargs) %}}
{hazard_legend_html}
{vulnerability_legend_html}
{population_legend_html}
<script>
document.addEventListener("DOMContentLoaded", function() {{
    var map = {population_map_discrete.get_name()};
    var hazardLegend = document.getElementById("hazard-legend");
    var vulnerabilityLegend = document.getElementById("vulnerability-legend");
    var exposureLegend = document.getElementById("exposure-legend");
    var hazardLayer = {hazard_layer.get_name()};
    var exposureLayer = {exposure_layer.get_name()};
    var vulnerabilityLayer = {vulnerability_layer.get_name()};

    function updateLegendByLayer(layer, visible) {{
        if (layer === hazardLayer && hazardLegend) {{
            hazardLegend.style.display = visible ? "block" : "none";
        }}
        if (layer === vulnerabilityLayer && vulnerabilityLegend) {{
            vulnerabilityLegend.style.display = visible ? "block" : "none";
        }}
        if (layer === exposureLayer && exposureLegend) {{
            exposureLegend.style.display = visible ? "block" : "none";
        }}
    }}

    updateLegendByLayer(hazardLayer, map.hasLayer(hazardLayer));
    updateLegendByLayer(vulnerabilityLayer, map.hasLayer(vulnerabilityLayer));
    updateLegendByLayer(exposureLayer, map.hasLayer(exposureLayer));

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
print("暴露度(總人口數)互動式地圖已輸出:", population_html_path.resolve())
