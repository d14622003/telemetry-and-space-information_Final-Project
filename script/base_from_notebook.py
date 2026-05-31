import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import folium
import numpy as np
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt

output_dir = Path("../output")
output_dir.mkdir(exist_ok=True)

twd97_crs = "EPSG:3826" #公尺
wgs84 = "EPSG:4326" #經緯度

lu_gdf = gpd.read_file("../data/11_港口溪流域/01_土地利用/2007_LU.shp")

# 如果沒有 CRS，假設是 TWD97 / TM2
if lu_gdf.crs is None:
    lu_gdf = lu_gdf.set_crs(twd97_crs)

# 統一轉成 EPSG:3826，方便後續距離、面積、centroid 計算
lu_3826 = lu_gdf.to_crs(twd97_crs)

lu_3826["LCODE_C1"] = lu_3826["LCODE_C1"].astype(str).str.zfill(2)
lu_3826["LCODE_C2"] = lu_3826["LCODE_C2"].astype(str).str.zfill(4)
lu_3826["LCODE_C3"] = lu_3826["LCODE_C3"].astype(str).str.zfill(6)
print(lu_3826[["LCODE_C1", "LCODE_C2", "LCODE_C3"]].head(5))

#%% 1.1 加入土地利用分類中文名稱

c1_map = {
    "01": "農業使用土地",
    "02": "森林使用土地",
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
    "050202": "兼工業使用住宅",
    "050203": "兼商業使用住宅",
    "050204": "兼其他使用住宅",
    "060300": "醫療保健",
}

lu_3826["C1_NAME"] = lu_3826["LCODE_C1"].map(c1_map)
lu_3826["C2_NAME"] = lu_3826["LCODE_C2"].map(c2_map)
lu_3826["C3_NAME"] = lu_3826["LCODE_C3"].map(c3_map)

#%% 2. 選擇特定土地利用資料

# 起點：住宅用地
residential_3826 = lu_3826[lu_3826["LCODE_C2"] == "0502"].copy()

# 目的地 1：醫療保健用地
medical_3826 = lu_3826[lu_3826["LCODE_C2"] == "0603"].copy()

print("住宅用地數量:", len(residential_3826))
print("醫療保健用地數量:", len(medical_3826))

residential_3826.head(3)

#%% 2.1 載入避難所資料

shelters_csv_path = Path("../data/避難收容處所_清理後.csv")

if shelters_csv_path.exists():
    shelters_csv = pd.read_csv(shelters_csv_path, encoding="utf-8-sig")
    print("原始避難收容處所筆數:", len(shelters_csv))

    if "座標有效性" in shelters_csv.columns:
        valid_shelters = shelters_csv[
            shelters_csv["座標有效性"] == "有效"
        ].copy()
    else:
        print('Warning: "座標有效性" column not found, using all records')
        valid_shelters = shelters_csv.copy()

    print("有效避難收容處所筆數:", len(valid_shelters))

else:
    raise FileNotFoundError(f"找不到避難收容所資料：{shelters_csv_path}")
valid_shelters.head(1)

#%% 2.2 將避難所資料CSV轉換成圖資GeoDataFrame

shelters_wgs84 = gpd.GeoDataFrame(
    valid_shelters,
    geometry=gpd.points_from_xy(
        valid_shelters["經度"],
        valid_shelters["緯度"]
    ),
    crs=wgs84
)
# 將其轉成  EPSG:3826
shelters_3826 = shelters_wgs84.to_crs(twd97_crs)

print("避難所 CRS:", shelters_3826.crs)

#%% 2.3 Select shelters within study area

# 用土地利用資料建立研究區範圍，合併polygon
study_area_3826 = gpd.GeoDataFrame(
    geometry=[lu_3826.geometry.union_all()],
    crs=lu_3826.crs
)

# 篩選落在研究區範圍內的避難所
shelters_in_study_3826 = gpd.sjoin(
    shelters_3826,
    study_area_3826,
    how="inner",
    predicate="within"
).copy()

# 移除 spatial join 產生的 index_right 欄位
if "index_right" in shelters_in_study_3826.columns:
    shelters_in_study_3826 = shelters_in_study_3826.drop(columns="index_right")

print("研究區內避難所數量:", len(shelters_in_study_3826))


#%% 2.4 抓取研究區域向外1000公尺內的避難所

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

print("研究區外擴 1 km 內避難所數量:", len(shelters_in_buffer_3826))


#%% 3. 創建不同polygon中心點，並保留原polygon

# 起點：住宅用地 centroid
residential_origins_3826 = residential_3826.copy()
residential_origins_3826["geometry"] = residential_origins_3826.geometry.centroid
residential_origins_3826["origin_id"] = range(len(residential_origins_3826))

# 目的地 1：醫療保健用地 centroid
medical_dest_3826 = medical_3826.copy()
medical_dest_3826["geometry"] = medical_dest_3826.geometry.centroid
medical_dest_3826["medical_id"] = range(len(medical_dest_3826))


# 目的地 2：避難收容所
shelters_in_buffer_3826

print("住宅起點數量:", len(residential_origins_3826))
print("醫療目的地數量:", len(medical_dest_3826))
print("避難所目的地數量:", len(shelters_in_buffer_3826))

shelters_in_buffer_3826.head(1)

#%% 4. 抓取研究區域內路網

# 用研究區(流域)外擴 500 m 作為道路網抓取範圍
road_boundary_3826 = study_area_3826.copy()
road_boundary_3826["geometry"] = road_boundary_3826.geometry.buffer(500)

# OSMnx 抓路網需要 WGS84，抓取polygon
road_boundary_wgs84 = road_boundary_3826.to_crs(wgs84)
road_boundary_polygon = road_boundary_wgs84.geometry.iloc[0]

print("道路網抓取範圍 CRS:", road_boundary_wgs84.crs)

#%% 4.1 下載 OSM 路網資料

G_4326 = ox.graph_from_polygon(
    road_boundary_polygon,
    # network_type="drive",
    network_type="all",
    simplify=True,
    retain_all=True,
    # retain_all=False,
    truncate_by_edge=True
)

print("OSM 道路網下載完成")

#%% 4.2 Project road network to EPSG:3826

# 轉換坐標系統，轉成公尺
G_3826 = ox.project_graph(G_4326, to_crs=twd97_crs)

# 
nodes_3826, edges_3826 = ox.graph_to_gdfs(G_3826)

print("節點數量:", len(nodes_3826))
print("路段數量:", len(edges_3826))
print("道路網 CRS:", edges_3826.crs)

# 確認道路類型

# 如果 highway 是 list，就取第一個類型
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
    cmap_name="Reds",
    vmin=0,
    vmax=1,
    opacity=0.80,
    show=False
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

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(arr))
    rgba[np.isnan(arr), 3] = 0
    rgba_img = (rgba * 255).astype(np.uint8)

    folium.raster_layers.ImageOverlay(
        image=rgba_img,
        bounds=folium_bounds,
        name=layer_name,
        opacity=opacity,
        interactive=True,
        cross_origin=False,
        zindex=1,
        show=show
    ).add_to(m)

    return m


# 轉 WGS84
study_area_wgs84 = study_area_3826.to_crs(wgs84)
road_boundary_wgs84 = road_boundary_3826.to_crs(wgs84)

residential_wgs84 = residential_3826.to_crs(wgs84)
residential_result_wgs84 = residential_origins_3826.to_crs(wgs84)

medical_wgs84 = medical_3826.to_crs(wgs84)
medical_dest_wgs84 = medical_dest_3826.to_crs(wgs84)

shelters_result_wgs84 = shelters_in_buffer_3826.to_crs(wgs84)

nodes_wgs84 = nodes_3826.to_crs(wgs84)
edges_wgs84 = edges_3826.to_crs(wgs84)

medical_lookup = medical_dest_wgs84.set_index("medical_id")
shelter_lookup = shelters_result_wgs84.set_index("shelter_id")

center = study_area_wgs84.geometry.union_all().centroid

m = folium.Map(
    location=[center.y, center.x],
    zoom_start=12,
    tiles="OpenStreetMap"
)

# 淹水圖層
flood_paths = {
    "Q1.1": "../data/Geo_RA/Q1point1_depth_max.GangKoudem.2022dem.tif",
    "Q10":  "../data/Geo_RA/Q10_depth_max.GangKoudem.2022dem.tif",
    "Q25":  "../data/Geo_RA/Q25_depth_max.GangKoudem.2022dem.tif",
    "Q50":  "../data/Geo_RA/Q50_depth_max.GangKoudem.2022dem.tif",
    "Q100": "../data/Geo_RA/Q100_depth_max.GangKoudem.2022dem.tif",
}

for layer_name, tif_path in flood_paths.items():
    m = add_flood_raster_to_folium(
        m=m,
        tif_path=tif_path,
        layer_name=layer_name,
        cmap_name="Reds",
        vmin=0,
        vmax=1,
        opacity=0.8,
        show=False
    )

# 研究區
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

# 住宅 polygon
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

# 醫療 polygon
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

# 路網
folium.GeoJson(
    edges_wgs84,
    name="路網",
    style_function=lambda x: {
        "color": "black",
        "weight": 2.5,
        "opacity": 0.9
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[col for col in ["highway_type", "speed_kmh", "length", "travel_time_min"] if col in edges_wgs84.columns],
        aliases=["道路類型", "速度(km/h)", "長度(m)", "通行時間(min)"]
    )
).add_to(m)

# 圖層群組
road_node_layer = folium.FeatureGroup(name="路網節點", show=False)
residential_origin_layer = folium.FeatureGroup(name="住宅起點 centroid", show=True)
medical_dest_layer = folium.FeatureGroup(name="醫療目的地 centroid", show=True)
shelter_layer = folium.FeatureGroup(name="避難所", show=True)

# 路網節點
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

# 住宅點：加入最近醫療/避難所與時間資訊
for idx, row in residential_result_wgs84.iterrows():
    res_lon = row.geometry.x
    res_lat = row.geometry.y

    medical_text = "最近醫療：無法到達"
    shelter_text = "最近避難所：無法到達"

    if pd.notna(row["nearest_medical_id"]):
        medical_row = medical_lookup.loc[row["nearest_medical_id"]]
        medical_text = (
            f"最近醫療 ID: {row['nearest_medical_id']}<br>"
            f"醫療經度: {medical_row.geometry.x:.6f}<br>"
            f"醫療緯度: {medical_row.geometry.y:.6f}<br>"
            f"最快時間: {row['time_to_medical_min']:.2f} 分鐘"
        )

    if pd.notna(row["nearest_shelter_id"]):
        shelter_row = shelter_lookup.loc[row["nearest_shelter_id"]]
        shelter_text = (
            f"最近避難所 ID: {row['nearest_shelter_id']}<br>"
            f"避難所經度: {shelter_row.geometry.x:.6f}<br>"
            f"避難所緯度: {shelter_row.geometry.y:.6f}<br>"
            f"最快時間: {row['time_to_shelter_min']:.2f} 分鐘"
        )

    popup_text = (
        f"住宅起點 ID: {row.get('origin_id', '')}<br>"
        f"第一級分類: {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
        f"第二級分類: {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
        f"第三級分類: {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}<br>"
        f"住宅經度: {res_lon:.6f}<br>"
        f"住宅緯度: {res_lat:.6f}<br><br>"
        f"{medical_text}<br><br>"
        f"{shelter_text}"
    )

    folium.CircleMarker(
        location=[res_lat, res_lon],
        radius=4,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.85,
        popup=folium.Popup(popup_text, max_width=380)
    ).add_to(residential_origin_layer)

# 醫療點
for idx, row in medical_dest_wgs84.iterrows():
    popup_text = (
        f"醫療目的地 ID: {row.get('medical_id', '')}<br>"
        f"第一級分類: {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
        f"第二級分類: {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
        f"第三級分類: {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}<br>"
        f"經度: {row.geometry.x:.6f}<br>"
        f"緯度: {row.geometry.y:.6f}"
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

# 避難所
for idx, row in shelters_result_wgs84.iterrows():
    popup_text = (
        f"避難所 ID: {row.get('shelter_id', '')}<br>"
        f"避難所名稱: {row.get('避難收容處所名稱', '')}<br>"
        f"縣市鄉鎮: {row.get('縣市及鄉鎮市區', '')}<br>"
        f"村里: {row.get('村里', '')}<br>"
        f"地址: {row.get('避難收容處所地址', '')}<br>"
        f"預計收容村里: {row.get('預計收容村里', '')}<br>"
        f"預計收容人數: {row.get('預計收容人數', '')}<br>"
        f"適用災害類別: {row.get('適用災害類別', '')}<br>"
        f"管理人: {row.get('管理人姓名', '')}<br>"
        f"管理人電話: {row.get('管理人電話', '')}<br>"
        f"經度: {row.geometry.x:.6f}<br>"
        f"緯度: {row.geometry.y:.6f}"
    )

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
residential_origin_layer.add_to(m)
medical_dest_layer.add_to(m)
shelter_layer.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

m

#%% 15. 儲存互動式地圖為 HTML

html_path = output_dir / "住宅_醫療_避難所_最短時間地圖.html"
m.save(html_path)

print("互動式地圖已儲存至：", html_path)



# ---------- 選一個淹水情境 ----------
scenario_name = "Q10"
flood_tif_path = f"../data/Geo_RA/{scenario_name}_depth_max.GangKoudem.2022dem.tif"

# scenario_name = "Q25"
# scenario_name = "Q50"
# scenario_name = "Q100"



#%% 19. 儲存情境比較地圖

html_path = output_dir / f"{scenario_name}_可及性比較地圖.html"
m.save(html_path)

print("互動式地圖已儲存至：", html_path.resolve())



