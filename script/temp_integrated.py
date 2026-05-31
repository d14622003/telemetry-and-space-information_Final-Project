#%% 霈??隞?
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import folium
import numpy as np
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt

#%% 閮剖?瑼?頝臬?
output_dir = Path("../output")
output_dir.mkdir(exist_ok=True)

twd97_crs = "EPSG:3826" #?砍偕
wgs84 = "EPSG:4326" #蝬楝摨?

#%% ?渡???拍鞈?

lu_gdf = gpd.read_file("../data/11_皜臬皞芣???01_??拍/2007_LU.shp")

# 憒?瘝? CRS嚗?閮剜 TWD97 / TM2
if lu_gdf.crs is None:
    lu_gdf = lu_gdf.set_crs(twd97_crs)

# 蝯曹?頧? EPSG:3826嚗靘踹?蝥??Ｕ蝛entroid 閮?
lu_3826 = lu_gdf.to_crs(twd97_crs)

lu_3826["LCODE_C1"] = lu_3826["LCODE_C1"].astype(str).str.zfill(2)
lu_3826["LCODE_C2"] = lu_3826["LCODE_C2"].astype(str).str.zfill(4)
lu_3826["LCODE_C3"] = lu_3826["LCODE_C3"].astype(str).str.zfill(6)
print(lu_3826[["LCODE_C1", "LCODE_C2", "LCODE_C3"]].head(5))

#%% 1.1 ???拍??銝剜??迂
c1_map = {
    "01": "颲脫平雿輻?",
    "02": "璉格?雿輻?",
    "03": "鈭日蝙?典???,
    "04": "瘞游雿輻?",
    "05": "撱箇?雿輻?",
    "06": "?砍雿輻?",
    "07": "?雿輻?",
    "08": "蝷阡厭雿輻?",
    "09": "?嗡?雿輻?",
}
c2_map = {
    "0502": "雿?",
    "0603": "?怎?靽",
}
c3_map = {
    "050201": "蝝?摰?,
    "050202": "?澆極璆凋蝙?其?摰?,
    "050203": "?澆?璆凋蝙?其?摰?,
    "050204": "?澆隞蝙?其?摰?,
    "060300": "?怎?靽",
}
lu_3826["C1_NAME"] = lu_3826["LCODE_C1"].map(c1_map)
lu_3826["C2_NAME"] = lu_3826["LCODE_C2"].map(c2_map)
lu_3826["C3_NAME"] = lu_3826["LCODE_C3"].map(c3_map)

#%% 2. ?豢??孵???拍鞈?
# 韏琿?嚗?摰??
residential_3826 = lu_3826[lu_3826["LCODE_C2"] == "0502"].copy()
# ?桃???1嚗???亦??
medical_3826 = lu_3826[lu_3826["LCODE_C2"] == "0603"].copy()
print("雿??典?賊?:", len(residential_3826))
print("?怎?靽?典?賊?:", len(medical_3826))

#%% ?渡??輸?鞈?-頛?輸?鞈?
shelters_csv_path = Path("../data/?輸?嗅捆??_皜?敺?csv")

if shelters_csv_path.exists():
    shelters_csv = pd.read_csv(shelters_csv_path, encoding="utf-8-sig")
    print("???輸?嗅捆??蝑:", len(shelters_csv))

    if "摨扳????? in shelters_csv.columns:
        valid_shelters = shelters_csv[
            shelters_csv["摨扳?????] == "??"
        ].copy()
    else:
        print('Warning: "摨扳????? column not found, using all records')
        valid_shelters = shelters_csv.copy()

    print("???輸?嗅捆??蝑:", len(valid_shelters))

else:
    raise FileNotFoundError(f"?曆??圈??摰寞?鞈?嚗shelters_csv_path}")
valid_shelters.head(1)
#%% 2.2 撠???鞈?CSV頧???鞈eoDataFrame
shelters_wgs84 = gpd.GeoDataFrame(
    valid_shelters,
    geometry=gpd.points_from_xy(
        valid_shelters["蝬漲"],
        valid_shelters["蝺臬漲"]
    ),
    crs=wgs84
)
# 撠頧?  EPSG:3826
shelters_3826 = shelters_wgs84.to_crs(twd97_crs)

print("?輸? CRS:", shelters_3826.crs)
#%% 2.3 Select shelters within study area

# ?典??啣?刻??遣蝡?蝛嗅?蝭?嚗?雿皖olygon
study_area_3826 = gpd.GeoDataFrame(
    geometry=[lu_3826.geometry.union_all()],
    crs=lu_3826.crs
)

# 蝭拚?賢?弦?蝭??抒??輸?
shelters_in_study_3826 = gpd.sjoin(
    shelters_3826,
    study_area_3826,
    how="inner",
    predicate="within"
).copy()

# 蝘駁 spatial join ?Ｙ???index_right 甈?
if "index_right" in shelters_in_study_3826.columns:
    shelters_in_study_3826 = shelters_in_study_3826.drop(columns="index_right")

print("?弦??折????賊?:", len(shelters_in_study_3826))

#%% 2.4 ???弦???憭?000?砍偕?抒??輸?

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

print("?弦?憭 1 km ?折????賊?:", len(shelters_in_buffer_3826))

#%% 3. ?萄遣銝?polygon銝剖?暺?銝虫???polygon

# 韏琿?嚗?摰??centroid
residential_origins_3826 = residential_3826.copy()
residential_origins_3826["geometry"] = residential_origins_3826.geometry.centroid
residential_origins_3826["origin_id"] = range(len(residential_origins_3826))

# ?桃???1嚗???亦??centroid
medical_dest_3826 = medical_3826.copy()
medical_dest_3826["geometry"] = medical_dest_3826.geometry.centroid
medical_dest_3826["medical_id"] = range(len(medical_dest_3826))

# ?桃???2嚗??摰寞?
shelters_in_buffer_3826

print("雿?韏琿??賊?:", len(residential_origins_3826))
print("?怎??桃??唳??", len(medical_dest_3826))
print("?輸??桃??唳??", len(shelters_in_buffer_3826))

#%% 4.頝舐雯撱箇蔭_???弦??頝舐雯

# ?函?蝛嗅?(瘚?)憭 500 m 雿?楝蝬脫?????
road_boundary_3826 = study_area_3826.copy()
road_boundary_3826["geometry"] = road_boundary_3826.geometry.buffer(500)

# OSMnx ?楝蝬脤?閬?WGS84嚗??olygon
road_boundary_wgs84 = road_boundary_3826.to_crs(wgs84)
road_boundary_polygon = road_boundary_wgs84.geometry.iloc[0]

print("?楝蝬脫?????CRS:", road_boundary_wgs84.crs)

#%% 4.1 銝? OSM 頝舐雯鞈?
G_4326 = ox.graph_from_polygon(
    road_boundary_polygon,
    # network_type="drive",
    network_type="all",
    simplify=True,
    retain_all=True,
    # retain_all=False,
    truncate_by_edge=True
)
print("OSM ?楝蝬脖?頛???)

#%% 4.2 Project road network to EPSG:3826

# 頧???蝟餌絞嚗??撠?
G_3826 = ox.project_graph(G_4326, to_crs=twd97_crs)

# 
nodes_3826, edges_3826 = ox.graph_to_gdfs(G_3826)

print("蝭暺??", len(nodes_3826))
print("頝舀挾?賊?:", len(edges_3826))
print("?楝蝬?CRS:", edges_3826.crs)
# In[36]: # 蝣箄??楝憿?

# 憒? highway ??list嚗停?洵銝????
def clean_highway(x):
    if isinstance(x, list):
        return x[0]
    else:
        return x

edges_3826["highway_simple"] = edges_3826["highway"].apply(clean_highway)

print(edges_3826["highway_simple"].unique())

#%% 5. 蝯血??身?漲

default_speed = {
    "primary": 50,
    "primary_link": 40,
    "secondary": 50,
    "secondary_link": 40,
    "tertiary": 40,
    "residential": 30,
    "unclassified": 30,

    # 撠??楝??仿?頝?
    "service": 20,
    "track": 15,

    # 甇亥?憿?頝?
    "path": 5,
    "footway": 5,
    "steps": 2,
}
# G_3826?桐??箏撠?
# u      ?挾?楝?絲暺?暺?
# v      ?挾?楝??暺?暺?
# k      edge key嚗靘???銝蝯?暺???憭?頝?
# data   ?挾?楝?惇?扯???
for u, v, k, data in G_3826.edges(keys=True, data=True):
    highway_type = clean_highway(data.get("highway", None))

    # 銝Ⅱ摰?閬?閮剖?撠?
    speed_kmh = default_speed.get(highway_type, 10)
    length_m = data.get("length", 0)
    travel_time_min = length_m / 1000 / speed_kmh * 60

    data["highway_type"] = highway_type
    data["speed_kmh"] = speed_kmh
    data["travel_time_min"] = travel_time_min
    
#%% 5.1 ????啁?頝舐雯??暺????銵???蝑?

nodes_3826, edges_3826 = ox.graph_to_gdfs(G_3826)

#%% 6. 蝜芾ˊ鈭?撘??瑼Ｘ?桀?撱箇???撅?

# ------------------------------------------------------------
# 1. 頧? WGS84嚗策 Folium 雿輻
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
# 2. 撱箇?摨?
# ------------------------------------------------------------
center = study_area_wgs84.geometry.union_all().centroid

m = folium.Map(
    location=[center.y, center.x],
    zoom_start=12,
    tiles="OpenStreetMap"
)
# ------------------------------------------------------------
# 3. 瘚?蝭???頝舐雯??蝭?
# ------------------------------------------------------------
folium.GeoJson(
    road_boundary_wgs84,
    name="?弦???,
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
# 4. 雿???? polygon
# ------------------------------------------------------------
folium.GeoJson(
    residential_wgs84,
    name="雿??典 0502",
    style_function=lambda x: {
        "fillColor": "yellow",
        "color": "orange",
        "weight": 1,
        "fillOpacity": 0.45,
        "opacity": 0.9
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["LCODE_C2", "LCODE_C3"],
        aliases=["蝚砌?蝝誨蝣?, "蝚砌?蝝誨蝣?]
    )
).add_to(m)

folium.GeoJson(
    medical_wgs84,
    name="?怎?靽?典 0603",
    style_function=lambda x: {
        "fillColor": "blue",
        "color": "blue",
        "weight": 1,
        "fillOpacity": 0.35,
        "opacity": 0.9
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["LCODE_C2", "LCODE_C3"],
        aliases=["蝚砌?蝝誨蝣?, "蝚砌?蝝誨蝣?]
    )
).add_to(m)
# ------------------------------------------------------------
# 5. 頝舐雯?惜
# ------------------------------------------------------------

folium.GeoJson(
    edges_wgs84,
    name="頝舐雯",
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
            "?楝憿?", "?漲(km/h)", "?瑕漲(m)", "????(min)"
        ]
    )
).add_to(m)
# ------------------------------------------------------------
# 6. 撱箇??舫???暺??惜
# ------------------------------------------------------------

road_node_layer = folium.FeatureGroup(
    name="頝舐雯蝭暺?,
    show=False
)

residential_origin_layer = folium.FeatureGroup(
    name="雿?韏琿? centroid",
    show=True
)

medical_dest_layer = folium.FeatureGroup(
    name="?怎??桃???centroid",
    show=True
)

shelter_layer = folium.FeatureGroup(
    name="?輸?",
    show=True
)
# ------------------------------------------------------------
# 7. ?頝舐雯蝭暺?
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
# 8. ?雿?韏琿? centroid
# ------------------------------------------------------------

# 撠?銝??摰??centroid ?雿?韏琿??惜
for idx, row in residential_origins_wgs84.iterrows():

    popup_text = (
        f"雿?韏琿? ID: {row.get('origin_id', '')}<br>"
        f"蝚砌?蝝?憿? {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
        f"蝚砌?蝝?憿? {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
        f"蝚砌?蝝?憿? {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}"
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
# 9. ??怎??桃???centroid
# ------------------------------------------------------------

# 撠?銝????亦??centroid ??怎??桃??啣?撅?
for idx, row in medical_dest_wgs84.iterrows():

    popup_text = (
        f"?怎??桃???ID: {row.get('medical_id', '')}<br>"
        f"蝚砌?蝝?憿? {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
        f"蝚砌?蝝?憿? {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
        f"蝚砌?蝝?憿? {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}"
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
# 10. ??輸?
# ------------------------------------------------------------

# 撠?銝??????輸??惜
for idx, row in shelters_wgs84.iterrows():

    popup_text = (
        f"?輸??迂: {row.get('?輸?嗅捆???迂', '')}<br>"
        f"蝮???: {row.get('蝮?????桀??', '')}<br>"
        f"??: {row.get('??', '')}<br>"
        f"?啣?: {row.get('?輸?嗅捆???啣?', '')}<br>"
        f"???嗅捆??: {row.get('???嗅捆??', '')}<br>"
        f"???嗅捆鈭箸: {row.get('???嗅捆鈭箸', '')}<br>"
        f"?拍?賢拿憿: {row.get('?拍?賢拿憿', '')}<br>"
        f"蝞∠?鈭? {row.get('蝞∠?鈭箏???, '')}<br>"
        f"蝞∠?鈭粹閰? {row.get('蝞∠?鈭粹閰?, '')}<br>"
        f"蝬漲: {row.get('蝬漲', '')}<br>"
        f"蝺臬漲: {row.get('蝺臬漲', '')}"
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
# 11. 撠?雿?撅文??亙??
# ------------------------------------------------------------
road_node_layer.add_to(m)
residential_origin_layer.add_to(m)
medical_dest_layer.add_to(m)
shelter_layer.add_to(m)
# ------------------------------------------------------------
# 12. ?惜?批?＊蝷?
# ------------------------------------------------------------
folium.LayerControl(collapsed=False).add_to(m)
m

#%% 13. ?脣?鈭?撘? HTML

html_path = output_dir / "頝舐雯??蝯?.html"

m.save(html_path)

print("鈭?撘?歇?脣??喉?", html_path)
#%%
# =============================================================================
# # # ??蝺?
# =============================================================================

# In[41]:
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
# In[45]:
tif_path ="../data/Geo_RA/Q1point1_depth_max.GangKoudem.2022dem.tif"

with rasterio.open(tif_path) as src:
    arr = src.read(1).astype(float)
    nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    # 憒? 0 隞?”瘝?瘛寞偌嚗隞亙憭? > 0 ????
    arr_positive = arr[arr > 0]

    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Nodata:", nodata)
    print("?券????min:", np.nanmin(arr))
    print("?券????max:", np.nanmax(arr))
    print("?券????mean:", np.nanmean(arr))
    print("?券????median:", np.nanmedian(arr))

    print("憭扳 0 ???:", arr_positive.size)

    if arr_positive.size > 0:
        print("瘛寞偌瘛勗漲 > 0 min:", np.nanmin(arr_positive))
        print("瘛寞偌瘛勗漲 > 0 max:", np.nanmax(arr_positive))
        print("瘛寞偌瘛勗漲 > 0 mean:", np.nanmean(arr_positive))
        print("瘛寞偌瘛勗漲 > 0 median:", np.nanmedian(arr_positive))
        print("95?曉?雿:", np.nanpercentile(arr_positive, 95))
        print("99?曉?雿:", np.nanpercentile(arr_positive, 99))
# In[46]:
# =============================================================================
# # # ??蝺?
# =============================================================================
#%% 8. ?豢????刻楝蝬?

G_analysis = G_3826
nodes_analysis_3826 = nodes_3826
edges_analysis_3826 = edges_3826

#%% 9. 撠?摰???? snap ?啣??頝舐雯嚗???撠??啣?楝蝬脩?暺?

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

print("雿??????撌脤???snap ?啣??頝舐雯")

#%% 10. ?湔?曹?摰?敺?桃??啗?蝞?敹怨楝敺?

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
            # 閮?敺?摰?暺?潘??唳????暺???剝???
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

            # ?曉??????剔??桃??啁?暺?
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

#%% 11. 閮?雿??唳?餈??

residential_origins_3826 = find_nearest_destination_direct(
    G=G_analysis,
    origins_gdf=residential_origins_3826,
    destinations_gdf=medical_dest_3826,
    origin_node_col="nearest_node",
    dest_node_col="nearest_node",
    dest_id_col="medical_id",
    prefix="medical"
)

#%% 12. 閮?雿??唳?餈???

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

#%% 13. 瑼Ｘ?臬??抒???

print("?⊥??圈??怎??湔???摰??賊?:")
print(residential_origins_3826["time_to_medical_min"].isna().sum())

print("?⊥??圈??輸???摰??賊?:")
print(residential_origins_3826["time_to_shelter_min"].isna().sum())
#%%
road_boundary_out = output_dir / "road_boundary_wgs84.geojson"
road_boundary_wgs84.to_file(road_boundary_out, driver="GeoJSON")

print("road_boundary_wgs84 撌脣摮嚗?, road_boundary_out.resolve())
#%% 14. 憭?憓????瘛寞偌 + 頝舐雯 + ?臬???+ 憭?憓撓??

import folium
import rasterio
import numpy as np
from rasterio.warp import transform_bounds
from matplotlib import colors
from shapely.geometry import LineString, MultiLineString
from branca.element import MacroElement, Template


# ?憭批?亙???嚗?芾?隤踵
max_accept_time_medical = 30.0
max_accept_time_shelter = 30.0

scenario_names = ["Q1point1", "Q10", "Q25", "Q50", "Q100"]


def add_flood_raster_to_folium(
    m,
    tif_path,
    layer_name="瘛寞偌瘛勗漲",
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

    # ??get_flood_time_factor ?詨???
    flood_bins = [0.0, 0.05, 0.10, 0.25, 0.50, effective_vmax]
    flood_colors = [
        "#fff5f0",  # 0 - 0.05
        "#fcbba1",  # 0.05 - 0.10
        "#fc9272",  # 0.10 - 0.25
        "#ef3b2c",  # 0.25 - 0.50
        "#99000d",  # > 0.50
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


def add_flood_legend_to_folium(m, title="瘛寞偌瞏嚗溯瘞湔楛摨佗??砍偕嚗?):
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
    residential_origin_layer = folium.FeatureGroup(name="雿?韏琿? centroid", show=True)

    for idx, row in residential_result_wgs84.iterrows():
        res_lon = row.geometry.x
        res_lat = row.geometry.y

        medical_text_normal = "銝?祆?憓?餈???⊥??圈?"
        shelter_text_normal = "銝?祆?憓?餈???嚗瘜??
        medical_text_scenario = f"{scenario_name} ???餈???⊥??圈?"
        shelter_text_scenario = f"{scenario_name} ???餈???嚗瘜??

        med_score_normal = row.get("medical_access_score_normal", 0.0)
        shel_score_normal = row.get("shelter_access_score_normal", 0.0)
        med_score_scenario = row.get(medical_score_scenario_col, 0.0)
        shel_score_scenario = row.get(shelter_score_scenario_col, 0.0)

        if pd.notna(row.get("nearest_medical_id", np.nan)):
            medical_row = medical_lookup.loc[row["nearest_medical_id"]]
            medical_text_normal = (
                f"銝?祆?憓?餈??ID: {row['nearest_medical_id']}<br>"
                f"?怎?蝬漲: {medical_row.geometry.x:.6f}<br>"
                f"?怎?蝺臬漲: {medical_row.geometry.y:.6f}<br>"
                f"?敹急??? {row['time_to_medical_min']:.2f} ??<br>"
                f"?怎??臬??批??? {med_score_normal:.3f}"
            )
        else:
            medical_text_normal = (
                "銝?祆?憓?餈???⊥??圈?<br>"
                f"?怎??臬??批??? {med_score_normal:.3f}"
            )

        if pd.notna(row.get("nearest_shelter_id", np.nan)):
            shelter_row = shelter_lookup.loc[row["nearest_shelter_id"]]
            shelter_text_normal = (
                f"銝?祆?憓?餈??? ID: {row['nearest_shelter_id']}<br>"
                f"?輸?蝬漲: {shelter_row.geometry.x:.6f}<br>"
                f"?輸?蝺臬漲: {shelter_row.geometry.y:.6f}<br>"
                f"?敹急??? {row['time_to_shelter_min']:.2f} ??<br>"
                f"?輸?臬??批??? {shel_score_normal:.3f}"
            )
        else:
            shelter_text_normal = (
                "銝?祆?憓?餈???嚗瘜??br>"
                f"?輸?臬??批??? {shel_score_normal:.3f}"
            )

        if pd.notna(row.get(medical_id_col, np.nan)):
            medical_row_s = medical_lookup.loc[row[medical_id_col]]
            medical_text_scenario = (
                f"{scenario_name} ???餈??ID: {row[medical_id_col]}<br>"
                f"?怎?蝬漲: {medical_row_s.geometry.x:.6f}<br>"
                f"?怎?蝺臬漲: {medical_row_s.geometry.y:.6f}<br>"
                f"?敹急??? {row[medical_time_col]:.2f} ??<br>"
                f"?怎??臬??批??? {med_score_scenario:.3f}"
            )
        else:
            medical_text_scenario = (
                f"{scenario_name} ???餈???⊥??圈?<br>"
                f"?怎??臬??批??? {med_score_scenario:.3f}"
            )

        if pd.notna(row.get(shelter_id_col, np.nan)):
            shelter_row_s = shelter_lookup.loc[row[shelter_id_col]]
            shelter_text_scenario = (
                f"{scenario_name} ???餈??? ID: {row[shelter_id_col]}<br>"
                f"?輸?蝬漲: {shelter_row_s.geometry.x:.6f}<br>"
                f"?輸?蝺臬漲: {shelter_row_s.geometry.y:.6f}<br>"
                f"?敹急??? {row[shelter_time_col]:.2f} ??<br>"
                f"?輸?臬??批??? {shel_score_scenario:.3f}"
            )
        else:
            shelter_text_scenario = (
                f"{scenario_name} ???餈???嚗瘜??br>"
                f"?輸?臬??批??? {shel_score_scenario:.3f}"
            )

        popup_text = (
            f"雿?韏琿? ID: {row.get('origin_id', '')}<br>"
            f"蝚砌?蝝?憿? {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
            f"蝚砌?蝝?憿? {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
            f"蝚砌?蝝?憿? {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}<br>"
            f"雿?蝬漲: {res_lon:.6f}<br>"
            f"雿?蝺臬漲: {res_lat:.6f}<br><br>"
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
        layer_name=f"{scenario_name} 瘛寞偌瞏",
        vmin=0,
        vmax=1,
        opacity=0.8,
        show=True,
        add_colorbar=False
    )
    m = add_flood_legend_to_folium(
        m,
        title=f"{scenario_name} 瘛寞偌瞏嚗撠綽?"
    )

    folium.GeoJson(
        road_boundary_wgs84,
        name="?弦???,
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
        name="雿??典 0502",
        style_function=lambda x: {
            "fillColor": "yellow",
            "color": "orange",
            "weight": 1,
            "fillOpacity": 0.45,
            "opacity": 0.9
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["LCODE_C2", "LCODE_C3"],
            aliases=["蝚砌?蝝誨蝣?, "蝚砌?蝝誨蝣?]
        )
    ).add_to(m)

    folium.GeoJson(
        medical_wgs84,
        name="?怎?靽?典 0603",
        style_function=lambda x: {
            "fillColor": "blue",
            "color": "blue",
            "weight": 1,
            "fillOpacity": 0.35,
            "opacity": 0.9
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["LCODE_C2", "LCODE_C3"],
            aliases=["蝚砌?蝝誨蝣?, "蝚砌?蝝誨蝣?]
        )
    ).add_to(m)

    folium.GeoJson(
        edges_wgs84,
        name="??頝舐雯",
        style_function=lambda x: {
            "color": "gray",
            "weight": 2,
            "opacity": 0.45
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[col for col in ["highway_type", "speed_kmh", "length", "travel_time_min"] if col in edges_wgs84.columns],
            aliases=["?楝憿?", "?漲(km/h)", "?瑕漲(m)", "????(min)"]
        )
    ).add_to(m)

    folium.GeoJson(
        edges_scenario_passable_wgs84,
        name=f"{scenario_name} ?舫??楝",
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
                "?楝憿?", "?楝?瑕漲(m)", "??????(min)",
                "?憭扳溯瘞湔楛摨?m)", "????", f"{scenario_name}??????(min)"
            ]
        )
    ).add_to(m)

    folium.GeoJson(
        edges_scenario_blocked_wgs84,
        name=f"{scenario_name} 銝???楝",
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
                "?楝憿?", "?楝?瑕漲(m)", "??????(min)",
                "?憭扳溯瘞湔楛摨?m)", "?臬?餅"
            ]
        )
    ).add_to(m)

    road_node_layer = folium.FeatureGroup(name="頝舐雯蝭暺?, show=False)
    medical_dest_layer = folium.FeatureGroup(name="?怎??桃???centroid", show=True)
    shelter_layer = folium.FeatureGroup(name="?輸?", show=True)

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

    for idx, row in medical_dest_wgs84.iterrows():
        popup_text = (
            f"?怎??桃???ID: {row.get('medical_id', '')}<br>"
            f"蝚砌?蝝?憿? {row.get('LCODE_C1', '')} {row.get('C1_NAME', '')}<br>"
            f"蝚砌?蝝?憿? {row.get('LCODE_C2', '')} {row.get('C2_NAME', '')}<br>"
            f"蝚砌?蝝?憿? {row.get('LCODE_C3', '')} {row.get('C3_NAME', '')}<br>"
            f"蝬漲: {row.geometry.x:.6f}<br>"
            f"蝺臬漲: {row.geometry.y:.6f}"
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

    for idx, row in shelters_result_wgs84.iterrows():
        popup_text = (
            f"?輸? ID: {row.get('shelter_id', '')}<br>"
            f"?輸??迂: {row.get('?輸?嗅捆???迂', '')}<br>"
            f"蝮???: {row.get('蝮?????桀??', '')}<br>"
            f"??: {row.get('??', '')}<br>"
            f"?啣?: {row.get('?輸?嗅捆???啣?', '')}<br>"
            f"???嗅捆??: {row.get('???嗅捆??', '')}<br>"
            f"???嗅捆鈭箸: {row.get('???嗅捆鈭箸', '')}<br>"
            f"?拍?賢拿憿: {row.get('?拍?賢拿憿', '')}<br>"
            f"蝞∠?鈭? {row.get('蝞∠?鈭箏???, '')}<br>"
            f"蝞∠?鈭粹閰? {row.get('蝞∠?鈭粹閰?, '')}<br>"
            f"蝬漲: {row.geometry.x:.6f}<br>"
            f"蝺臬漲: {row.geometry.y:.6f}"
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
    medical_dest_layer.add_to(m)
    shelter_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ?遣蝡??祆?憓?批??賂?敺???憓??
residential_base_3826 = residential_origins_3826.copy()
residential_base_3826["medical_access_score_normal"] = residential_base_3826["time_to_medical_min"].apply(
    lambda x: calc_accessibility_score(x, max_accept_time_medical)
)
residential_base_3826["shelter_access_score_normal"] = residential_base_3826["time_to_shelter_min"].apply(
    lambda x: calc_accessibility_score(x, max_accept_time_shelter)
)

scenario_csv_frames = []

for scenario_name in scenario_names:
    flood_tif_path = f"../data/Geo_RA/{scenario_name}_depth_max.GangKoudem.2022dem.tif"
    print(f"\n===== ??????嚗scenario_name} =====")

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

    print(f"{scenario_name} ???楝瘛寞偌撅祆扯?蝞???)
    print("?楝蝮賣嚗?, len(edges_scenario_3826))
    print("銝???楝?賂?", int(edges_scenario_3826[blocked_col].sum()))
    print("?舫??楝?賂?", int((~edges_scenario_3826[blocked_col]).sum()))

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

    print(f"{scenario_name} ??雿??臬??扯?蝞???)
    print("?⊥??圈??怎??湔???摰??賊?嚗?, residential_scenario_3826[medical_time_col].isna().sum())
    print("?⊥??圈??輸???摰??賊?嚗?, residential_scenario_3826[shelter_time_col].isna().sum())

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

    html_path = output_dir / f"{scenario_name}_?游??臬??扯??楝瘛寞偌?啣?.html"
    m.save(html_path)
    print("鈭?撘?歇?脣??喉?", html_path.resolve())

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


comparison_df = residential_base_3826[[
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

comparison_df["x_3826"] = comparison_df.geometry.x
comparison_df["y_3826"] = comparison_df.geometry.y
comparison_df = comparison_df.drop(columns="geometry")

for scenario_df in scenario_csv_frames:
    comparison_df = comparison_df.merge(scenario_df, on="origin_id", how="left")

csv_path = output_dir / "雿?暺憭?憓?扳?頛?csv"
comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("憭?憓?頛?CSV 撌脣摮嚗?, csv_path.resolve())
#%% 霈?犖????撣?撅?
population_gpkg_path = Path("../data/鈭箏?賊???.gpkg")
population_gdf = gpd.read_file(population_gpkg_path)

if population_gdf.crs is None:
    population_gdf = population_gdf.set_crs(wgs84)
else:
    population_gdf = population_gdf.to_crs(wgs84)

population_col_candidates = [
    col for col in population_gdf.columns
    if "鈭箏?? in col and "?瑟? not in col and "憟單? not in col
]

if not population_col_candidates:
    raise KeyError("?曆??唬犖??甈?嚗?蝣箄?鈭箏?惜甈??迂??)

population_value_col = population_col_candidates[0]
population_gdf[population_value_col] = pd.to_numeric(
    population_gdf[population_value_col],
    errors="coerce"
)
population_gdf = population_gdf.dropna(subset=[population_value_col]).copy()

if population_gdf.empty:
    raise ValueError("鈭箏?惜瘝??舐?犖??鞈???)

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




#%% ?湧摨?蝮賭犖??)鈭?撘??- ?Ｘ?脣??
population_bins = np.linspace(population_min, population_max, 6)
population_bins = np.round(population_bins).astype(int)
population_bins[0] = int(np.floor(population_min))
population_bins[-1] = int(np.ceil(population_max))

for i in range(1, len(population_bins)):
    if population_bins[i] <= population_bins[i - 1]:
        population_bins[i] = population_bins[i - 1] + 1

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
    lower = population_bins[idx]
    upper = population_bins[idx + 1]
    if idx == len(population_colors) - 1:
        label = f"{lower:,} - {upper:,}"
    else:
        label = f"{lower:,} - {upper - 1:,}"
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
    bottom: 40px;
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
    <div style="font-weight:700; margin-bottom:10px;">\u66b4\u9732\u5ea6(\u7e3d\u4eba\u53e3\u6578)</div>
    {population_items_html}
</div>
"""

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
    layer_name="\u5371\u5bb3\u5ea6(\u6df9\u6c34\u6f5b\u52e2\u5716-Q100)",
    opacity=0.80,
    show=True,
    return_layer=True
)

hazard_legend_html = """
<div id="hazard-legend" style="
    display:none;
    position: fixed;
    bottom: 460px;
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
    <div style="font-weight:700; margin-bottom:10px;">\u5371\u5bb3\u5ea6(\u6df9\u6c34\u6f5b\u52e2\u5716-Q100)</div>
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
    name="\u66b4\u9732\u5ea6(\u7e3d\u4eba\u53e3\u6578)",
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
    if idx == len(vulnerability_colors) - 1:
        label = f"{lower:.2f} - {upper:.2f}"
    else:
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
    aliases=["蝮??", "?撣?", "??", "Q100 撟喳??摹摨?, "雿?暺"],
    localize=True,
    labels=True,
    style="background-color: white;"
)

vulnerability_tooltip = folium.GeoJsonTooltip(
    fields=["TOWNNAME", "VILLNAME", "vulnerability_avg_Q100"],
    aliases=["?撣?", "??", "Q100 撟喳??摹摨?],
    localize=True,
    sticky=False
)

vulnerability_layer = folium.GeoJson(
    data=population_gdf,
    name="\u8106\u5f31\u5ea6(\u91ab\u7642\u3001\u907f\u96e3\u53ef\u53ca\u6027)",
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
    bottom: 250px;
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
    <div style="font-weight:700; margin-bottom:10px;">\u8106\u5f31\u5ea6(\u91ab\u7642\u3001\u907f\u96e3\u53ef\u53ca\u6027)</div>
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
print("\u66b4\u9732\u5ea6(\u7e3d\u4eba\u53e3\u6578)\u4e92\u52d5\u5f0f\u5730\u5716\u5df2\u8f38\u51fa:", population_html_path.resolve())
