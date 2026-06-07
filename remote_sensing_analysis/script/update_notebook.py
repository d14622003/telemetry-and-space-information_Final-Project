import json

notebook_path = r'd:\Downloads\gis_project\期末報告\remote_sensing_analysis\script\historical_flood_extraction.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "def extract_flood_footprint(roi, pre_start, pre_end, flood_start, flood_end, threshold=-1.25):\n",
    "    '''\n",
    "    從 Sentinel-1 影像中萃取淹水範圍\n",
    "    '''\n",
    "    # 載入 Sentinel-1 GRD 影像集合，並加入軌道方向過濾 (Orbit Pass Filter)\n",
    "    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \\\n",
    "        .filterBounds(roi) \\\n",
    "        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \\\n",
    "        .filter(ee.Filter.eq('instrumentMode', 'IW')) \\\n",
    "        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))\n",
    "    \n",
    "    # 篩選日期\n",
    "    pre_flood = collection.filterDate(pre_start, pre_end).mean().select('VH')\n",
    "    during_flood = collection.filterDate(flood_start, flood_end).mean().select('VH')\n",
    "    \n",
    "    # 使用 Focal Median 進行平滑處理，減少 SAR 的斑點雜訊 (Speckle noise)\n",
    "    smooth_radius = 50\n",
    "    pre_filtered = pre_flood.focal_median(smooth_radius, 'circle', 'meters')\n",
    "    during_filtered = during_flood.focal_median(smooth_radius, 'circle', 'meters')\n",
    "    \n",
    "    # 計算差異 (dB 值相減等同於原始值的 Log Ratio)\n",
    "    difference = during_filtered.subtract(pre_filtered)\n",
    "    \n",
    "    # 閾值切割 (Thresholding)\n",
    "    flood_mask = difference.lt(threshold)\n",
    "    \n",
    "    # 排除永久水體 (使用 JRC Global Surface Water 資料)\n",
    "    jrc = ee.Image(\"JRC/GSW1_4/GlobalSurfaceWater\").select('seasonality')\n",
    "    permanent_water = jrc.gte(10).unmask(0) # 存在 10 個月以上視為永久水體\n",
    "    flood_mask = flood_mask.updateMask(permanent_water.eq(0))\n",
    "    \n",
    "    # 加入地形坡度遮罩 (Slope Mask)，排除大於 5 度的山區地形\n",
    "    dem = ee.Image('USGS/SRTMGL1_003')\n",
    "    slope = ee.Terrain.slope(dem)\n",
    "    flat_areas = slope.lte(5)\n",
    "    flood_mask = flood_mask.updateMask(flat_areas)\n",
    "    \n",
    "    # 清理零碎像素\n",
    "    flood_mask = flood_mask.focal_mode(30, 'circle', 'meters').mask(flood_mask)\n",
    "    \n",
    "    return difference, flood_mask\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if len(cell['source']) > 0 and 'def extract_flood_footprint(' in cell['source'][0]:
            cell['source'] = new_source
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook updated successfully.")
