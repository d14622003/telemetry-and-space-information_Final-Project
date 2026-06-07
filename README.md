# 港口溪流域路網可及性與淹水風險分析

本專案以兩支主程式為核心：

- `script/路網分析(整合).py`
- `script/路網分析(整合)_英文版.py`

兩者的分析流程相同，只差在輸出文字與圖層說明語言不同。程式會結合土地利用資料、避難收容處所、淹水深度 raster、人口分布，以及從 OpenStreetMap 下載的道路網，進行住宅點到醫療設施與避難所的路網可及性分析，並進一步整合 exposure、hazard、vulnerability 與 risk 的互動地圖成果。

## 專案結構

```text
期末報告/
├─ data/
│  ├─ 11_港口溪流域/
│  │  └─ 01_土地利用/
│  │     ├─ 2007_LU.shp
│  │     ├─ 2007_LU.shx
│  │     ├─ 2007_LU.dbf
│  │     ├─ 2007_LU.sbn
│  │     ├─ 2007_LU.sbx
│  │     └─ 2007_LU.shp.xml
│  ├─ Geo_RA/
│  │  ├─ Q1point1_depth_max.GangKoudem.2022dem.tif
│  │  ├─ Q10_depth_max.GangKoudem.2022dem.tif
│  │  ├─ Q25_depth_max.GangKoudem.2022dem.tif
│  │  ├─ Q50_depth_max.GangKoudem.2022dem.tif
│  │  └─ Q100_depth_max.GangKoudem.2022dem.tif
│  ├─ 人口數量分布.gpkg
│  └─ 避難收容處所_清理後.csv
├─ output/
│  ├─ Q1point1_路網分析互動地圖.html
│  ├─ Q10_路網分析互動地圖.html
│  ├─ Q25_路網分析互動地圖.html
│  ├─ Q50_路網分析互動地圖.html
│  ├─ Q100_路網分析互動地圖.html
│  ├─ Q1point1_road_network_analysis_interactive_map.html
│  ├─ Q10_road_network_analysis_interactive_map.html
│  ├─ Q25_road_network_analysis_interactive_map.html
│  ├─ Q50_road_network_analysis_interactive_map.html
│  ├─ Q100_road_network_analysis_interactive_map.html
│  ├─ residential_accessibility_comparison.csv
│  └─ exposure_total_population.html
├─ script/
│  ├─ 路網分析(整合).py
│  └─ 路網分析(整合)_英文版.py
├─ README.md
└─ requirements.txt
```

## 已移除的無關或未使用檔案

已依照兩支主程式實際讀寫的路徑清理下列類型檔案：

- 舊版或重複腳本：`路網分析(整合) - 複製.py`
- Notebook 與 checkpoint：`.ipynb`、`.ipynb_checkpoints/`
- Python 快取：`__pycache__/`
- OSM 快取：`script/cache/`
- 未被主程式直接使用的原始資料或中繼資料，例如：
  - `1995_LU.*`
  - 土壤資料夾 `02_土壤/`
  - `riverpoly/`
  - 行政區與村里原始 shapefile / csv
  - 舊版整合中介成果 `chiayi_city_road_flood_analysis.gpkg`
  - 未被程式直接讀取的 `Geo_RA/*.vrt`
- 舊版輸出：`Q*_整合可及性與道路淹水地圖.html`

## 主程式怎麼運作

### 1. 讀取與整理基礎資料

程式會讀取：

- `data/11_港口溪流域/01_土地利用/2007_LU.shp`
- `data/避難收容處所_清理後.csv`
- `data/Geo_RA/*.tif`
- `data/人口數量分布.gpkg`

主要前處理內容：

- 將土地利用資料統一轉為 `EPSG:3826`
- 挑出住宅用地 `LCODE_C2 == 0502`
- 挑出醫療保健用地 `LCODE_C2 == 0603`
- 將住宅與醫療 polygon 轉為 centroid
- 過濾研究區內或研究區外擴 1 km 內的避難收容處所

### 2. 建立道路網

程式會以研究區外擴 500 公尺的範圍，透過 `osmnx.graph_from_polygon()` 從 OpenStreetMap 抓取道路網，之後：

- 投影到 `EPSG:3826`
- 依道路型別設定預設車速
- 計算每條道路的長度與旅行時間 `travel_time_min`

這一步需要網路連線，因為道路網不是放在專案資料夾中，而是執行時動態下載。

### 3. 可及性分析

每個住宅 centroid、醫療 centroid、避難所點位都會先 snap 到最近的道路節點，接著在五個淹水情境下分別分析：

- `Q1point1`
- `Q10`
- `Q25`
- `Q50`
- `Q100`

每個情境都會把對應的淹水深度 raster 套到道路上，判定哪些道路受淹水影響，再計算：

- 住宅到最近醫療點的距離與時間
- 住宅到最近避難所的距離與時間
- 在可接受時間門檻內是否可達
- 各情境之間的可及性差異

### 4. 風險整合

最後會把路網分析結果與 `人口數量分布.gpkg` 結合，產出：

- `Exposure`: 人口暴露量
- `Hazard`: Q100 情境下道路受淹比例
- `Vulnerability`: 住宅點在 Q100 下的醫療 / 避難可及性脆弱度
- `Risk`: 綜合 hazard、exposure、vulnerability 的風險指標

## 主要輸出成果

中文主程式 `路網分析(整合).py` 會輸出：

- `output/Q1point1_路網分析互動地圖.html`
- `output/Q10_路網分析互動地圖.html`
- `output/Q25_路網分析互動地圖.html`
- `output/Q50_路網分析互動地圖.html`
- `output/Q100_路網分析互動地圖.html`
- `output/住宅可及性綜整比較.csv`
- `output/exposure_total_population.html`

英文主程式 `路網分析(整合)_英文版.py` 會輸出：

- `output/Q1point1_road_network_analysis_interactive_map.html`
- `output/Q10_road_network_analysis_interactive_map.html`
- `output/Q25_road_network_analysis_interactive_map.html`
- `output/Q50_road_network_analysis_interactive_map.html`
- `output/Q100_road_network_analysis_interactive_map.html`
- `output/residential_accessibility_comparison.csv`
- `output/exposure_total_population.html`

兩支程式都還會先產生一張初步檢查用的總覽地圖：

- 中文版：`output/路網分析結果.html`
- 英文版：`output/road_network_analysis_result.html`

如果目前資料夾裡還沒有這兩個檔案，重新執行主程式後就會產生。

## 執行方式

請從 `script/` 資料夾內執行，因為程式使用了相對路徑 `../data` 與 `../output`。

中文版：

```powershell
cd script
python "路網分析(整合).py"
```

英文版：

```powershell
cd script
python "路網分析(整合)_英文版.py"
```

## 注意事項

- 兩支主程式都有 `from IPython.display import display`，因此建議安裝 `ipython`
- 如果要重跑分析，`output/` 內的 HTML 與 CSV 會被覆寫
- 若 OpenStreetMap 下載失敗，通常是網路或連線限制造成，不是本地資料缺失
