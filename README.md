# 嘉義市淹水情境下的路網可及性分析

本專案將原始 `Final Report` 中混合於 Colab notebook 的程式，整理為一個可在本機 Jupyter Lab 中維護的獨立期末報告專案。這一版先聚焦在「路網分析與視覺化」，保留醫院與避難收容處所的可及性分析、繞道分析、關鍵節點識別，以及互動式地圖輸出，並將主要程式彙整為單一 notebook。

## 專案目標

- 比較不同淹水情境下，嘉義市道路網對醫療設施與避難收容處所的可達性變化
- 找出淹水後需要明顯繞道的節點熱區
- 透過路網中介中心性與孤立風險，辨識高風險咽喉節點
- 產出適合期末報告使用的靜態圖與互動式地圖

## 專案結構

```text
Final Report(重建架構)/
├─ data/
│  ├─ chiayi_city_road_flood_analysis.gpkg
│  ├─ 嘉義市醫院診所名單.json
│  └─ 嘉義市避難收容處所.json
├─ output/
│  ├─ figures/
│  ├─ maps/
│  └─ tables/
├─ script/
│  └─ route_network_analysis.ipynb
├─ remote_sensing_analysis/
│  ├─ output/
│    ├─ data/
│    └─ maps/
│  ├─ script/
│  │  └─ historical_flood_extraction.ipynb
├─ README.md
└─ requirements.txt
```

## 資料來源與目前使用資料

這一版只保留路網分析真正需要的資料，避免把尚未整理完成的其他分析一起帶入：

- `chiayi_city_road_flood_analysis.gpkg`
  - 原始 notebook 在 Phase 1 匯出的道路 GeoPackage
  - 已包含嘉義市道路幾何、`length` 欄位，以及各淹水情境的 `impassable_*` 布林欄位
- `嘉義市醫院診所名單.json`
  - 作為醫療設施點位來源
- `嘉義市避難收容處所.json`
  - 作為避難收容處所點位來源

目前不納入 DEM 與完整淹水 shp/zip，因為這一版的核心工作是先穩定完成路網分析與視覺化。

## 執行方式

### 在 Jupyter Lab 中逐段執行

開啟下列 notebook：

- `script/route_network_analysis.ipynb`

這份 notebook 已經依照報告流程整理，且每一段 code cell 前都放入對應的 markdown 說明，方便展示、修改與擴寫。所有主要分析函式也都集中在同一份 notebook 內，不再依賴額外的 `.py` 腳本。

## 主要分析內容

### 1. 設施點位整理與路網貼齊

- 讀取醫院與避難所 JSON
- 篩選與水災分析相關的設施
- 將點位轉成 `EPSG:3826`
- 吸附到最近的路網節點，作為可及性分析的目的地

### 2. 可及性分析

- 以平時路網計算各節點到最近醫院與避難所的距離
- 對每個淹水情境移除不可通行路段
- 重算各節點最短距離，統計新增孤立節點

### 3. 繞道分析

- 計算淹水後與正常情況的距離比值
- 識別需大幅繞道的熱點節點
- 輸出靜態圖與 Excel 詳細表

### 4. 關鍵節點識別

- 估算路網中介中心性
- 與淹水情境下的孤立風險交叉比對
- 找出可能造成系統性交通受阻的咽喉點

### 5. 視覺化輸出

- 設施分布圖
- 可及性比較圖
- 醫院與避難所距離差異圖
- 繞道熱點圖
- 關鍵咽喉節點圖
- Folium 互動式地圖

## 重構原則

本次整理特別針對原始 notebook 常見問題做了處理：

- 移除 Colab 專用絕對路徑
- 全面改為以專案根目錄為基準的相對路徑
- 將重複的 import、重複邏輯與一次性測試 cell 重新整理成 notebook 內的共用函式
- 把輸出路徑集中管理到 `output/`
- 保留單一 notebook 架構，方便期末報告後續持續擴寫

## 後續可擴充方向

- 補回前段非路網主題的資料分析
- 納入更多淹水深度或時間情境的比較摘要
- 增加行政區或里別彙整統計
- 整合 DEM、高程與淹水潛勢圖做更完整的空間解釋
