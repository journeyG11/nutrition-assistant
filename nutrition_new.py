# -*- coding: utf-8 -*-
"""
您的专属营养助手 v2
-------------------
相比 v1 的改动：
1. 真实检索：PDF 切段 + TF-IDF 检索，只把相关段落喂给 AI（原来是把整本书塞进 prompt）
2. 扫描版 PDF 自动识别并跳过，不再假装"读取成功"
3. 新增运动模块：活动系数、MET 消耗、训练计划
4. 7 天曲线真实绘图，记录持久化到 nutrition_history.json
5. API Key 不再硬编码，改为环境变量 / .env / 侧边栏输入
6. 没有 Key 时也能用本地规则给出评估（AI 只是增强）

运行方式（Anaconda Prompt，在项目文件夹下）：
    streamlit run nutrition_new.py
"""

import json
import os
import re
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------
# 路径与全局配置
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PDF_DIR = Path(os.environ.get("NUTRITION_PDF_DIR", str(APP_DIR / "pdf")))
HISTORY_FILE = APP_DIR / "nutrition_history.json"
TRAINING_FILE = APP_DIR / "training_history.json"
PROFILE_FILE = APP_DIR / "profile.json"
PREPROCESS_VERSION = "2026-10-19-2"  # 改文本处理逻辑时改这里，强制重建索引


def _detect_cloud():
    """判断是否跑在 Streamlit Community Cloud（多用户共用同一个进程和文件系统）。"""
    try:
        if os.environ.get("IS_STREAMLIT_CLOUD"):
            return True
        if Path("/mount/src").exists():
            return True
        if os.environ.get("HOME", "").startswith("/home/appuser"):
            return True
    except Exception:
        pass
    return False


IS_CLOUD = _detect_cloud()

# 支持 .env 文件：在项目文件夹里写 DEEPSEEK_API_KEY=sk-xxx 即可，不必改代码
try:
    from dotenv import load_dotenv
    load_dotenv(APP_DIR / ".env")
except Exception:
    pass

st.set_page_config(page_title="您的专属营养助手", page_icon="🥗", layout="wide")

CSS = """
<style>
    :root {
        --brand: #2F8F5B;
        --brand-2: #4CB47F;
        --amber: #E8A33D;
        --blue: #4A90D9;
        --rose: #D96B6B;
    }
    .stApp {
        background:
            radial-gradient(900px 520px at 100% -10%, rgba(47,143,91,.08), transparent 62%),
            radial-gradient(700px 420px at -10% 110%, rgba(232,163,61,.05), transparent 60%);
    }
    .block-container {padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1200px;}

    /* 顶部横幅 */
    .hero {
        display: flex; justify-content: space-between; align-items: center; gap: 20px;
        background:
            radial-gradient(120% 180% at 0% 0%, rgba(47,143,91,.20), transparent 62%),
            linear-gradient(120deg, rgba(47,143,91,0.10), rgba(73,179,125,0.03));
        border: 1px solid rgba(47,143,91,0.18);
        border-radius: 20px;
        padding: 22px 26px;
        margin-bottom: 18px;
    }
    .hero h1 {margin: 0; font-size: 2rem; letter-spacing: .5px; color: var(--text-color);}
    .hero p {margin: .35rem 0 0; opacity: .68; font-size: .95rem;}
    .chips {display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;}
    .chip {
        background: rgba(47,143,91,.12); border: 1px solid rgba(47,143,91,.22);
        padding: 4px 12px; border-radius: 999px; font-size: .8rem;
    }
    .hero-art {font-size: 64px; opacity: .92; filter: drop-shadow(0 8px 16px rgba(47,143,91,.25));}

    /* 健康分环形 */
    .ring-box {
        display: flex; align-items: center; justify-content: center; gap: 10px;
        background: rgba(128,128,128,.10);
        border: 1px solid rgba(128,128,128,.18); border-radius: 14px;
        padding: 10px; min-height: 118px;
        box-shadow: 0 2px 8px rgba(0,0,0,.04);
    }

    /* 小标题左侧色条 */
    h3 {border-left: 4px solid var(--brand); padding-left: 10px; margin-top: 1.3rem !important;}

    /* 标签页：胶囊式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: rgba(47,143,91,0.08); padding: 4px; border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {border-radius: 10px; padding: 6px 16px; background: transparent;}
    .stTabs [aria-selected="true"] {background: var(--brand) !important;}
    .stTabs [aria-selected="true"] p {color: #fff !important;}

    /* 按钮 */
    .stButton > button {
        border-radius: 12px; font-weight: 600;
        border: 1px solid rgba(47,143,91,.25); transition: all .15s ease;
    }
    .stButton > button:hover {transform: translateY(-1px);}
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2F8F5B, #4CB47F);
        border: none; box-shadow: 0 4px 14px rgba(47,143,91,.25);
    }

    /* 输入框圆角 */
    .stTextInput input, .stTextArea textarea, .stNumberInput input,
    div[data-baseweb="select"] > div {border-radius: 10px !important;}

    /* 指标卡片 */
    .mcard {
        background-color: rgba(128,128,128,.10);
        border: 1px solid rgba(128,128,128,.18);
        border-left: 4px solid var(--accent, #2F8F5B);
        border-radius: 14px; padding: 14px 16px; min-height: 96px;
        box-shadow: 0 2px 8px rgba(0,0,0,.04);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .mcard:hover {transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,.07);}
    .mlabel {font-size: .78rem; opacity: .62; letter-spacing: .4px;}
    .mval {font-size: 1.75rem; font-weight: 700; margin-top: 2px; line-height: 1.2;}
    .msub {font-size: .78rem; opacity: .62; margin-top: 2px;}

    /* 进度条 */
    .pbar {background: rgba(128,128,128,.16); border-radius: 999px; height: 8px; overflow: hidden;}
    .pbar > span {display: block; height: 100%; border-radius: 999px;}
    .prow {display: flex; justify-content: space-between; font-size: .8rem; opacity: .7; margin-bottom: 4px;}

    /* 其他 */
    .tag {
        display: inline-block; background: rgba(47,143,91,0.16); color: inherit;
        border-radius: 999px; padding: 3px 12px; margin: 3px 4px 3px 0;
        font-size: 0.86rem;
    }
    .hint {opacity: 0.7; font-size: 0.85rem;}
    .divider-line {height: 1px; background: rgba(128,128,128,0.25); margin: 0.6rem 0 1rem 0;}
    .footer {text-align: center; opacity: .45; font-size: .78rem; margin-top: 2rem;}
    div[data-testid="stAlert"] {border-radius: 14px;}
    /* 侧边栏交给 Streamlit 主题自己处理，避免深色模式下浅底浅字 */
    @media (prefers-color-scheme: dark) {
        .tag {background: rgba(47,143,91,0.28);}
        .hero {background: linear-gradient(120deg, rgba(47,143,91,0.22), rgba(73,179,125,0.08));}
        .mcard, .ring-box {background-color: rgba(255,255,255,.06); border-color: rgba(255,255,255,.14);}
    }
    /* 手机上让图表不抢手势，页面可以正常上下滑动 */
    @media (max-width: 640px) {
        div[data-testid="stPlotlyChart"] {pointer-events: none; touch-action: pan-y;}
        .stTabs [data-baseweb="tab-list"] {flex-wrap: wrap;}
        .hero-art {display: none;}
        .hero h1 {font-size: 1.55rem;}
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# 图表交互配置：关掉滚轮缩放和拖拽，避免手机上滑不动页面
PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False,
                 "dragmode": False, "doubleClick": "reset"}


def metric_card(col, label, value, sub="", color="#2F8F5B"):
    """自绘指标卡，比 st.metric 更容易统一视觉。"""
    col.markdown(
        f'<div class="mcard" style="--accent:{color}">'
        f'<div class="mlabel">{label}</div>'
        f'<div class="mval">{value}</div>'
        f'<div class="msub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def score_ring(score, color="#2F8F5B", size=104):
    """健康分环形仪表：比一个数字更有"体感"。"""
    r = 34
    circ = 2 * 3.1415926 * r
    off = circ * (1 - max(0, min(100, score)) / 100)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 84 84">'
        f'<circle cx="42" cy="42" r="{r}" fill="none" stroke="rgba(128,128,128,.18)" stroke-width="8"/>'
        f'<circle cx="42" cy="42" r="{r}" fill="none" stroke="{color}" stroke-width="8" '
        f'stroke-linecap="round" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{off:.1f}" '
        f'transform="rotate(-90 42 42)"/>'
        f'<text x="42" y="47" text-anchor="middle" font-size="20" font-weight="700" '
        f'fill="currentColor">{score}</text>'
        f'<text x="42" y="62" text-anchor="middle" font-size="8" fill="currentColor" '
        f'opacity=".55">健康分</text>'
        '</svg>'
    )


# --------------------------------------------------------------------------
# 食物数据库（每 100g 的数值；GL 为血糖负荷；anti 为抗炎评分 1-10）
# --------------------------------------------------------------------------
def _f(name, kcal, p, c, f, gl, anti, seasons, unit_g=100, unit="份", tags=()):
    return {
        "name": name, "kcal": kcal, "p": p, "c": c, "f": f,
        "gl": gl, "anti": anti, "seasons": seasons,
        "unit_g": unit_g, "unit": unit, "tags": tuple(tags),
    }


FOODS = [
    # 主食
    _f("米饭", 116, 2.6, 25.9, 0.3, 23, 2, ["all"], 150, "碗"),
    _f("馒头", 223, 7.0, 47.0, 1.1, 41, 2, ["all"], 100, "个"),
    _f("面条", 110, 4.5, 24.0, 0.5, 21, 3, ["all"], 200, "碗"),
    _f("全麦面包", 246, 9.0, 46.0, 3.3, 30, 4, ["all"], 35, "片", ["小麦"]),
    _f("面包", 265, 8.0, 49.0, 4.5, 35, 2, ["all"], 60, "个", ["小麦"]),
    _f("燕麦", 377, 15.0, 66.0, 7.0, 40, 6, ["all"], 40, "份"),
    _f("玉米", 112, 4.0, 22.8, 1.2, 15, 4, ["autumn"], 150, "根"),
    _f("红薯", 86, 1.6, 20.0, 0.1, 13, 6, ["autumn"], 150, "个"),
    _f("土豆", 77, 2.0, 17.5, 0.2, 10, 3, ["autumn", "winter"], 150, "个"),
    _f("小米粥", 46, 1.4, 8.4, 0.7, 6, 4, ["all"], 250, "碗"),
    _f("包子", 227, 8.0, 40.0, 4.0, 25, 2, ["all"], 100, "个", ["小麦"]),
    _f("油条", 386, 6.9, 51.0, 17.6, 45, 1, ["all"], 60, "根", ["小麦"]),
    _f("饺子", 240, 9.0, 30.0, 9.0, 22, 3, ["all"], 20, "个", ["小麦"]),
    # 蛋白质
    _f("鸡蛋", 144, 13.3, 2.8, 8.8, 0, 5, ["all"], 50, "个", ["蛋"]),
    _f("牛奶", 54, 3.0, 3.4, 3.2, 3, 4, ["all"], 250, "杯", ["乳"]),
    _f("酸奶", 72, 2.5, 9.3, 2.7, 6, 5, ["all"], 150, "杯", ["乳"]),
    _f("豆浆", 31, 3.0, 1.2, 1.6, 1, 5, ["all"], 250, "杯", ["大豆"]),
    _f("豆腐", 82, 8.1, 4.2, 3.7, 2, 6, ["all"], 150, "份", ["大豆"]),
    _f("鸡胸肉", 118, 23.0, 0.0, 1.9, 0, 7, ["all"], 100, "份"),
    _f("鸡腿", 181, 16.0, 0.0, 13.0, 0, 4, ["all"], 100, "个"),
    _f("牛肉", 125, 19.9, 2.0, 4.2, 0, 4, ["all"], 100, "份", ["铁"]),
    _f("猪瘦肉", 143, 20.3, 1.5, 6.2, 0, 2, ["all"], 100, "份"),
    _f("三文鱼", 139, 17.2, 0.0, 7.8, 0, 9, ["all"], 100, "份"),
    _f("鱼", 104, 18.0, 0.0, 3.0, 0, 8, ["all"], 100, "份"),
    _f("虾", 93, 18.6, 2.8, 0.8, 0, 7, ["all"], 100, "份", ["甲壳"]),
    _f("火腿肠", 212, 14.0, 10.0, 14.0, 3, 1, ["all"], 50, "根"),
    # 蔬菜
    _f("菠菜", 28, 2.9, 4.5, 0.4, 2, 8, ["spring", "autumn"], 100, "份", ["铁"]),
    _f("西兰花", 36, 4.1, 4.3, 0.6, 2, 9, ["autumn", "winter"], 100, "份"),
    _f("生菜", 15, 1.3, 2.0, 0.3, 1, 7, ["spring"], 100, "份"),
    _f("番茄", 20, 0.9, 4.0, 0.2, 2, 8, ["summer"], 150, "个"),
    _f("黄瓜", 16, 0.8, 2.9, 0.2, 1, 6, ["summer"], 150, "根"),
    _f("胡萝卜", 39, 1.0, 8.8, 0.2, 3, 7, ["autumn"], 100, "份"),
    _f("白菜", 17, 1.5, 3.2, 0.1, 1, 6, ["winter"], 150, "份"),
    _f("蘑菇", 24, 2.7, 4.1, 0.1, 1, 7, ["all"], 100, "份"),
    _f("海带", 13, 1.2, 2.1, 0.1, 1, 6, ["all"], 100, "份", ["碘"]),
    _f("紫菜", 250, 26.7, 44.0, 1.1, 20, 6, ["all"], 10, "张"),
    _f("南瓜", 23, 0.7, 5.3, 0.1, 4, 7, ["autumn"], 150, "份"),
    _f("西兰苔", 40, 4.5, 4.0, 0.8, 1, 8, ["autumn", "winter"], 150, "份"),
    _f("芦笋", 22, 2.6, 3.0, 0.1, 1, 7, ["spring"], 150, "份"),
    _f("秋葵", 37, 2.0, 7.0, 0.2, 2, 6, ["summer"], 150, "份"),
    _f("油麦菜", 15, 1.2, 2.1, 0.4, 1, 6, ["spring", "autumn"], 150, "份"),
    _f("芥蓝", 26, 2.8, 3.7, 0.4, 1, 7, ["autumn"], 150, "份"),
    _f("菜心", 20, 1.7, 2.6, 0.3, 1, 7, ["spring"], 150, "份"),
    _f("娃娃菜", 13, 1.2, 2.0, 0.2, 1, 6, ["winter"], 200, "份"),
    _f("冬瓜", 12, 0.4, 2.6, 0.2, 1, 5, ["summer"], 200, "份"),
    _f("茄子", 23, 1.1, 4.9, 0.2, 2, 4, ["summer"], 150, "个"),
    _f("青椒", 22, 1.0, 5.4, 0.2, 2, 6, ["summer"], 100, "个"),
    _f("洋葱", 40, 1.1, 9.0, 0.2, 3, 7, ["all"], 100, "个"),
    _f("豆角", 30, 2.4, 5.7, 0.2, 2, 5, ["summer"], 150, "份"),
    _f("莲藕", 70, 1.9, 16.4, 0.2, 5, 4, ["autumn"], 150, "份"),
    _f("木耳", 27, 1.5, 6.0, 0.2, 1, 7, ["all"], 50, "份"),
    _f("金针菇", 32, 2.4, 6.0, 0.4, 1, 6, ["all"], 100, "份"),
    _f("豆芽", 16, 1.7, 2.6, 0.1, 1, 5, ["all"], 150, "份"),
    _f("芹菜", 22, 1.2, 4.5, 0.2, 1, 6, ["all"], 150, "份"),
    # 水果
    _f("苹果", 53, 0.4, 13.7, 0.2, 6, 6, ["autumn"], 200, "个"),
    _f("香蕉", 93, 1.4, 22.0, 0.2, 12, 4, ["summer"], 120, "根"),
    _f("橙子", 48, 0.8, 11.1, 0.2, 6, 7, ["winter"], 200, "个"),
    _f("蓝莓", 57, 0.7, 14.5, 0.3, 7, 9, ["summer"], 100, "盒"),
    _f("草莓", 32, 1.0, 7.1, 0.2, 5, 7, ["spring"], 150, "份"),
    _f("西瓜", 31, 0.6, 7.9, 0.1, 5, 4, ["summer"], 300, "份"),
    _f("葡萄", 45, 0.5, 10.3, 0.2, 8, 4, ["autumn"], 150, "份"),
    # 油脂坚果
    _f("橄榄油", 884, 0.0, 0.0, 100.0, 0, 8, ["all"], 10, "勺"),
    _f("花生油", 899, 0.0, 0.0, 99.9, 0, 2, ["all"], 10, "勺"),
    _f("核桃", 646, 14.9, 19.1, 58.8, 3, 7, ["autumn"], 20, "个", ["坚果"]),
    _f("杏仁", 578, 21.0, 21.0, 50.0, 2, 6, ["all"], 20, "份", ["坚果"]),
    _f("花生", 567, 24.0, 21.0, 44.0, 2, 4, ["autumn"], 30, "份", ["坚果"]),
    # 零食饮料
    _f("可乐", 43, 0.0, 10.6, 0.0, 9, 1, ["all"], 330, "罐"),
    _f("奶茶", 78, 1.0, 14.0, 2.0, 10, 1, ["all"], 500, "杯"),
    _f("蛋糕", 347, 5.0, 60.0, 10.0, 35, 1, ["all"], 100, "块", ["小麦", "蛋", "乳"]),
    _f("薯片", 536, 7.0, 49.0, 35.0, 30, 1, ["all"], 50, "袋"),
    _f("巧克力", 546, 4.9, 59.0, 31.0, 25, 2, ["all"], 30, "块"),
    _f("咖啡", 2, 0.1, 0.0, 0.0, 0, 3, ["all"], 250, "杯"),
    _f("绿茶", 1, 0.0, 0.0, 0.0, 0, 7, ["all"], 250, "杯"),
    # 常见家常菜（按一份的 100g 数值折算）
    _f("番茄炒蛋", 88, 4.0, 3.2, 6.4, 2, 5, ["all"], 250, "份", ["蛋"]),
    _f("西红柿炒蛋", 88, 4.0, 3.2, 6.4, 2, 5, ["all"], 250, "份", ["蛋"]),
    _f("辣椒炒肉", 128, 6.0, 3.0, 10.4, 2, 3, ["all"], 250, "份"),
    _f("青椒肉丝", 112, 5.6, 2.8, 8.4, 1, 3, ["all"], 250, "份"),
    _f("鱼香肉丝", 132, 5.6, 4.8, 8.8, 3, 3, ["all"], 250, "份"),
    _f("宫保鸡丁", 136, 7.2, 4.8, 9.6, 2, 3, ["all"], 250, "份", ["花生"]),
    _f("红烧肉", 208, 7.2, 2.4, 19.2, 1, 2, ["all"], 150, "份"),
    _f("麻婆豆腐", 104, 6.0, 3.2, 7.6, 1, 5, ["all"], 250, "份", ["大豆"]),
    _f("土豆丝", 72, 1.2, 8.0, 4.0, 5, 3, ["all"], 250, "份"),
    _f("蛋炒饭", 184, 4.8, 22.0, 8.0, 14, 2, ["all"], 250, "份", ["蛋"]),
    _f("炒青菜", 48, 1.2, 2.4, 3.6, 1, 6, ["all"], 250, "份"),
    _f("清炒时蔬", 48, 1.2, 2.4, 3.6, 1, 6, ["all"], 250, "份"),
    _f("牛肉面", 100, 4.4, 14.0, 3.2, 9, 3, ["all"], 500, "碗"),
    _f("炸酱面", 112, 4.0, 15.6, 3.6, 10, 2, ["all"], 500, "碗", ["小麦"]),
    _f("小笼包", 230, 9.0, 26.0, 9.0, 18, 2, ["all"], 130, "笼", ["小麦"]),
    _f("煎饼果子", 180, 6.0, 22.0, 7.2, 16, 2, ["all"], 250, "个", ["小麦", "蛋"]),
    _f("汉堡", 250, 12.5, 22.5, 12.5, 20, 1, ["all"], 200, "个", ["小麦", "乳"]),
    _f("炸鸡", 300, 17.5, 10.0, 21.0, 5, 1, ["all"], 200, "份"),
    _f("麻辣烫", 120, 5.0, 8.0, 7.0, 6, 2, ["all"], 500, "份"),
    _f("火锅", 180, 8.0, 8.0, 12.0, 4, 2, ["all"], 500, "份"),
]

DATA_CANDIDATES = [APP_DIR / "foods_cn.json",
                   APP_DIR / "data" / "foods_cn.json",
                   APP_DIR / "pdf" / "foods_cn.json"]


def load_food_data():
    """加载《中国食物成分表》转换后的扩展库（1600+ 条），缺文件时只用手写库。"""
    path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if path is None:
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for it in raw:
        name = str(it.get("name", "")).strip()
        if len(name) < 2:
            continue
        out.append({
            "name": name,
            "kcal": float(it.get("kcal", 0)),
            "p": float(it.get("p", 0)),
            "c": float(it.get("c", 0)),
            "f": float(it.get("f", 0)),
            "gl": float(it.get("gl", 0)),
            "anti": int(it.get("anti", 5)),
            "seasons": ["all"],
            "unit_g": 100,
            "unit": "份",
            "tags": (),
        })
    return out


DATA_FOODS = load_food_data()

DISH_CANDIDATES = [APP_DIR / "dishes_cn.json",
                   APP_DIR / "data" / "dishes_cn.json",
                   APP_DIR / "pdf" / "dishes_cn.json"]


def load_dishes():
    """加载菜品库（133 道常见菜），按「一份」估算。"""
    path = next((p for p in DISH_CANDIDATES if p.exists()), None)
    if path is None:
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for it in raw:
        name = str(it.get("name", "")).strip()
        if len(name) < 2:
            continue
        out.append({
            "name": name,
            "kcal": float(it.get("kcal", 0)),
            "p": float(it.get("p", 0)),
            "c": float(it.get("c", 0)),
            "f": float(it.get("f", 0)),
            "gl": float(it.get("c", 0)) * 0.6,
            "anti": int(it.get("anti", 5)),
            "seasons": ["all"],
            "unit_g": float(it.get("serving_g", 100)),
            "unit": "份",
            "tags": (),
        })
    return out


DISH_FOODS = load_dishes()
FOOD_INDEX = {}
for _food in DATA_FOODS + DISH_FOODS + FOODS:   # 手写库放最后，同名时优先级最高
    FOOD_INDEX[_food["name"]] = _food

ALIASES = {name: name for name in FOOD_INDEX}
ALIASES.update({
    "白米饭": "米饭", "米饭": "米饭",
    "鸡胸": "鸡胸肉", "鸡胸肉": "鸡胸肉",
    "纯牛奶": "牛奶", "牛奶": "牛奶",
    "牛肉": "牛肉", "牛排": "牛肉",
    "鸡蛋": "鸡蛋", "煮鸡蛋": "鸡蛋", "水煮蛋": "鸡蛋",
    "全麦面包": "全麦面包", "面包": "面包",
    "燕麦片": "燕麦", "燕麦": "燕麦",
    "西兰花": "西兰花", "西蓝花": "西兰花",
    "酸奶": "酸奶", "豆浆": "豆浆",
    "虾仁": "虾", "虾": "虾",
    "可乐": "可乐", "奶茶": "奶茶", "蛋糕": "蛋糕",
    "红薯": "红薯", "地瓜": "红薯", "玉米": "玉米",
    "苹果": "苹果", "香蕉": "香蕉", "橙子": "橙子",
    "蓝莓": "蓝莓", "草莓": "草莓", "西瓜": "西瓜",
    "坚果": "核桃", "核桃": "核桃",
    "鱼": "鱼", "三文鱼": "三文鱼",
    "西蓝花": "西兰花", "花生米": "花生", "炒鸡蛋": "鸡蛋",
    "煎蛋": "鸡蛋",
})

# 数据集里常带括号/方括号说明（如「鲈鱼［鲈花］」「西兰花（绿菜花）」），
# 去掉括号得到本名，括号里用顿号分隔的别名也一并登记
for _name in list(FOOD_INDEX):
    _base = re.split(r"[（(［\[]", _name)[0].strip()
    if len(_base) >= 2 and _base not in FOOD_INDEX and _base not in ALIASES:
        ALIASES[_base] = _name
    _m = re.search(r"[（(［\[](.*?)[）)］\]]", _name)
    if _m:
        for _alias in re.split(r"[、,，]", _m.group(1)):
            _alias = _alias.strip()
            if len(_alias) >= 2 and _alias not in FOOD_INDEX:
                ALIASES.setdefault(_alias, _name)

# 长词优先匹配，避免"全麦面包"被"面包"抢先；合并成一个正则，速度快很多
ALIAS_KEYS = sorted(ALIASES.keys(), key=len, reverse=True)
FOOD_PATTERN = re.compile("|".join(re.escape(k) for k in ALIAS_KEYS))

# --------------------------------------------------------------------------
# 补剂库：常见剂量、服用时间、注意事项
# --------------------------------------------------------------------------
SUPP_CANDIDATES = [APP_DIR / "supplements_cn.json",
                   APP_DIR / "data" / "supplements_cn.json",
                   APP_DIR / "pdf" / "supplements_cn.json"]
DEFAULT_SUPPLEMENTS = ["乳清蛋白粉", "维生素D3", "铁剂", "钙片",
                       "鱼油（Omega-3）", "复合维生素", "肌酸（一水肌酸）"]


def load_supplements():
    path = next((p for p in SUPP_CANDIDATES if p.exists()), None)
    if path is not None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            db = {d["name"]: d for d in data if d.get("name")}
            if db:
                return db
        except Exception:
            pass
    return {n: {"name": n, "dose": "", "timing": "", "note": "", "caution": "", "tags": []}
            for n in DEFAULT_SUPPLEMENTS}


SUPP_DB = load_supplements()


def supplement_tips(selected, profile):
    """根据所选补剂给出搭配和使用提示（通用信息，不构成医学建议）。"""
    s = set(selected)
    tips = []
    if "钙片" in s and ("铁剂" in s or "铁+叶酸复合" in s):
        tips.append("钙和铁同服会互相影响吸收，建议间隔 2 小时")
    if "钙片" in s and "锌" in s:
        tips.append("钙和锌有吸收竞争，建议错开服用")
    if ("铁剂" in s or "铁+叶酸复合" in s) and "维生素C" in s:
        tips.append("维生素C 能促进铁吸收，可以一起吃")
    if "铁剂" in s or "铁+叶酸复合" in s:
        tips.append("铁剂避免与茶、咖啡同服，间隔 1-2 小时")
    if "锌" in s and "铁剂" in s:
        tips.append("锌和铁同服会竞争吸收，建议分开")
    if "褪黑素" in s:
        tips.append("褪黑素睡前 30-60 分钟服用，服后不要开车")
    if "咖啡因" in s and profile.get("sleep_h", 7) < 7:
        tips.append("你睡眠偏少，咖啡因建议只在上午，下午后不要再摄入")
    if "肌酸（一水肌酸）" in s:
        tips.append("补肌酸期间每天饮水 1.5-2 L")
    if {"乳清蛋白粉", "大豆蛋白粉", "增肌粉"} & s:
        tips.append("蛋白粉要计入当日蛋白质总量，别只算三餐")
    if "益生菌" in s:
        tips.append("益生菌与抗生素间隔 2 小时，水温低于 40℃")
    if "鱼油（Omega-3）" in s:
        tips.append("鱼油与抗凝药同用前请先咨询医生")
    return tips


SEASON_CN = {"spring": "春", "summer": "夏", "autumn": "秋", "winter": "冬"}
SEASON_MAP = {"春季": "spring", "夏季": "summer", "秋季": "autumn", "冬季": "winter"}
EXERCISES = ["深蹲", "硬拉", "卧推", "划船", "肩推", "引体向上", "俯卧撑",
             "弓步蹲", "腿举", "臀桥", "平板支撑", "卷腹", "哑铃弯举", "绳索下压"]
VEG_NAMES = {
    "菠菜", "西兰花", "生菜", "番茄", "黄瓜", "胡萝卜", "白菜", "蘑菇", "海带",
    "南瓜", "西兰苔", "芦笋", "秋葵", "油麦菜", "芥蓝", "菜心", "娃娃菜", "冬瓜",
    "茄子", "青椒", "洋葱", "豆角", "莲藕", "木耳", "金针菇", "豆芽", "芹菜",
}
# 家常菜里隐含的蔬菜量（按一份估算），否则蔬菜量会被低估
DISH_VEG = {
    "炒青菜": 250, "清炒时蔬": 250, "番茄炒蛋": 100, "西红柿炒蛋": 100,
    "辣椒炒肉": 100, "青椒肉丝": 100, "鱼香肉丝": 80, "土豆丝": 200,
    "麻婆豆腐": 0, "宫保鸡丁": 60, "红烧肉": 0, "蛋炒饭": 30,
}

MET = {
    "快走": 3.5, "慢跑": 7.0, "跑步": 9.0, "骑行": 6.0, "游泳": 8.0,
    "跳绳": 12.0, "力量训练": 5.0, "瑜伽": 3.0, "八段锦": 3.0,
    "羽毛球": 5.5, "乒乓球": 4.0, "健身操": 6.0, "几乎不运动": 1.0,
}


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
def season_today(month=None):
    m = month or datetime.now().month
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "autumn"
    return "winter"


NUM_UNIT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(kg|千克|g|克|ml|毫升)")
SERVING = re.compile(r"^\s*(半|一|两|二|三|四|五|\d+)\s*(个|碗|杯|片|根|勺|份|盒|袋|块|张|罐|条)")
CN_NUM = {"半": 0.5, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5}


def _grams_after(text, pos, food):
    """在食物名之后解析重量，识别 "100g" / "一碗" 这类写法。"""
    tail = text[pos:pos + 12]
    m = NUM_UNIT.match(tail)
    if m and m.group(1):
        val = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit in ("kg", "千克"):
            return val * 1000
        if unit in ("ml", "毫升"):
            return val  # 近似按 1ml≈1g
        return val
    s = SERVING.match(tail)
    if s:
        raw = s.group(1)
        count = float(raw) if raw.isdigit() else CN_NUM.get(raw, 1)
        return food["unit_g"] * count
    return food["unit_g"]


def estimate_meal(text):
    """把 "鸡胸肉100g，米饭一碗" 解析成营养估算。未识别食材会单独列出。"""
    text = (text or "").strip()
    result = {"kcal": 0.0, "p": 0.0, "c": 0.0, "f": 0.0, "gl": 0.0,
              "items": [], "unknown": [], "veg_g": 0.0}
    if not text:
        return result

    taken = []
    for m in FOOD_PATTERN.finditer(text):
        key = m.group(0)
        if any(a <= m.start() < b for a, b in taken):
            continue
        taken.append((m.start(), m.end()))
        food = FOOD_INDEX[ALIASES[key]]
        grams = _grams_after(text, m.end(), food)
        ratio = grams / 100.0
        result["kcal"] += food["kcal"] * ratio
        result["p"] += food["p"] * ratio
        result["c"] += food["c"] * ratio
        result["f"] += food["f"] * ratio
        result["gl"] += food["gl"] * ratio
        if food["name"] in VEG_NAMES:
            result["veg_g"] += grams
        result["veg_g"] += DISH_VEG.get(food["name"], 0)
        result["items"].append({"name": food["name"], "grams": round(grams),
                                "kcal": round(food["kcal"] * ratio)})

    # 没命中任何食材的片段（例如"辣椒炒肉一份"）单独列出，避免被悄悄当成 0
    leftovers = []
    for seg in re.finditer(r"[^，,、;；\s]+", text):
        if not any(seg.start() < b and seg.end() > a for a, b in taken):
            leftovers.append(seg.group(0))
    result["unknown"] = leftovers[:8]
    return result


def merge_estimates(ests):
    total = {"kcal": 0.0, "p": 0.0, "c": 0.0, "f": 0.0, "gl": 0.0, "veg_g": 0.0,
             "items": [], "unknown": []}
    for e in ests:
        for k in ("kcal", "p", "c", "f", "gl", "veg_g"):
            total[k] += e[k]
        total["items"].extend(e["items"])
        total["unknown"].extend(e["unknown"])
    return total


def local_risks(total, profile):
    risks = []
    if total["kcal"] > profile["target"] + 250:
        risks.append(f"热量偏高：估算 {total['kcal']:.0f} kcal，比目标 {profile['target']:.0f} kcal 多约 {total['kcal'] - profile['target']:.0f} kcal")
    if total["kcal"] < profile["target"] - 350:
        risks.append(f"热量偏低：估算 {total['kcal']:.0f} kcal，长期低于目标易掉肌肉、影响代谢")
    protein_need = profile.get("protein_need", profile["weight"])
    if total["p"] < protein_need * 0.8:
        risks.append(f"蛋白质不足：约 {total['p']:.0f} g，建议至少 {protein_need:.0f} g/天")
    if total["veg_g"] < 300:
        risks.append(f"蔬菜偏少：约 {total['veg_g']:.0f} g，建议 300-500 g/天，补充膳食纤维")
    if total["gl"] > 80:
        risks.append(f"血糖负荷偏高（GL≈{total['gl']:.0f}），建议主食换成杂粮并放慢进食速度")
    names = [i["name"] for i in total["items"]]
    if any(n in names for n in ("可乐", "奶茶", "蛋糕", "巧克力")):
        risks.append("含添加糖食物：当天建议不再叠加甜饮或甜点")
    if any(n in names for n in ("油条", "薯片")):
        risks.append("含油炸/膨化食品：反式脂肪与钠偏高，建议替换为坚果或水果")
    if profile.get("sleep_h", 7) < 7:
        risks.append(f"睡眠不足：平均 {profile['sleep_h']:.1f} 小时，建议 7-9 小时，"
                     "睡眠不足会放大饥饿感和肌肉流失")
    if profile.get("water_ml", 1500) < 1500:
        risks.append(f"饮水偏少：约 {profile['water_ml']} ml，建议 1500-1700 ml，分次喝")
    if profile.get("fasting_glu", 0) >= 6.1:
        risks.append(f"空腹血糖 {profile['fasting_glu']:.1f} mmol/L 偏高（≥6.1），"
                     "建议复查并咨询医生，不要只靠饮食自行判断")
    bf = profile.get("body_fat", 0)
    if bf:
        limit = 30 if profile["gender"] == "女" else 25
        if bf >= limit:
            risks.append(f"体脂率 {bf:.0f}% 偏高（{profile['gender']}性参考上限约 {limit}%）")
    if profile.get("supplements"):
        risks.append("已记录补剂：" + "、".join(profile["supplements"])
                     + "（补剂不计入饮食热量；蛋白粉要算进当日蛋白质）")
    if not risks:
        risks.append("未发现明显的膳食结构问题，继续保持")
    return risks


def allergens_in(total):
    found = {}
    for item in total["items"]:
        food = FOOD_INDEX.get(item["name"])
        if not food:
            continue
        for tag in food["tags"]:
            if tag in ("小麦", "乳", "甲壳", "坚果", "蛋", "大豆"):
                found.setdefault(tag, set()).add(item["name"])
    label = {"小麦": "🍞 小麦/麸质", "乳": "🥛 乳制品", "甲壳": "🦐 甲壳类",
             "坚果": "🥜 坚果", "蛋": "🥚 蛋类", "大豆": "🫘 大豆"}
    return [f"{label[k]}：{'、'.join(sorted(v))}" for k, v in found.items()]


def health_score(total, profile, risks):
    score = 100
    diff = abs(total["kcal"] - profile["target"])
    score -= min(30, diff / max(profile["target"], 1) * 40)
    if total["p"] < profile.get("protein_need", profile["weight"]) * 0.8:
        score -= 12
    if total["veg_g"] < 300:
        score -= 10
    if total["gl"] > 80:
        score -= 10
    score -= 8 * max(0, len([r for r in risks if "未发现" not in r]) - 2)
    return int(max(35, min(100, round(score))))


def gap_advice(total, profile, ex):
    """算出今天还差什么，用具体食物给出补齐方案。"""
    tips = []
    dp = profile["protein_need"] - total["p"]
    if dp >= 8:
        if dp <= 12:
            tips.append(("蛋白质还差", "水煮蛋 2 个", "约 14 g 蛋白 / 140 kcal"))
        elif dp <= 25:
            tips.append(("蛋白质还差", "无糖酸奶 200 g + 鸡蛋 1 个", "约 19 g 蛋白 / 240 kcal"))
        else:
            tips.append(("蛋白质还差", "鸡胸肉 150 g 或豆腐 200 g", f"约 30 g 蛋白，缺口 {dp:.0f} g"))
    if total["veg_g"] < 300:
        tips.append(("蔬菜还差", "炒青菜一份（约 250 g）",
                     f"补 {300 - total['veg_g']:.0f} g / 约 120 kcal"))
    dk = ex["target"] - total["kcal"]
    if dk > 250:
        tips.append(("热量还差", "杂粮饭半碗 + 一小把坚果", f"约 {min(dk, 300):.0f} kcal"))
    elif dk < -250:
        tips.append(("热量超出", "晚餐主食减半，或饭后快走 30 分钟", f"超出 {-dk:.0f} kcal"))
    if not tips:
        tips.append(("三项达标", "保持今天的搭配", "热量、蛋白质、蔬菜都在合理区间"))
    return tips[:3]


def exercise_plan(profile, activity_factor, ex_type, minutes, sessions):
    met = MET.get(ex_type, 3.5)
    per_session = met * 3.5 * profile["weight"] / 200 * minutes
    weekly = per_session * sessions
    daily_avg = weekly / 7
    tdee = profile["bmr"] * activity_factor + daily_avg
    deficit = profile["weight_loss"] / 100 * 500
    surplus = profile.get("muscle", 0) / 100 * 300
    floor = 1200 if profile["gender"] == "女" else 1500
    target = max(tdee + surplus - deficit, floor, profile["bmr"] * 1.05)

    rows = []
    if profile.get("ex_sessions", 0) == 0 or profile.get("ex_type") == "几乎不运动":
        rows = [
            ("日常步行", "6000 步", "每天", "不追求运动量，先把日常活动补起来"),
            ("久坐提醒", "每小时起身 2-3 分钟", "学习/工作时", "减少久坐对代谢和腰背的影响"),
            ("爬楼梯 / 快走", "10-15 分钟", "每天 1-2 次", "最容易坚持的入门活动"),
        ]
    elif profile["period"] == "经期中":
        rows = [
            ("散步 / 拉伸", "20-30 分钟", "每天", "经期以低强度为主，缓解腹部不适"),
            ("瑜伽（阴瑜伽/修复）", "20 分钟", "3-4 次/周", "避免倒立和高强度跳跃"),
            ("补铁饮食", "—", "每天", "红肉、菠菜、动物肝脏搭配维C促进吸收"),
        ]
    elif profile["exam"] == "是":
        rows = [
            ("快走", "20 分钟", "每天", "提升专注力，缓解久坐"),
            ("跳绳间歇", "10 分钟（40秒跳+20秒歇）", "3 次/周", "时间短、心率提升快"),
            ("睡前拉伸", "10 分钟", "每天", "改善睡眠质量"),
        ]
    else:
        if profile.get("muscle", 0) >= 50:
            rows.append(("力量训练（深蹲/卧推/划船/硬拉）", "45 分钟", "3-4 次/周",
                         "增肌核心：渐进超负荷，每组留 1-2 次余力"))
            rows.append(("有氧（快走/骑车）", "20-30 分钟", "1-2 次/周",
                         "控制有氧量，避免影响肌肉恢复"))
            rows.append(("蛋白质分配", "每餐 25-30 g", "每天",
                         "分餐摄入比一顿吃够更利于肌肉合成"))
        if profile["weight_loss"] >= 60:
            rows.append(("力量训练（深蹲/俯卧撑/弹力带）", "30 分钟", "3 次/周", "保住肌肉，提高基础代谢"))
            rows.append(("慢跑或快走", "30-40 分钟", "3 次/周", "增加热量缺口，注意循序渐进"))
        if profile["sugar_ctrl"] >= 60:
            rows.append(("饭后散步", "15 分钟", "每天三餐后", "餐后步行能明显降低血糖峰值"))
        if profile["anti_inflammatory"] >= 60:
            rows.append(("中等强度有氧", "30 分钟", "4 次/周", "抗炎以中等强度、规律性为关键"))
        rows.append(("拉伸 / 泡沫轴放松", "10 分钟", "每天", "减少训练后的炎症反应"))

    return {
        "per_session": per_session,
        "weekly": weekly,
        "daily_avg": daily_avg,
        "tdee": tdee,
        "target": target,
        "rows": rows,
    }


def protein_target(profile):
    """不同目标下的每日蛋白质目标（g）。"""
    w = profile["weight"]
    if profile.get("muscle", 0) >= 50:
        return w * 1.6          # 增肌：1.6 g/kg
    if profile["weight_loss"] >= 50:
        return w * 1.2          # 减脂：1.2 g/kg 保肌肉
    return w * 1.0


# --------------------------------------------------------------------------
# PDF 知识库：加载 → 切段 → 检索
# --------------------------------------------------------------------------
def _pdf_signature():
    if not PDF_DIR.exists():
        return ()
    sig = []
    for p in sorted(PDF_DIR.glob("*.pdf")):
        try:
            s = p.stat()
            sig.append((p.name, s.st_size, int(s.st_mtime)))
        except OSError:
            continue
    return tuple(sig)


def chunk_text(text, size=700, overlap=120):
    # 有些 PDF（例如竖排/逐字排版）会把每个字单独放一行：换行密度很高时直接全部拼回
    if text.count("\n") > max(20, len(text) // 5):
        text = text.replace("\n", "")
    else:
        text = re.sub(
            r"(?<=[\u4e00-\u9fa5，。、；：（）《》“”0-9A-Za-z%])"
            r"\n(?=[\u4e00-\u9fa5，。、；：（）《》“”0-9A-Za-z%（])",
            "", text,
        )
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + size, n)
        if j < n:
            window = text[i + size // 2:j]
            cut = max(window.rfind(ch) for ch in "。！？；\n")
            if cut > 0:
                j = i + size // 2 + cut + 1
        piece = text[i:j].strip()
        if piece:
            if len(piece) < 60 and chunks:
                chunks[-1] = chunks[-1] + piece  # 页尾零碎文字并入上一段
            elif len(piece) >= 60:
                chunks.append(piece)
        if j >= n:
            break
        i = max(j - overlap, i + 1)
    return chunks


@st.cache_data(show_spinner=False)
def load_corpus(sig, version):
    """返回 (chunks, files_info)。扫描版 PDF 会被标记并跳过。"""
    chunks, files_info = [], []
    for name, _size, _mtime in sig:
        path = PDF_DIR / name
        info = {"name": name, "pages": 0, "chars": 0, "chunks": 0, "status": "ok", "note": ""}
        try:
            reader = PdfReader(str(path))
            info["pages"] = len(reader.pages)
            # 先抽样，判断有没有文字层（扫描版会全是空字符串）
            n = len(reader.pages)
            sample_idx = sorted({int(n * k / 6) for k in range(6)} | {0, min(1, n - 1)})
            sample_chars = 0
            for idx in sample_idx:
                if idx < n:
                    sample_chars += len((reader.pages[idx].extract_text() or "").strip())
            if sample_chars < 50:
                info["status"] = "scanned"
                info["note"] = "扫描版，没有文字层，已跳过（需要 OCR 才能读取）"
                files_info.append(info)
                continue
            for i, page in enumerate(reader.pages):
                txt = (page.extract_text() or "").strip()
                if not txt:
                    continue
                info["chars"] += len(txt)
                # 短文档（如几页的要点版）切细一点，长文档保持大块，避免上下文碎片化
                size = 600 if n <= 10 else 700
                for ch in chunk_text(txt, size=size, overlap=size // 6):
                    chunks.append({"source": name, "page": i + 1, "text": ch})
                    info["chunks"] += 1
        except Exception as exc:  # 损坏的 PDF 不应该让整个页面挂掉
            info["status"] = "error"
            info["note"] = f"读取失败：{exc}"
        files_info.append(info)
    return chunks, files_info


EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


@st.cache_resource(show_spinner=False)
def get_embedder():
    """本地向量模型（约 50MB）。没装 fastembed 时返回 None，程序自动退回关键词检索。"""
    try:
        from fastembed import TextEmbedding
        return TextEmbedding(EMBED_MODEL, cache_dir=str(APP_DIR / ".fastembed"))
    except Exception:
        return None


def build_embeddings(chunks, sig, version):
    model = get_embedder()
    if model is None or not chunks:
        return None
    key = hashlib.md5(
        (version + "|" + "|".join(f"{a}:{b}:{c}" for a, b, c in sig)).encode("utf-8")
    ).hexdigest()[:12]
    cache = APP_DIR / f".emb_cache_{key}.npz"
    if cache.exists():
        try:
            return np.load(cache)["emb"]
        except Exception:
            pass
    vecs = np.array(list(model.embed([c["text"] for c in chunks], batch_size=32)),
                    dtype=np.float32)
    try:
        np.savez_compressed(cache, emb=vecs)
    except Exception:
        pass
    return vecs


@st.cache_resource(show_spinner=False)
def build_index(sig, version):
    chunks, files_info = load_corpus(sig, version)
    if not chunks:
        return None, chunks, files_info
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1, max_features=60000)
    matrix = vec.fit_transform([c["text"] for c in chunks])
    emb = build_embeddings(chunks, sig, version)
    return {"tfidf": (vec, matrix), "emb": emb}, chunks, files_info


SYNONYMS = {
    "运动": "身体活动 锻炼 活动量",
    "走": "步行 走路 身体活动",
    "喝水": "饮水 饮用水",
    "饮水": "喝水 饮用水",
    "缺口": "能量限制 能量负平衡",
    "热量": "能量 摄入",
    "减脂": "减重 肥胖 超重",
    "减重": "减脂 肥胖 能量缺口",
    "糖": "血糖 添加糖 精制糖",
    "盐": "钠 食盐",
    "蛋白质": "蛋白 优质蛋白",
    "多少": "推荐 摄入量",
    "吃": "摄入 膳食",
    "早餐": "三餐 膳食安排",
}

# 提问里的虚词会稀释匹配，检索前先去噪
QUERY_STOPWORDS = ["每天", "每周", "每日", "应该", "多少", "怎么", "什么", "如何",
                   "建议", "可以", "是否", "适合", "有哪些", "吗", "呢", "的", "了",
                   "是", "吃", "喝", "要"]


def clean_query(query):
    original = query
    for w in QUERY_STOPWORDS:
        query = query.replace(w, " ")
    query = re.sub(r"\s+", " ", query).strip()
    return query if len(query) >= 2 else original


def expand_query(query):
    extra = [v for k, v in SYNONYMS.items() if k in query]
    return query + " " + " ".join(extra)


def retrieve(index, chunks, query, k=6):
    if index is None or not query.strip():
        return []
    raw_query = query
    query = clean_query(query)
    vec, matrix = index["tfidf"]
    # 原句 / 去噪句 / 同义扩展各算一次，取最高分，互不稀释
    variants = {query, clean_query(query),
                expand_query(query), expand_query(clean_query(query))}
    sims = np.zeros(matrix.shape[0])
    for v in variants:
        if not v.strip():
            continue
        s = (matrix @ vec.transform([v]).T).toarray().ravel()
        sims = np.maximum(sims, s)

    emb = index.get("emb")
    if emb is not None:
        try:
            q_emb = np.array(list(get_embedder().embed([raw_query]))[0], dtype=np.float32)
            cos = emb @ q_emb / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q_emb) + 1e-9)
            cos = np.clip(cos, 0, None)
            tf = sims / (sims.max() + 1e-9)
            sem = cos / (cos.max() + 1e-9)
            sims = 0.6 * sem + 0.4 * tf
        except Exception:
            pass
    order = sims.argsort()[::-1]
    picked, seen = [], set()
    for idx in order:
        if sims[idx] <= 0.0:
            break
        if picked and sims[idx] < 0.015:
            break
        doc = chunks[int(idx)]
        key = (doc["source"], doc["page"])
        if key in seen:
            continue
        seen.add(key)
        picked.append({"source": doc["source"], "page": doc["page"],
                       "text": doc["text"], "score": float(sims[idx])})
        if len(picked) >= k:
            break

    # 文档多样性：避免一份文档把所有名额占满，另一份完全没机会出现
    have = {p["source"] for p in picked}
    for idx in order[:300]:
        src = chunks[int(idx)]["source"]
        if src in have or sims[idx] <= 0.01:
            continue
        doc = chunks[int(idx)]
        picked.append({"source": doc["source"], "page": doc["page"],
                       "text": doc["text"], "score": float(sims[idx])})
        have.add(src)
        if len(picked) >= k + 2:
            break
    return picked


# --------------------------------------------------------------------------
# AI 调用
# --------------------------------------------------------------------------
def get_api_key():
    if st.session_state.get("api_key"):
        return st.session_state["api_key"]
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("DEEPSEEK_API_KEY", "")
        except Exception:
            key = ""
    return key or ""


def call_ai(api_key, system, user, timeout=90):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=timeout)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=1400,
    )
    return resp.choices[0].message.content


def parse_ai_estimate(text):
    """从 AI 回复里取出 <estimate>{...}</estimate>，作为整日热量的最终数值。"""
    if not text:
        return None
    m = re.search(r"<estimate>\s*(\{.*?\})\s*</estimate>", text, re.S)
    if not m:
        m = re.search(r"\{[^{}]*\"kcal\"[^{}]*\}", text)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
        kcal = float(d.get("kcal", 0))
        p = float(d.get("protein", 0))
        c = float(d.get("carb", 0))
        f = float(d.get("fat", 0))
    except Exception:
        return None
    if not (300 <= kcal <= 6000):
        return None
    return {"kcal": kcal, "p": max(0.0, p), "c": max(0.0, c), "f": max(0.0, f)}


def strip_estimate(text):
    return re.sub(r"<estimate>.*?</estimate>", "", text or "", flags=re.S).strip()


def build_prompt(profile, meals, total, ex, sources, kb_note, training=None):
    excerpts = "\n\n".join(
        f"[{i + 1}]《{s['source']}》第 {s['page']} 页：{s['text']}"
        for i, s in enumerate(sources)
    ) or "（本次检索没有命中相关段落）"

    meal_txt = "\n".join(f"- {k}：{v or '未填写'}" for k, v in meals.items())
    plan_txt = "\n".join(f"- {r[0]}，{r[1]}，{r[2]}（{r[3]}）" for r in ex["rows"])
    bf_txt = f"{profile['body_fat']:.0f}%" if profile.get("body_fat") else "未填"
    glu_txt = f"{profile['fasting_glu']:.1f} mmol/L" if profile.get("fasting_glu") else "未填"
    sup_txt = "、".join(profile.get("supplements") or []) or "无"
    sup_detail = "；".join(
        f"{n}（{SUPP_DB[n].get('dose', '')}，{SUPP_DB[n].get('timing', '')}）"
        for n in (profile.get("supplements") or []) if n in SUPP_DB
    ) or "无"

    return f"""
你是专业营养师，请用中文、条理清晰地评估下面这一天的饮食。

【用户档案】
性别：{profile['gender']}，年龄：{profile['age']} 岁，身高 {profile['height']} cm，体重 {profile['weight']} kg，BMI {profile['bmi']:.1f}
基础代谢 BMR：{profile['bmr']:.0f} kcal；TDEE：{ex['tdee']:.0f} kcal；今日建议摄入：{ex['target']:.0f} kcal
身体状态：生理期 {profile['period']}，考试周 {profile['exam']}，备注：{profile['note'] or '无'}
睡眠：{profile.get('sleep_h', 7):.1f} 小时/天；饮水：{profile.get('water_ml', 1500)} ml/天
体脂率：{bf_txt}；空腹血糖：{glu_txt}；补剂：{sup_txt}
补剂常见用法：{sup_detail}
目标权重：减脂 {profile['weight_loss']}，增肌 {profile.get('muscle', 0)}，控糖 {profile['sugar_ctrl']}，抗炎 {profile['anti_inflammatory']}
每日蛋白质目标：{profile['protein_need']:.0f} g（{profile['protein_need'] / profile['weight']:.1f} g/kg）

【今日饮食】
{meal_txt}

【程序初步估算】（仅供参考，可指出偏差）
热量 {total['kcal']:.0f} kcal，蛋白质 {total['p']:.0f} g，碳水 {total['c']:.0f} g，脂肪 {total['f']:.0f} g，GL {total['gl']:.0f}，蔬菜 {total['veg_g']:.0f} g
未识别食材：{', '.join(total['unknown']) if total['unknown'] else '无'}

【运动情况】
运动类型：{profile['ex_type']}，每次 {profile['ex_minutes']} 分钟，每周 {profile['ex_sessions']} 次，约消耗 {ex['per_session']:.0f} kcal/次
建议计划：
{plan_txt}

【今日训练记录】
{training_summary(training) if training else '今日未记录训练'}

【知识库检索片段】
{excerpts}

【知识库状态】{kb_note}

请严格按以下结构输出（Markdown 小标题，总字数 700 字以内）：
### 1. 今日营养概况
用 2-3 句话给出整体判断，并说明距离目标还差多少。
### 2. 三餐逐项点评
每餐一句话，指出最值得改的一点。
### 3. 风险与过敏提示
只写真实存在的风险；没有就写"未发现明显风险"。
### 4. 今天还能怎么补
给 3-5 条具体到食物和克数的建议，优先利用家中常见食材。
### 5. 明日搭配思路
结合应季食材和身体状态，给出一份可直接照做的三餐示例。

规则：
- 引用文档时在句末标注【{sources[0]['source'] if sources else '指南'} p.X】，只能引用上面给出的片段；
- 不要编造文档里没有的数字或结论，不确定就写"不确定"；
- 扫描版 PDF 没能读取，不要假装引用它；
- 语气温和，不要制造焦虑。
- 如血糖或体脂率异常，只给生活方式建议并提示就医，不要做医学诊断。

最后一行必须单独输出整日估算（不要加其他文字）：
<estimate>{{"kcal": 数字, "protein": 数字, "carb": 数字, "fat": 数字}}</estimate>
请把所有菜品都算进去，包括程序没识别出来的家常菜（例如辣椒炒肉、番茄炒蛋）。
""".strip()


def local_report(profile, total, risks, ex):
    lines = []
    lines.append(f"**今日估算**：热量 {total['kcal']:.0f} kcal（目标 {ex['target']:.0f} kcal），"
                 f"蛋白质 {total['p']:.0f} g，蔬菜 {total['veg_g']:.0f} g，GL {total['gl']:.0f}。")
    gap = total["kcal"] - ex["target"]
    if abs(gap) <= 200:
        lines.append("热量基本落在目标区间，重点是提高蛋白质和蔬菜的质量。")
    elif gap > 0:
        lines.append(f"比目标多约 {gap:.0f} kcal，建议晚餐减少主食或油脂，或增加 30 分钟快走。")
    else:
        lines.append(f"比目标少约 {-gap:.0f} kcal，容易掉肌肉，建议补充一份优质蛋白（鸡蛋、豆腐、酸奶）。")
    for r in risks:
        lines.append(f"- {r}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 历史记录
# --------------------------------------------------------------------------
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_history(history):
    if IS_CLOUD:
        return True
    try:
        HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        st.warning(f"历史记录写入失败：{exc}")
        return False


def load_training():
    if TRAINING_FILE.exists():
        try:
            return json.loads(TRAINING_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_training(data):
    if IS_CLOUD:
        return True
    try:
        TRAINING_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        st.warning(f"训练记录写入失败：{exc}")
        return False


def load_profile():
    if PROFILE_FILE.exists():
        try:
            return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_profile(data):
    """记住用户上次填的档案，下次打开不用重填。"""
    if IS_CLOUD:
        return
    try:
        txt = json.dumps(data, ensure_ascii=False, indent=2)
        if not PROFILE_FILE.exists() or PROFILE_FILE.read_text(encoding="utf-8") != txt:
            PROFILE_FILE.write_text(txt, encoding="utf-8")
    except Exception:
        pass


def training_summary(record):
    """把一条训练记录拼成一句给 AI 看的描述。"""
    if not record:
        return "今日未记录训练"
    parts = []
    for name, d in record.get("exercises", {}).items():
        if d.get("weight", 0) > 0:
            parts.append(f"{name} {d['weight']:g}kg×{d['sets']}组×{d['reps']}次")
        else:
            parts.append(f"{name} {d['sets']}组×{d['reps']}次（自重）")
    return (f"{record.get('duration', 0)} 分钟，{'、'.join(parts) or '未填写动作'}，"
            f"感受：{record.get('feel', '未填')}，训练容量 {record.get('volume', 0):.0f} kg")


def last_exercise_stats(training, name, skip_date=None):
    """找这个动作最近一次的训练数据，用于"上次练了多少"的提示。"""
    for day in sorted(training.keys(), reverse=True):
        if skip_date and day == skip_date:
            continue
        info = training[day].get("exercises", {}).get(name)
        if info:
            return day, info
    return None, None


# --------------------------------------------------------------------------
# 侧边栏
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = {} if IS_CLOUD else load_history()
if "training" not in st.session_state:
    st.session_state.training = {} if IS_CLOUD else load_training()
if "saved_profile" not in st.session_state:
    st.session_state.saved_profile = {} if IS_CLOUD else load_profile()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

saved = st.session_state.saved_profile


def _pick(options, key, default):
    value = saved.get(key, default)
    return options.index(value) if value in options else options.index(default)


with st.sidebar:
    st.header("👤 用户基础档案")
    gender = st.selectbox("性别", ["女", "男"], index=_pick(["女", "男"], "gender", "女"))
    age = st.number_input("年龄", min_value=10, max_value=90, value=int(saved.get("age", 20)))
    height = st.number_input("身高（cm）", min_value=120, max_value=220, value=int(saved.get("height", 160)))
    weight = st.number_input("体重（kg）", min_value=30, max_value=200, value=int(saved.get("weight", 55)))

    bmr = 10 * weight + 6.25 * height - 5 * age - (161 if gender == "女" else 5)
    bmi = weight / ((height / 100) ** 2)
    st.info(f"🔥 BMR {bmr:.0f} kcal/天　·　BMI {bmi:.1f}")
    if bmi < 18.5:
        st.caption("⚠️ BMI 偏低，不建议继续减重，可把「减脂权重」调低。")
    elif bmi >= 28:
        st.caption("提示：BMI 偏高，减重节奏建议每周 0.5-1 kg。")

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.header("🩺 身体状态")
    if gender == "女":
        period = st.selectbox("生理期状态", ["非经期", "经期中", "经期前后"],
                              index=_pick(["非经期", "经期中", "经期前后"], "period", "非经期"))
    else:
        period = "不适用"
        st.caption("男性用户不显示生理期设置")
    exam = st.selectbox("是否考试周", ["否", "是"], index=_pick(["否", "是"], "exam", "否"))
    note = st.text_area("其他备注", height=70, value=saved.get("note", ""),
                        placeholder="例如：容易疲劳、肠胃敏感、乳糖不耐")
    with st.expander("📋 更多健康数据（可选）"):
        sleep_h = st.slider("平均睡眠（小时）", 0.0, 12.0, float(saved.get("sleep_h", 7.0)), step=0.5)
        water_ml = st.slider("每日饮水（ml）", 0, 4000, int(saved.get("water_ml", 1500)), step=100)
        body_fat = st.number_input("体脂率 %（不知道填 0）", 0.0, 60.0,
                                   float(saved.get("body_fat", 0.0)), step=1.0)
        fasting_glu = st.number_input("空腹血糖 mmol/L（不知道填 0）", 0.0, 30.0,
                                      float(saved.get("fasting_glu", 0.0)), step=0.1)
        _supp_default = [x for x in saved.get("supplements", []) if x in SUPP_DB]
        supplements = st.multiselect("正在吃的补剂", list(SUPP_DB.keys()),
                                     default=_supp_default,
                                     help="选好后，评估结果里会给出服用时间和搭配提示")

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.header("🎯 营养目标权重")
    weight_loss = st.slider("减脂", 0, 100, int(saved.get("weight_loss", 60)))
    muscle = st.slider("增肌", 0, 100, int(saved.get("muscle", 0)),
                       help="增肌目标越高，热量会留出盈余，蛋白质目标提到 1.6 g/kg")
    sugar_ctrl = st.slider("控糖", 0, 100, int(saved.get("sugar_ctrl", 50)))
    anti_inflammatory = st.slider("抗炎", 0, 100, int(saved.get("anti_inflammatory", 50)))
    if muscle >= 50 and weight_loss >= 50:
        st.caption("⚠️ 减脂和增肌同时拉高会接近维持热量，建议主攻一个。")

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.header("🏃 运动情况")
    ACTIVITY_OPTIONS = ["久坐（办公室/上课为主）", "轻度（经常走动）",
                        "中度（体力工作/每天运动）", "高度（高强度训练）"]
    activity_label = st.selectbox("日常活动量", ACTIVITY_OPTIONS,
                                  index=_pick(ACTIVITY_OPTIONS, "activity", ACTIVITY_OPTIONS[0]))
    activity_factor = dict(zip(ACTIVITY_OPTIONS, [1.2, 1.375, 1.55, 1.725]))[activity_label]
    ex_type = st.selectbox("主要运动", list(MET.keys()),
                           index=_pick(list(MET.keys()), "ex_type", "快走"))
    ex_minutes = st.slider("每次时长（分钟）", 0, 120, int(saved.get("ex_minutes", 30)), step=5)
    ex_sessions = st.slider("每周次数", 0, 7, int(saved.get("ex_sessions", 3)))

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    with st.expander("🔑 API 设置"):
        key_input = st.text_input("DeepSeek API Key", type="password",
                                  help="也可以写在 .env 的 DEEPSEEK_API_KEY 里")
        if key_input:
            st.session_state["api_key"] = key_input.strip()
        st.caption("未填写时使用本地规则评估，AI 评估会自动降级。")

profile = {
    "gender": gender, "age": age, "height": height, "weight": weight,
    "bmr": bmr, "bmi": bmi, "period": period, "exam": exam, "note": note,
    "sleep_h": sleep_h, "water_ml": water_ml, "body_fat": body_fat,
    "fasting_glu": fasting_glu, "supplements": supplements,
    "weight_loss": weight_loss, "muscle": muscle, "sugar_ctrl": sugar_ctrl,
    "anti_inflammatory": anti_inflammatory,
    "ex_type": ex_type, "ex_minutes": ex_minutes, "ex_sessions": ex_sessions,
}
save_profile({
    "gender": gender, "age": age, "height": height, "weight": weight,
    "period": period, "exam": exam, "note": note,
    "sleep_h": sleep_h, "water_ml": water_ml, "body_fat": body_fat,
    "fasting_glu": fasting_glu, "supplements": supplements,
    "weight_loss": weight_loss, "muscle": muscle, "sugar_ctrl": sugar_ctrl,
    "anti_inflammatory": anti_inflammatory, "activity": activity_label,
    "ex_type": ex_type, "ex_minutes": ex_minutes, "ex_sessions": ex_sessions,
})
profile["protein_need"] = protein_target(profile)
ex = exercise_plan(profile, activity_factor, ex_type, ex_minutes, ex_sessions)
profile["target"] = ex["target"]


# --------------------------------------------------------------------------
# 知识库
# --------------------------------------------------------------------------
sig = _pdf_signature()
with st.spinner("加载文档索引…（首次需要下载向量模型，约 1 分钟）"):
    index, chunks, files_info = build_index(sig, PREPROCESS_VERSION)


def kb_summary():
    if not files_info:
        return "未找到 pdf 文件夹或其中没有 PDF"
    ok = [f for f in files_info if f["status"] == "ok"]
    bad = [f for f in files_info if f["status"] != "ok"]
    text = f"已建立索引 {len(ok)} 份（{len(chunks)} 段）"
    if bad:
        text += f"，跳过 {len(bad)} 份"
    return text


# --------------------------------------------------------------------------
# 主界面
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <div>
        <h1>🥗 您的专属营养助手</h1>
        <p>真实文档检索 · 膳食评估 · 运动建议 · 应季食材</p>
        <div class="chips">
          <span class="chip">🎯 今日目标 {ex['target']:.0f} kcal</span>
          <span class="chip">🥩 蛋白质 {profile['protein_need']:.0f} g</span>
          <span class="chip">📅 已记录 {len(st.session_state.history)} 天</span>
        </div>
      </div>
      <div class="hero-art">🥗</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_today, tab_train, tab_trend, tab_kb = st.tabs(
    ["📊 今日评估", "🏋️ 训练", "📈 7 天曲线", "📚 知识库"])


# ---------------------------- 今日评估 ----------------------------
with tab_today:
    st.markdown("#### 🍽️ 一日三餐录入")
    PRESETS = {
        "食堂套餐": {"bf": "小笼包一笼，豆浆一杯",
                     "lunch": "米饭150g，青椒肉丝一份，炒青菜一份",
                     "dinner": "米饭100g，番茄炒蛋一份，紫菜蛋花汤一碗",
                     "snack": ""},
        "减脂日": {"bf": "燕麦40g，牛奶250ml，鸡蛋1个",
                   "lunch": "米饭100g，鸡胸肉150g，西兰花200g",
                   "dinner": "红薯150g，鱼100g，菠菜150g",
                   "snack": "苹果1个"},
        "增肌日": {"bf": "燕麦60g，牛奶250ml，鸡蛋2个",
                   "lunch": "米饭200g，牛肉150g，西兰花150g",
                   "dinner": "米饭150g，鸡胸肉150g，豆腐100g",
                   "snack": "酸奶150g，核桃10g"},
    }

    def _apply_preset(p):
        for _k, _v in p.items():
            st.session_state[f"meal_{_k}"] = _v

    st.caption("懒得打字？点一下直接填好一整天：")
    _pc = st.columns(4)
    _pc[0].button("🍱 食堂套餐", on_click=_apply_preset,
                  args=(PRESETS["食堂套餐"],), width="stretch")
    _pc[1].button("🥗 减脂日", on_click=_apply_preset,
                  args=(PRESETS["减脂日"],), width="stretch")
    _pc[2].button("💪 增肌日", on_click=_apply_preset,
                  args=(PRESETS["增肌日"],), width="stretch")
    if _pc[3].button("🧹 清空", width="stretch"):
        for _k in ("bf", "lunch", "dinner", "snack"):
            st.session_state[f"meal_{_k}"] = ""

    c1, c2, c3 = st.columns(3)
    breakfast = c1.text_area("早餐", height=90, key="meal_bf",
                             placeholder="例：燕麦40g，牛奶250ml，鸡蛋1个")
    lunch = c2.text_area("午餐", height=90, key="meal_lunch",
                         placeholder="例：米饭150g，鸡胸肉100g，西兰花150g")
    dinner = c3.text_area("晚餐", height=90, key="meal_dinner",
                          placeholder="例：红薯150g，鱼100g，菠菜100g")
    snack = st.text_input("加餐（可选）", key="meal_snack", placeholder="例：苹果1个，酸奶150g")

    with st.expander("📏 不知道多少克？常见份量速查"):
        st.markdown(
            "- 一碗米饭 ≈ 150 g，一个馒头 ≈ 100 g\n"
            "- 一个鸡蛋 ≈ 50 g，一杯牛奶 ≈ 250 ml\n"
            "- 一盒酸奶 ≈ 150 g，一个苹果 ≈ 200 g\n"
            "- 一份炒菜 ≈ 250 g（其中肉约 50-80 g）\n"
            "- 一勺油 ≈ 10 g，一把坚果 ≈ 20 g"
        )
        st.caption("写成「一碗」「一份」「一个」也可以，程序会自动换算，不用精确到克。")

    st.caption(f"🍚 已收录 {len(FOOD_INDEX)} 种食物与菜品，写「椒麻鸡一份」也能直接识别。")

    do_eval = st.button("开始评估", type="primary", width="stretch")

    if do_eval:
        meals = {"早餐": breakfast, "午餐": lunch, "晚餐": dinner, "加餐": snack}
        ests = [estimate_meal(v) for v in meals.values()]
        total = merge_estimates(ests)
        local_kcal = total["kcal"]
        query = " ".join([v for v in meals.values() if v]) + f" 减脂 控糖 蛋白质 {period} {note}"
        sources = retrieve(index, chunks, query, k=8)
        kb_note = kb_summary()
        for f in files_info:
            if f["status"] != "ok":
                kb_note += f"；{f['name']}：{f['note']}"

        today_training = st.session_state.training.get(date.today().isoformat())
        ai_text, ai_error = None, None
        api_key = get_api_key()
        if api_key:
            with st.spinner("检索指南并生成评估…"):
                try:
                    ai_text = call_ai(
                        api_key,
                        "你是专业营养学评估助手，回答要具体、克制、不制造焦虑。",
                        build_prompt(profile, meals, total, ex, sources, kb_note,
                                     training=today_training),
                    )
                except Exception as exc:
                    ai_error = str(exc)[:300]
        else:
            ai_error = "未配置 API Key"

        ai_est = parse_ai_estimate(ai_text)
        if ai_est:
            total.update(ai_est)
        risks = local_risks(total, profile)
        score = health_score(total, profile, risks)

        record = {
            "date": date.today().isoformat(),
            "kcal": round(total["kcal"]),
            "protein": round(total["p"], 1),
            "carb": round(total["c"], 1),
            "fat": round(total["f"], 1),
            "gl": round(total["gl"], 1),
            "veg": round(total["veg_g"]),
            "exercise_kcal": round(ex["daily_avg"]),
            "target": round(ex["target"]),
            "meals": meals,
            "score": score,
        }
        st.session_state.history[record["date"]] = record
        saved = save_history(st.session_state.history)

        st.session_state.last_result = {
            "meals": meals, "total": total, "risks": risks, "score": score,
            "sources": sources, "ai_text": strip_estimate(ai_text) if ai_text else None,
            "ai_error": ai_error, "saved": saved, "kb_note": kb_note,
            "ai_est": bool(ai_est), "local_kcal": local_kcal,
        }

    res = st.session_state.last_result
    if res:
        total = res["total"]
        st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
        if total["unknown"]:
            st.warning("这些内容在本地食材表里没找到，未计入本地估算：" + "、".join(total["unknown"]))
        score_color = "#2F8F5B" if res["score"] >= 80 else ("#E8A33D" if res["score"] >= 60 else "#D96B6B")
        m0, m1, m2, m3, m4 = st.columns([1.15, 1, 1, 1, 1])
        m0.markdown(
            f'<div class="ring-box">{score_ring(res["score"], score_color)}</div>',
            unsafe_allow_html=True,
        )
        metric_card(m1, "估算热量", f"{total['kcal']:.0f}",
                    f"{total['kcal'] - ex['target']:+.0f} kcal vs 目标", "#E8A33D")
        metric_card(m2, "蛋白质", f"{total['p']:.0f} g",
                    f"目标 {profile['protein_need']:.0f} g", "#4A90D9")
        metric_card(m3, "蔬菜", f"{total['veg_g']:.0f} g", "建议 300 g 以上", "#4CB47F")
        metric_card(m4, "运动消耗", f"{ex['daily_avg']:.0f}", "kcal / 天", "#8E7CC3")

        p_prot = min(100.0, total["p"] / max(profile["protein_need"], 1) * 100)
        p_veg = min(100.0, total["veg_g"] / 300 * 100)
        st.markdown(
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:14px">'
            f'<div><div class="prow"><span>蛋白质达成</span>'
            f'<span>{total["p"]:.0f} / {profile["protein_need"]:.0f} g</span></div>'
            f'<div class="pbar"><span style="width:{p_prot:.0f}%;background:#4A90D9"></span></div></div>'
            f'<div><div class="prow"><span>蔬菜达成</span>'
            f'<span>{total["veg_g"]:.0f} / 300 g</span></div>'
            f'<div class="pbar"><span style="width:{p_veg:.0f}%;background:#4CB47F"></span></div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if res.get("ai_est"):
            st.caption(f"热量与三大营养素由 AI 按菜品估算；本地食材表估算 {res['local_kcal']:.0f} kcal，仅作对照。")

        if res["saved"]:
            if IS_CLOUD:
                st.caption("✅ 已记录到本次会话（云端不写入共享文件，关闭页面后清空）")
            else:
                st.caption(f"✅ 已保存 {date.today().isoformat()} 的记录（当天重复评估会覆盖）")

        st.markdown("#### 🎯 今天还差什么")
        _gcols = st.columns(3)
        for _col, (_tag, _food, _note) in zip(_gcols, gap_advice(total, profile, ex)):
            _color = "#2F8F5B" if "达标" in _tag else "#E8A33D"
            _col.markdown(
                f'<div class="mcard" style="--accent:{_color}">'
                f'<div class="mlabel">{_tag}</div>'
                f'<div class="mval" style="font-size:1.05rem">{_food}</div>'
                f'<div class="msub">{_note}</div></div>',
                unsafe_allow_html=True,
            )

        left, right = st.columns([1.6, 1])
        with left:
            st.markdown("#### 🤖 AI 营养评估")
            if res["ai_text"]:
                st.markdown(res["ai_text"])
            else:
                st.info(f"AI 未启用（{res['ai_error']}），以下为本地规则评估。")
                st.markdown(local_report(profile, total, res["risks"], ex))
            if res["sources"]:
                with st.expander(f"📎 本次参考的原文段落（{len(res['sources'])} 段）"):
                    for s in res["sources"]:
                        st.markdown(f"**《{s['source']}》p.{s['page']}** · 相关度 {s['score']:.2f}")
                        st.caption(s["text"][:240] + ("…" if len(s["text"]) > 240 else ""))
                    st.caption(res["kb_note"])

        with right:
            st.markdown("#### ⚠️ 风险提示")
            for r in res["risks"]:
                st.markdown(f"- {r}")
            allergens = allergens_in(total)
            if allergens:
                st.markdown("**可能涉及过敏原**")
                for a in allergens:
                    st.markdown(f"- {a}")
            else:
                st.caption("未识别常见过敏原")

            if profile["supplements"]:
                st.markdown("**💊 补剂提示**")
                _tips = supplement_tips(profile["supplements"], profile)
                if _tips:
                    for _t in _tips:
                        st.markdown(f"- {_t}")
                else:
                    st.caption("所选补剂之间没有需要特别注意的搭配")
                with st.expander("补剂常见用法"):
                    for _n in profile["supplements"]:
                        _d = SUPP_DB.get(_n, {})
                        st.caption(f"{_n}：{_d.get('dose', '')}｜{_d.get('timing', '')}｜{_d.get('note', '')}")

            if total["items"]:
                with st.expander("🍚 食材估算明细"):
                    df_items = pd.DataFrame(total["items"])
                    df_items.columns = ["食材", "克数", "估算热量"]
                    st.dataframe(df_items, hide_index=True, width="stretch")
                    if total["unknown"]:
                        st.caption("未识别：" + "、".join(total["unknown"]))

        st.markdown("#### 🏃 今日运动建议")
        st.caption(f"{ex_type} 每次约消耗 {ex['per_session']:.0f} kcal；"
                   f"每周合计 {ex['weekly']:.0f} kcal，折合每天 {ex['daily_avg']:.0f} kcal。"
                   f"TDEE ≈ {ex['tdee']:.0f} kcal，建议摄入 {ex['target']:.0f} kcal。")
        plan_df = pd.DataFrame(ex["rows"], columns=["项目", "时长", "频率", "为什么"])
        st.dataframe(plan_df, hide_index=True, width="stretch")

        # 应季食材推荐
        cur_season = season_today()
        base = [f for f in FOODS if "all" in f["seasons"] or cur_season in f["seasons"]]
        if muscle >= 50:
            # 增肌优先推高蛋白食材
            picks = [f for f in base if f["anti"] >= 5]
            picks = sorted(picks, key=lambda x: (-x["p"], x["kcal"]))[:10]
            title = f"#### 💪 增肌优先食材（{SEASON_CN[cur_season]}季）"
        else:
            picks = [f for f in base if f["anti"] >= 6]
            if sugar_ctrl >= 60:
                picks = [f for f in picks if f["gl"] <= 12]
            picks = sorted(picks, key=lambda x: (-x["anti"], x["kcal"]))[:10]
            title = f"#### 🌿 当季推荐食材（{SEASON_CN[cur_season]}季）"
        st.markdown(title)
        st.markdown(" ".join(f'<span class="tag">{p["name"]}</span>' for p in picks), unsafe_allow_html=True)
        with st.expander("查看营养数据"):
            rec = pd.DataFrame([{
                "食材": p["name"], "热量/100g": p["kcal"], "蛋白质(g)": p["p"],
                "抗炎评分": p["anti"], "GL": p["gl"],
                "适合季节": "四季" if "all" in p["seasons"] else "、".join(SEASON_CN[s] for s in p["seasons"]),
            } for p in picks])
            st.dataframe(rec, hide_index=True, width="stretch")

        # 导出报告
        report = f"""您的专属营养助手 · 膳食评估报告
生成时间：{datetime.now():%Y-%m-%d %H:%M}
============================================
【用户档案】
性别 {gender}｜年龄 {age}｜身高 {height}cm｜体重 {weight}kg
BMI {bmi:.1f}｜BMR {bmr:.0f} kcal｜TDEE {ex['tdee']:.0f} kcal｜建议摄入 {ex['target']:.0f} kcal
生理期 {period}｜考试周 {exam}｜备注：{note or '无'}
睡眠 {profile['sleep_h']:.1f} 小时｜饮水 {profile['water_ml']} ml｜体脂率 {profile['body_fat']:.0f}%｜空腹血糖 {profile['fasting_glu']:.1f} mmol/L｜补剂 {'、'.join(profile['supplements']) or '无'}
目标权重：减脂 {weight_loss}｜增肌 {muscle}｜控糖 {sugar_ctrl}｜抗炎 {anti_inflammatory}
蛋白质目标：{profile['protein_need']:.0f} g/天

【补剂】
""" + ("\n".join(
            f"- {n}：{SUPP_DB[n].get('dose', '')}｜{SUPP_DB[n].get('timing', '')}｜{SUPP_DB[n].get('note', '')}"
            for n in (profile["supplements"] or []) if n in SUPP_DB) or "- 无") + "\n" + (
            "\n".join(f"- 提示：{t}" for t in supplement_tips(profile["supplements"], profile))
            if profile["supplements"] else "") + f"""

【今日饮食】
""" + "\n".join(f"{k}：{v or '未填写'}" for k, v in res["meals"].items()) + f"""

【估算】
热量 {total['kcal']:.0f} kcal｜蛋白质 {total['p']:.0f} g｜碳水 {total['c']:.0f} g
脂肪 {total['f']:.0f} g｜GL {total['gl']:.0f}｜蔬菜 {total['veg_g']:.0f} g｜健康分 {res['score']}

【风险提示】
""" + "\n".join(f"- {r}" for r in res["risks"]) + f"""

【AI 评估】
{res['ai_text'] or local_report(profile, total, res['risks'], ex)}

【运动计划】
""" + "\n".join(f"- {r[0]}｜{r[1]}｜{r[2]}｜{r[3]}" for r in ex["rows"])
        report += f"""

【今日训练记录】
{training_summary(st.session_state.training.get(date.today().isoformat()))}
"""
        st.download_button("📄 下载评估报告", data=report,
                           file_name=f"营养评估_{date.today()}.txt", mime="text/plain")


# ---------------------------- 训练记录 ----------------------------
with tab_train:
    st.markdown("#### 🏋️ 今日训练记录")
    c1, c2, c3 = st.columns(3)
    t_date = c1.date_input("训练日期", value=date.today())
    t_duration = c2.slider("训练时长（分钟）", 0, 180, 45, step=5)
    t_feel = c3.selectbox("训练感受", ["轻松", "适中", "力竭"], index=1)

    picked = st.multiselect(
        "今天练了哪些动作", EXERCISES,
        default=[] if muscle < 50 else ["深蹲", "卧推", "划船"],
    )
    details = {}
    if picked:
        st.caption("重量填 0 表示自重动作（俯卧撑、引体向上等）")
        for name in picked:
            prev_day, prev = last_exercise_stats(st.session_state.training, name,
                                                 skip_date=t_date.isoformat())
            cw, cs, cr = st.columns([2, 1, 1])
            w = cw.number_input(
                f"{name}｜重量 (kg)", 0.0, 400.0,
                float(prev["weight"]) if prev else 0.0, step=2.5, key=f"w_{name}")
            s = cs.number_input(
                f"{name}｜组数", 1, 15,
                int(prev["sets"]) if prev else 3, key=f"s_{name}")
            r = cr.number_input(
                f"{name}｜次数", 1, 50,
                int(prev["reps"]) if prev else 10, key=f"r_{name}")
            details[name] = {"weight": float(w), "sets": int(s), "reps": int(r)}
            if prev:
                tip = f"↺ 上次（{prev_day[5:]}）：{prev['weight']:g}kg × {prev['sets']} 组 × {prev['reps']} 次"
                if prev["weight"] > 0:
                    tip += f"　→ 如果上次最后一组还有余力，这次可以试 {prev['weight'] + 2.5:g} kg"
                st.caption(tip)

    volume = sum(d["weight"] * d["sets"] * d["reps"] for d in details.values())
    st.caption(f"训练容量：{volume:.0f} kg（重量 × 组数 × 次数）")

    if st.button("保存训练记录", type="primary"):
        if not picked:
            st.warning("先选至少一个动作")
        else:
            rec = {"duration": int(t_duration), "feel": t_feel,
                   "exercises": details, "volume": round(volume, 1)}
            st.session_state.training[t_date.isoformat()] = rec
            if save_training(st.session_state.training):
                st.success(f"已保存 {t_date} 的训练：{len(details)} 个动作，容量 {volume:.0f} kg")

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    train = st.session_state.training
    if not train:
        st.info("还没有训练记录。填一次就会出现在这里。")
    else:
        rows = sorted(train.items(), key=lambda kv: kv[0])[-7:]
        df_train = pd.DataFrame([{
            "date": k,
            "volume": v.get("volume", 0),
            "duration": v.get("duration", 0),
            "actions": len(v.get("exercises", {})),
        } for k, v in rows])
        end_t = datetime.now().date()
        grid_t = pd.DataFrame({"date": [(end_t - timedelta(days=i)).isoformat()
                                        for i in range(6, -1, -1)]})
        df_train = grid_t.merge(df_train, on="date", how="left").fillna(
            {"volume": 0, "duration": 0, "actions": 0})
        df_train["date"] = pd.to_datetime(df_train["date"])

        st.markdown("#### 📊 最近 7 天训练")
        figv = go.Figure(go.Bar(x=df_train["date"], y=df_train["volume"],
                                marker_color="#7fb3d5", name="训练容量"))
        figv.update_layout(height=280, title="训练容量（kg）",
                           margin=dict(l=10, r=10, t=40, b=10))
        figv.update_xaxes(type="date", tickformat="%m-%d")
        st.plotly_chart(figv, width="stretch", config=PLOTLY_CONFIG)

        c1, c2 = st.columns(2)
        with c1:
            figd = go.Figure(go.Bar(x=df_train["date"], y=df_train["duration"],
                                    marker_color="#a8d5ba"))
            figd.update_layout(height=240, title="训练时长（分钟）",
                               margin=dict(l=10, r=10, t=40, b=10))
            figd.update_xaxes(type="date", tickformat="%m-%d")
            st.plotly_chart(figd, width="stretch", config=PLOTLY_CONFIG)
        with c2:
            figa = go.Figure(go.Bar(x=df_train["date"], y=df_train["actions"],
                                    marker_color="#f0c987"))
            figa.update_layout(height=240, title="动作数量",
                               margin=dict(l=10, r=10, t=40, b=10))
            figa.update_xaxes(type="date", tickformat="%m-%d")
            st.plotly_chart(figa, width="stretch", config=PLOTLY_CONFIG)

        st.caption(f"最近 7 天记录 {len(rows)} 次训练，合计容量 {df_train['volume'].sum():.0f} kg。")
        show_train = df_train.copy()
        show_train["date"] = show_train["date"].dt.strftime("%Y-%m-%d")
        show_train = show_train.rename(columns={
            "date": "日期", "volume": "容量(kg)", "duration": "时长(分钟)", "actions": "动作数"})
        st.dataframe(show_train, hide_index=True, width="stretch")


# ---------------------------- 7 天曲线 ----------------------------
with tab_trend:
    if IS_CLOUD:
        st.caption("云端为多人共用，记录只保存在本次浏览会话，不会和别人的数据混在一起。")
    hist = st.session_state.history
    if not hist:
        st.info("还没有记录。回到「今日评估」完成一次评估后，这里会自动生成曲线。")
    else:
        rows = sorted(hist.values(), key=lambda r: r["date"])[-7:]
        df = pd.DataFrame(rows)
        # 补齐最近 7 天，横轴连续；没记录的那天留空，避免折线连成假趋势
        end = datetime.now().date()
        grid = pd.DataFrame({"date": [(end - timedelta(days=i)).isoformat()
                                      for i in range(6, -1, -1)]})
        df = grid.merge(df, on="date", how="left")
        df["date"] = pd.to_datetime(df["date"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["kcal"], mode="lines+markers",
                                 name="摄入热量", connectgaps=False,
                                 line=dict(color="#2f8f5b", width=3)))
        fig.add_trace(go.Scatter(x=df["date"], y=df["target"], mode="lines",
                                 name="目标摄入", connectgaps=False,
                                 line=dict(color="#c9a227", dash="dash")))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_xaxes(type="date", tickformat="%m-%d")
        st.markdown("#### 热量摄入 vs 目标")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

        c1, c2 = st.columns(2)
        with c1:
            fig2 = go.Figure(go.Bar(x=df["date"], y=df["protein"], marker_color="#8fc9a8"))
            fig2.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                               title="蛋白质（g）")
            fig2.update_xaxes(type="date", tickformat="%m-%d")
            st.plotly_chart(fig2, width="stretch", config=PLOTLY_CONFIG)
        with c2:
            fig3 = go.Figure(go.Bar(x=df["date"], y=df["exercise_kcal"], marker_color="#f0c987"))
            fig3.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                               title="运动消耗（kcal/天，估算）")
            fig3.update_xaxes(type="date", tickformat="%m-%d")
            st.plotly_chart(fig3, width="stretch", config=PLOTLY_CONFIG)

        recorded = df.dropna(subset=["kcal"]).copy()
        recorded["date"] = recorded["date"].dt.strftime("%Y-%m-%d")
        show = recorded[["date", "kcal", "target", "protein", "gl", "veg",
                         "exercise_kcal", "score"]]
        show.columns = ["日期", "热量", "目标", "蛋白质", "GL", "蔬菜(g)", "运动消耗", "健康分"]
        st.dataframe(show, hide_index=True, width="stretch")

        if st.button("清空历史记录"):
            st.session_state.history = {}
            save_history({})
            st.rerun()


# ---------------------------- 知识库 ----------------------------
with tab_kb:
    st.markdown("#### 📚 文档索引状态")
    uploaded = st.file_uploader("添加参考文档（PDF）", type=["pdf"], accept_multiple_files=True,
                                help="上传后会保存到项目下的 pdf 文件夹，并自动重建索引")
    if uploaded:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for uf in uploaded:
            try:
                target = PDF_DIR / uf.name
                target.write_bytes(uf.getbuffer())
                saved.append(uf.name)
            except Exception as exc:
                st.error(f"{uf.name} 保存失败：{exc}")
        if saved:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    if not files_info:
        st.warning(f"没有找到 PDF。请把文档放进：`{PDF_DIR}`")
    else:
        for f in files_info:
            if f["status"] == "ok":
                st.success(f"✅ {f['name']}｜{f['pages']} 页｜{f['chars']:,} 字｜{f['chunks']} 段")
            elif f["status"] == "scanned":
                st.warning(f"⚠️ {f['name']}｜{f['pages']} 页｜{f['note']}")
            else:
                st.error(f"❌ {f['name']}｜{f['note']}")

    scanned = [f for f in files_info if f["status"] == "scanned"]
    if scanned:
        st.info(
            "扫描版 PDF 需要先做 OCR 才能检索。两种做法：\n"
            "1. 找一份带文字层的电子版替换它；\n"
            "2. 安装 Tesseract 后再接入 OCR（改动量较大，之后可以帮你做）。"
        )

    st.markdown("#### 🔍 检索测试")
    q = st.text_input("输入一个问题，看看会命中哪些原文", value="减重期间蛋白质吃多少")
    if q and index is not None:
        hits = retrieve(index, chunks, q, k=4)
        if not hits:
            st.caption("没有命中，换个说法或补充关键词试试。")
        for h in hits:
            with st.expander(f"《{h['source']}》p.{h['page']}｜相关度 {h['score']:.2f}"):
                st.write(h["text"][:600])

    with st.expander("ℹ️ 这个页面是怎么工作的"):
        st.markdown(
            "1. 启动时读取 `pdf` 文件夹，逐页提取文字；\n"
            "2. 把每页切成约 700 字的片段；\n"
            "3. 用本地向量模型 + 关键词混合打分，检索最相关的 8 段；\n"
            "4. 只把这 8 段发给 AI，并让它标注页码。\n\n"
            "这样既不会超出模型上下文，也能让回答有据可查。"
        )

st.markdown(
    '<div class="footer">本工具仅提供饮食参考，不能替代医生或注册营养师的诊断与治疗。</div>',
    unsafe_allow_html=True,
)
