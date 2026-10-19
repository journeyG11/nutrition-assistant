# 🥗 您的专属营养助手

一个基于 Streamlit 的饮食评估工具：录入三餐，自动估算热量与营养素，检索膳食指南原文，
给出分板块的改进建议，并记录 7 天趋势与训练情况。

## 功能

- **三餐评估**：解析「米饭100g」「辣椒炒肉一份」这类写法，估算热量、蛋白质、蔬菜量、GL
- **真实检索**：PDF 切段 + 本地向量模型（bge-small-zh-v1.5）+ TF-IDF 混合打分，回答带页码引用
- **运动与增肌**：活动系数、MET 消耗、TDEE、减脂/增肌/控糖/抗炎四种目标权重
- **训练记录**：动作、重量、组数、次数，自动算容量，提示上次重量
- **7 天曲线**：热量、蛋白质、运动消耗、训练容量
- **报告导出**：一键下载 txt 报告

## 本地运行

```bash
cd C:\Users\ASUS
streamlit run nutrition_new.py
```

首次使用向量检索需要安装依赖（约 1 分钟）：

```bash
C:\Users\ASUS\anaconda3\python.exe -m pip install fastembed
```

没有安装时程序不会报错，会自动退回关键词检索。

## 配置 API Key

三种方式任选其一：

1. 侧边栏「🔑 API 设置」直接粘贴
2. 项目目录下建 `.env`，写入 `DEEPSEEK_API_KEY=sk-xxx`
3. 环境变量 `DEEPSEEK_API_KEY`

## 目录

```
nutrition_new.py       主程序
pdf/                   参考文档（PDF）
eval_retrieval.py      检索评测脚本
检索评测报告.md        评测结果
nutrition_history.json 饮食记录（自动生成）
training_history.json  训练记录（自动生成）
```

## 检索评测

```bash
python eval_retrieval.py
```

当前结果：Hit@1 10/20，Hit@8 19/20（95%），MRR 0.675。
