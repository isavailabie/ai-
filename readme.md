
# weather-visualizer

本项目是一个天气数据预测及分类网站

---

## 🌱 分支说明

- `main` 分支：包含网页展示功能（`app.py`、HTML 模板、静态文件等）
- `model` 分支：包含模型训练与数据处理代码（如 `classification/`、`patchtst/`、`timesnet/` 等目录）


---

## 📁 项目目录结构（main 分支）

```
weather-visualizer/
├── app.py               # 项目主入口，启动 Flask 网站
├── requirements.txt     # 依赖列表，可用 pip 安装
├── readme.md            # 本说明文件
├── data/                # 存放天气数据
├── model/               # 存放模型权重文件（如 .pth）
├── static/              # 静态资源，如 CSS、图像等
└── templates/           # 网页 HTML 模板文件（Jinja2）
```

---

##  快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

---

### 2. 启动网站服务

```bash
python app.py
```

---

### 3. 打开浏览器访问

```
http://127.0.0.1:5000
```

---

## 📦 模型训练代码说明（在 model 分支）

`model` 分支中包含以下模块：

```
classification/     # 分类模型训练
patchtst/           # PatchTST 时间序列模型
timesnet/           # TimesNet 时间序列预测模型
tramsformer/        # Transformer 时间序列模型
```

每个目录中都包含相关的数据加载、模型定义和训练脚本，训练入口为 `main.py`，预测为`predict.py`或者`yuce.py`。

---


