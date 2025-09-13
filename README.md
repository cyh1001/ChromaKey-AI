# ChromaKey AI

这是一个利用Snapdragon® X Elite处理器端侧AI计算能力，实现实时视频背景替换的工具。项目旨在提供一个无需绿幕、零延迟、保护隐私的个人虚拟演播室体验。

## 项目背景与愿景
在数字内容创作时代，解决传统绿幕方案成本高昂、繁琐，以及云端AI抠图服务隐私泄露、高延迟、网络依赖等痛点。ChromaKey AI致力于在本地设备上运行，提供智能、安全、高效的实时视频背景处理。

## 安装与设置 (Installation & Setup)

请按照以下步骤设置您的开发环境并运行项目：

### 1. 克隆仓库 (Clone the Repository)
```bash
git clone https://github.com/your-username/ChromaKey-AI.git # 请替换为您的实际仓库地址
cd ChromaKey-AI
```

### 2. 创建并激活Python虚拟环境 (Create and Activate Python Virtual Environment)
强烈建议使用虚拟环境来管理项目依赖。

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖 (Install Dependencies)
我们推荐使用 `uv` 进行更快的安装。如果您尚未安装 `uv`，可以先通过 `pip install uv` 安装它。

#### 使用 `uv` (推荐)
```bash
uv pip install -r requirements.txt
```

#### 使用 `pip`
```bash
pip install -r requirements.txt
```

### 4. 下载AI模型 (Download the AI Model)
AI模型文件（约466MB）未包含在Git仓库中。请使用以下命令下载模型，并将其保存为 `model.onnx` 在项目根目录中。

```bash
curl -L https://huggingface.co/onnx-community/BiRefNet-portrait-ONNX/resolve/main/onnx/model_fp16.onnx -o model.onnx
```

## 使用方法 (Usage)

在完成上述安装步骤后，您可以通过运行主应用程序脚本来启动项目。

```bash
# 确保您已激活虚拟环境
python main_app.py # 替换为您的主应用程序脚本名称
```

## 核心技术栈 (Core Tech Stack)
*   **Python:** 主要开发语言。
*   **ONNX Runtime:** 用于加载 `.onnx` 模型并在 Snapdragon NPU 上进行硬件加速推理。
*   **OpenCV (`opencv-python`):** 负责视频流处理、图像预处理、画面合成与显示。
*   **NumPy:** 用于高效的图像数组运算。
*   **uv / pip:** 包管理工具。

## 许可证 (License)
本项目采用 MIT 许可证。
