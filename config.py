#该类用于配置模型
from pathlib import Path
import sys

# 获取当前文件的绝对路径
file_path = Path(__file__).resolve()

# 获取当前文件的父目录
root_path = file_path.parent

# 将根目录路径添加到 sys.path 中
if root_path not in sys.path:
    sys.path.append(str(root_path))

# 获取根目录相对于当前工作目录的相对路径
ROOT = root_path.relative_to(Path.cwd())

# 配置源列表
SOURCES_LIST = ["图片", "视频" , "摄像头"]

# 深度学习模型配置
Classification_MODEL_DIR = ROOT / 'weights' / 'detection'
# YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
# YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
# YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
# YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
# YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"
best_train = Classification_MODEL_DIR / "fruit81_best.pt"

Classification_MODEL_LIST = [
    'fruit81_best.pt',
    # "best_train.pt",
    # "yolov8n.pt",
    # "yolov8s.pt",
    # "yolov8m.pt",
    # "yolov8l.pt",
    # "yolov8x.pt"
]
FONT_DIR=ROOT/'font'/'SimHei.ttf'