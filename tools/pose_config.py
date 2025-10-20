"""
姿勢估計器配置系統
自動選擇 MediaPipe 或 YOLOv8
"""

import sys

# 檢測 MediaPipe 是否可用
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_VERSION = mp.__version__
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_VERSION = None

# 檢測 Ultralytics 是否可用
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_pose_extractor(prefer_mediapipe=True):
    """
    自動選擇並返回可用的姿勢提取器

    參數:
        prefer_mediapipe: 是否優先使用 MediaPipe（預設 True）

    返回:
        PoseExtractor 實例
    """

    if prefer_mediapipe and MEDIAPIPE_AVAILABLE:
        from pose_extractor_mediapipe import PoseExtractorMediaPipe
        print(f"✓ 使用 MediaPipe {MEDIAPIPE_VERSION}（Python {PYTHON_VERSION}）")
        print("  - 完整 33 個關節點")
        print("  - 3D 座標 (x, y, z)")
        print("  - 準確率最高\n")
        return PoseExtractorMediaPipe()

    elif YOLO_AVAILABLE:
        from pose_extractor import PoseExtractor
        print(f"✓ 使用 YOLOv8 Pose（Python {PYTHON_VERSION}）")
        print("  - 17 個 COCO 關節點")
        print("  - 2D 座標 (x, y)")
        print("  - 速度較快但準確率較低")
        print("  ⚠️  警告: 與訓練模型的特徵不完全匹配\n")
        return PoseExtractor()

    else:
        raise ImportError(
            "找不到可用的姿勢估計套件！\n\n"
            "請選擇以下其中一種安裝方式:\n\n"
            "方案 1: MediaPipe（推薦）\n"
            "  - 使用 Python 3.11: pyenv install 3.11.9\n"
            "  - 安裝: pip install mediapipe>=0.10.0\n\n"
            "方案 2: YOLOv8\n"
            "  - 安裝: pip install ultralytics>=8.0.0\n"
        )


def get_system_info():
    """獲取系統配置資訊"""
    info = {
        "python_version": PYTHON_VERSION,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "mediapipe_version": MEDIAPIPE_VERSION,
        "yolo_available": YOLO_AVAILABLE,
    }
    return info


def print_system_info():
    """顯示系統配置資訊"""
    info = get_system_info()

    print("=== 系統配置 ===")
    print(f"Python 版本: {info['python_version']}")
    print(f"MediaPipe: {'✓ ' + info['mediapipe_version'] if info['mediapipe_available'] else '✗ 未安裝'}")
    print(f"YOLOv8: {'✓ 可用' if info['yolo_available'] else '✗ 未安裝'}")
    print()

    if info['mediapipe_available']:
        print("推薦: 使用 MediaPipe 以獲得最佳準確率")
    elif info['yolo_available']:
        print("當前: 使用 YOLOv8（準確率可能較低）")
        print("建議: 安裝 Python 3.11 + MediaPipe 以提升準確率")
    else:
        print("錯誤: 未安裝任何姿勢估計套件")

    print()


if __name__ == "__main__":
    print_system_info()

    try:
        extractor = get_pose_extractor(prefer_mediapipe=True)
        print("✓ 姿勢提取器初始化成功")
    except ImportError as e:
        print(f"✗ 初始化失敗: {e}")
