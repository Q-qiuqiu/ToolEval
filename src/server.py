from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from ultralytics import YOLO
from fastmcp import FastMCP



# ========== 配置 ==========
MODELS_DIR = Path("/home/yzx/models_weight/YOLO/")  # 你的 .pt 文件目录（可改）
SERVER_NAME = "YOLO MCP Server"

mcp = FastMCP(SERVER_NAME)

def _sanitize_tool_suffix(name: str) -> str:
    """
    把文件名转换成更适合作为 tool name 的后缀（避免空格等）。
    """
    safe = []
    for ch in name:
        if ch.isalnum():
            safe.append(ch)
        else:
            safe.append("_")
    s = "".join(safe).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.lower() or "model"


def _results_to_dict(results, class_names: Dict[int, str]) -> Dict[str, Any]:
    """
    把 Ultralytics YOLO 推理结果转成 JSON 友好的结构
    """
    r0 = results[0]
    boxes = r0.boxes

    dets: List[Dict[str, Any]] = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            c = int(cls[i])
            dets.append(
                {
                    "class_id": c,
                    "class_name": class_names.get(c, str(c)),
                    "confidence": float(conf[i]),
                    "bbox_xyxy": [float(x) for x in xyxy[i].tolist()],
                }
            )

    return {
        "num_detections": len(dets),
        "detections": dets,
    }


def _make_detect_tool(model_name: str, model: YOLO):
    """
    为某个 YOLO 模型创建一个 tool 函数，并返回该函数。
    """

    def detect(
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        用指定 YOLOv8 模型对本地图片做目标检测。

        参数:
          - image_path: 本地图片路径
          - conf: 置信度阈值
          - iou: NMS IOU 阈值
          - imgsz: 推理尺寸
          - max_det: 最大检测数量
          - device: 推理设备（例如 "cpu", "0" 表示 GPU0）；None 表示自动
          - classes: 只保留这些类别 id（可选）

        返回:
          - JSON：包含 detections（bbox/类别/置信度）
        """
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"image_path not found: {image_path}")

        # 读图（你也可以直接传 path 给 model.predict；这里显式读图便于早发现坏图）
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"failed to read image: {image_path}")

        predict_kwargs = dict(
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
        )
        if device is not None:
            predict_kwargs["device"] = device
        if classes is not None:
            predict_kwargs["classes"] = classes

        results = model.predict(source=img, **predict_kwargs)
        # Ultralytics: model.names 通常是 dict[int,str]
        class_names = getattr(model, "names", {})
        print(f"[Model={model_name}] detected {_results_to_dict(results, class_names)}")
        return {
            "model": model_name,
            "image_path": str(p),
            "result": _results_to_dict(results, class_names),
        }

    return detect


def register_all_models(models_dir: Path) -> None:
    """
    扫描目录下所有 .pt，分别注册成不同的 MCP tools。
    """
    if not models_dir.exists():
        raise FileNotFoundError(f"models_dir not found: {models_dir.resolve()}")

    pt_files = sorted(models_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"no .pt files found in: {models_dir.resolve()}")

    for pt in pt_files:
        model_stem = pt.stem
        suffix = _sanitize_tool_suffix(model_stem)

        tool_name = f"detect_{suffix}"
        tool_desc = f"YOLO detection using model '{model_stem}' loaded from {pt.name}"

        yolo = YOLO(str(pt))  # 加载模型
        fn = _make_detect_tool(model_stem, yolo)

        # 关键：用带参数的 tool 装饰器“动态注册”函数
        # 官方文档说明 @mcp.tool(...) 可覆盖 name/description 等元数据
        mcp.tool(name=tool_name, description=tool_desc)(fn)

        print(f"[OK] registered tool: {tool_name}  (model={pt.name})")


if __name__ == "__main__":
    register_all_models(MODELS_DIR)

    # 默认是 STDIO 方式运行；也可改成 http/sse（看你客户端怎么连）
    # FastMCP 的 run 示例见官方 README / 文档
    mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=8000,
    )
