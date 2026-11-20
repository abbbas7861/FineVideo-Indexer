#!/usr/bin/env python3
import argparse
import io
import json
import math
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from ultralytics import YOLO

CLASS_MAP = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
CONF = 0.4
IOU = 0.7
IMGSZ = 640
BATCH_SIZE = 32
NUM_DECODERS = 4
FRAME_SKIP = 1          

def get_rotation(stream) -> int:
    rot = stream.metadata.get("rotate")
    if rot is not None:
        try:
            return int(rot)
        except ValueError:
            pass
    return 0

def apply_rotation(img, rotation: int):
    if rotation == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

class VideoTask:
    __slots__ = (
        "video_id", "parquet_name", "row_idx", "video_bytes",
        "enqueued", "processed", "done_decoding", "error",
        "second_data", "lock",
    )
    def __init__(self, video_id, parquet_name, row_idx, video_bytes):
        self.video_id = video_id
        self.parquet_name = parquet_name
        self.row_idx = row_idx
        self.video_bytes = video_bytes
        self.enqueued = 0
        self.processed = 0
        self.done_decoding = False
        self.error = None
        self.second_data = defaultdict(list)   
        self.lock = threading.Lock()

def decoder_worker(task_queue, frame_queue, task_registry, registry_lock):
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            container = av.open(io.BytesIO(task.video_bytes), format="mp4")
            stream = container.streams.video[0]
            rotation = get_rotation(stream)
            first_ts = None
            raw_idx = 0

            for frame in container.decode(stream):
                if FRAME_SKIP > 1 and raw_idx % FRAME_SKIP != 0:
                    raw_idx += 1
                    continue
                raw_idx += 1

                img = frame.to_ndarray(format="bgr24")
                if rotation:
                    img = apply_rotation(img, rotation)

                ts = float(frame.pts * frame.time_base)
                if first_ts is None:
                    first_ts = ts
                sec = int(ts - first_ts)

                with task.lock:
                    task.enqueued += 1
                frame_queue.put((task.video_id, sec, img))
        except Exception as e:
            task.error = str(e)[:200]
        finally:
            with task.lock:
                task.done_decoding = True
            task_queue.task_done()

def process_batch(frames, metas, model, task_registry):
    results = model(frames, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False, device=0)
    for res, (video_id, sec) in zip(results, metas):
        objs = set()
        people = 0
        if res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                if cls_id in CLASS_MAP:
                    name = CLASS_MAP[cls_id]
                    objs.add(name)
                    if name == "person":
                        people += 1
        task = task_registry[video_id]
        with task.lock:
            task.second_data[sec].append((objs, people))
            task.processed += 1

def finalize_completed_tasks(task_registry, registry_lock, output_file):
    to_finalize = []
    with registry_lock:
        for vid, task in list(task_registry.items()):
            with task.lock:
                if task.done_decoding and task.processed == task.enqueued:
                    to_finalize.append(task)
        for task in to_finalize:
            del task_registry[task.video_id]

    for task in to_finalize:
        if task.error:
            output_file.write(json.dumps({"video": f"{task.parquet_name}:{task.row_idx}", "error": task.error}) + "\n")
        for sec in sorted(task.second_data.keys()):
            detections = task.second_data[sec]
            total_frames = len(detections)
            if total_frames == 0:
                continue
            obj_counts = defaultdict(int)
            people_counts = []
            for objs, pc in detections:
                for o in objs:
                    obj_counts[o] += 1
                people_counts.append(pc)
            objects = sorted(o for o, c in obj_counts.items() if c >= total_frames / 2)
            people = int(np.median(people_counts))
            output_file.write(json.dumps({
                "video": f"{task.parquet_name}:{task.row_idx}",
                "second": sec,
                "objects": objects,
                "people": people,
            }) + "\n")
        output_file.flush()

def inference_worker(frame_queue, task_registry, registry_lock, model, output_file):
    batch_frames = []
    batch_meta = []
    while True:
        try:
            item = frame_queue.get(timeout=0.5)
        except queue.Empty:
            finalize_completed_tasks(task_registry, registry_lock, output_file)
            with registry_lock:
                if not task_registry:
                    break
            continue
        if item is None:
            break
        video_id, sec, img = item
        batch_frames.append(img)
        batch_meta.append((video_id, sec))
        if len(batch_frames) >= BATCH_SIZE:
            process_batch(batch_frames, batch_meta, model, task_registry)
            batch_frames = []
            batch_meta = []
    if batch_frames:
        process_batch(batch_frames, batch_meta, model, task_registry)
    finalize_completed_tasks(task_registry, registry_lock, output_file)

def producer(parquet_files, task_queue, task_registry, registry_lock, output_file):
    video_id_counter = 0
    for pf_path in parquet_files:
        try:
            pf = pq.ParquetFile(pf_path)
            row_idx = 0
            for batch in pf.iter_batches(batch_size=8, columns=["mp4"]):
                mp4_col = batch.column("mp4")
                for i in range(len(mp4_col)):
                    video_bytes = mp4_col[i].as_py()
                    if video_bytes is None:
                        output_file.write(json.dumps({"video": f"{pf_path.name}:{row_idx}", "error": "missing mp4 column"}) + "\n")
                        row_idx += 1
                        continue
                    with registry_lock:
                        vid = video_id_counter
                        video_id_counter += 1
                        task = VideoTask(vid, pf_path.name, row_idx, video_bytes)
                        task_registry[vid] = task
                    task_queue.put(task)
                    row_idx += 1
        except Exception as e:
            output_file.write(json.dumps({"video": f"{pf_path.name}:0", "error": f"parquet read failed: {str(e)[:100]}"}) + "\n")
    for _ in range(NUM_DECODERS):
        task_queue.put(None)

def prepare_model():
    print("Loading yolo11l.pt …")
    model = YOLO("yolo11l.pt")
    print("Exporting TensorRT engine …")
    model.export(format="engine", imgsz=IMGSZ, half=True, batch=BATCH_SIZE, device=0, workspace=4, dynamic=True, nms=False)
    print("Engine exported → yolo11l.engine")

def main():
    global FRAME_SKIP
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Folder containing .parquet files")
    parser.add_argument("--output", default="results.jsonl", help="Output JSONL")
    parser.add_argument("--prepare", action="store_true", help="Export TensorRT engine")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()
    FRAME_SKIP = max(1, args.frame_skip)
    if args.prepare:
        prepare_model()
        return
    if not args.input:
        print("--input is required unless --prepare is used.")
        sys.exit(1)
    engine_path = Path("yolo11l.engine")
    pt_path = Path("yolo11l.pt")
    if engine_path.exists():
        print(f"Loading engine: {engine_path}")
        model = YOLO(str(engine_path))
    elif pt_path.exists():
        print(f"Engine not found, falling back to PyTorch: {pt_path}")
        model = YOLO(str(pt_path))
    else:
        print("No model found. Run `python app.py --prepare` first.")
        sys.exit(1)
    input_dir = Path(args.input)
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found in {input_dir}")
        sys.exit(1)
    print(f"Found {len(parquet_files)} Parquet files.")
    output_file = open(args.output, "w", buffering=1)   
    task_registry = {}
    registry_lock = threading.Lock()
    task_queue = queue.Queue(maxsize=16)      
    frame_queue = queue.Queue(maxsize=512)    
    producer_thread = threading.Thread(target=producer, args=(parquet_files, task_queue, task_registry, registry_lock, output_file), daemon=True)
    producer_thread.start()
    decoder_threads = []
    for _ in range(NUM_DECODERS):
        t = threading.Thread(target=decoder_worker, args=(task_queue, frame_queue, task_registry, registry_lock), daemon=True)
        t.start()
        decoder_threads.append(t)
    inference_thread = threading.Thread(target=inference_worker, args=(frame_queue, task_registry, registry_lock, model, output_file), daemon=True)
    inference_thread.start()
    producer_thread.join()
    for t in decoder_threads:
        t.join()
    frame_queue.put(None)
    inference_thread.join()
    output_file.close()
    print(f"Done. Results written to {args.output}")

if __name__ == "__main__":
    main()
