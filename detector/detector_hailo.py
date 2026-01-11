import os
import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from threading import Thread
from queue import Queue
import paho.mqtt.client as mqtt
import json

# ==============================================================================
# Compatibility Imports (Reused from detector.py)
# ==============================================================================
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

# Hailo Integration
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                                InputVStreamParams, OutputVStreamParams, FormatType)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False

# ==============================================================================
# Configuration
# ==============================================================================
# Hailo Inference
HEF_PATH = Path("models/detection.hef") # Path to HEF model              

# ==============================================================================
# Reused Utilities (StreamLoader, AsyncVideoWriter)
# ==============================================================================
# NOTE: In a real refactor, these should be in a common 'utils.py'
# For this task, I will provide the full implementation in this target file.

class AsyncVideoWriter:
    def __init__(self, path, fourcc, fps, dims, queue_size=120):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.dims = dims
        self.queue = Queue(maxsize=queue_size)
        self.writer = cv2.VideoWriter(path, fourcc, fps, dims)
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def write(self, frame):
        if not self.writer.isOpened(): return
        if self.queue.full(): return 
        self.queue.put(frame)

    def update(self):
        while not self.stopped:
            if not self.queue.empty():
                frame = self.queue.get()
                self.writer.write(frame)
                self.queue.task_done()
            else:
                time.sleep(0.01)
        while not self.queue.empty():
            frame = self.queue.get()
            self.writer.write(frame)
            self.queue.task_done()
        self.writer.release()

    def release(self):
        self.stopped = True
        self.thread.join()

class StreamLoader:
    def __init__(self, source, width, height):
        self.source = source
        self.width = width
        self.height = height
        self.stopped = False
        self.grabbed = False
        self.frame = None
        self.source_type = "unknown"
        self.cap = None

        if source.startswith("usb"):
            self.source_type = "usb"
            idx = int(source.replace("usb", ""))
            self.cap = cv2.VideoCapture(idx)
        elif source.startswith("picamera"):
            if not PICAM_AVAILABLE:
                print("❌ Picamera2 library not installed.")
                sys.exit(1)
            self.source_type = "picamera"
            self.cap = Picamera2()
            self.config_picamera()
            self.cap.start()
        elif os.path.isfile(source):
            self.source_type = "video"
            self.cap = cv2.VideoCapture(source)
        else:
            raise ValueError(f"Unknown source: {source}")

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def config_picamera(self):
        config = self.cap.create_video_configuration(
            main={"format": "XRGB8888", "size": (self.width, self.height)},
            controls={"FrameRate": 30}
        )
        self.cap.configure(config)

    def update(self):
        while not self.stopped:
            if self.source_type in ["usb", "video"]:
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.frame = frame
                self.grabbed = True
            elif self.source_type == "picamera":
                frame = self.cap.capture_array()
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self.grabbed = True
            if self.source_type == "video":
                time.sleep(0.01) 

    def read(self):
        return self.grabbed, self.frame

    def release(self):
        self.stopped = True
        self.thread.join()
        if self.source_type in ["usb", "video"]:
            self.cap.release()
        elif self.source_type == "picamera":
            self.cap.stop()

# ==============================================================================
# Post-Processing: NMS & Centroid Tracker
# ==============================================================================
class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = {} # {id: (centroid, class_id)}
        self.disappeared = {} # {id: count}
        self.max_disappeared = max_disappeared

    def register(self, centroid, class_id):
        self.objects[self.next_id] = (centroid, class_id)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects, class_ids):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = []
        for (startX, startY, endX, endY) in rects:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], class_ids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [obj[0] for obj in self.objects.values()]

            # Simple Euclidean distance for matching
            from scipy.spatial import distance as dist
            D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = (input_centroids[col], class_ids[col])
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], class_ids[col])

        return self.objects

# ==============================================================================
# Hailo Inference Core
# ==============================================================================
class HailoInference:
    def __init__(self, hef_path, conf_threshold=0.3):
        if not HAILO_AVAILABLE:
            raise ImportError("HailoRT library (hailo_platform) not found.")
        
        self.conf_threshold = conf_threshold
        self.hef = HEF(hef_path)
        self.target = VDevice()
        
        # Configure network group
        configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        
        # Stream info
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        
        # Get input dimensions
        self.input_info = self.hef.get_input_vstream_infos()[0]
        print(f"🔍 Hailo Input: {self.input_info.name} | Shape: {self.input_info.shape} | Format: {self.input_info.format.type}")
        
        # Determine height/width (handle both object type and tuple)
        if hasattr(self.input_info.shape, 'height'):
            self.input_height = self.input_info.shape.height
            self.input_width = self.input_info.shape.width
        else:
            # Usually (Batch, H, W, C) or (H, W, C)
            if len(self.input_info.shape) == 4:
                self.input_height, self.input_width = self.input_info.shape[1:3]
            else:
                self.input_height, self.input_width = self.input_info.shape[0:2]
        
        self.input_shape = (self.input_height, self.input_width)
        
        # Detect if model is "Raw" (many outputs) or "NMS" (single detection output)
        self.output_infos = self.hef.get_output_vstream_infos()
        self.is_nms_model = any("nms" in info.name.lower() or "detection" in info.name.lower() for info in self.output_infos)
        print(f"📊 Outputs found: {len(self.output_infos)} | NMS Included: {self.is_nms_model}")
        for info in self.output_infos:
            print(f"  - {info.name}: {info.shape}")

        # Initialize the infer pipeline
        self.infer_pipeline = InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params)
        self._ng_ctx = None

    def __enter__(self):
        # Activate the network group before running inference.
        try:
            self._ng_ctx = self.network_group.activate(self.network_group_params)
        except TypeError:
            self._ng_ctx = self.network_group.activate()
        if self._ng_ctx is not None:
            self._ng_ctx.__enter__()
        self.infer_pipeline.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.infer_pipeline.__exit__(exc_type, exc_val, exc_tb)
        if self._ng_ctx is not None:
            self._ng_ctx.__exit__(exc_type, exc_val, exc_tb)
        self.target.release()

    def preprocess(self, frame):
        # YOLOv8 expectation: Resized, RGB
        resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
        return resized

    def run(self, frame):
        # Resize to model input size
        resized = self.preprocess(frame)

        # Prepare input tensor. Ensure dtype and shape match (N,H,W,C).
        input_tensor = np.ascontiguousarray(resized, dtype=np.uint8)
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)
        input_data = {self.input_info.name: input_tensor}

        try:
            # HailoRT API differs across versions; prefer no batch_size.
            try:
                outputs = self.infer_pipeline.infer(input_data)
            except TypeError as e:
                if "batch_size" not in str(e):
                    raise
                outputs = self.infer_pipeline.infer(input_data, batch_size=input_tensor.shape[0])
        except Exception as e:
            # If 3D fails, try 4D (Batch size 1) - though this shouldn't happen with batch_size param
            if "match the frame count" in str(e):
                for alt in (input_tensor, resized):
                    input_data[self.input_info.name] = np.ascontiguousarray(alt, dtype=np.uint8)
                    try:
                        outputs = self.infer_pipeline.infer(input_data)
                        break
                    except TypeError as e:
                        if "batch_size" not in str(e):
                            raise
                        outputs = self.infer_pipeline.infer(input_data, batch_size=1)
                        break
                else:
                    raise e
            else:
                raise e
                
        return self.postprocess(outputs, frame.shape)

    def postprocess(self, outputs, orig_shape):
        # NOTE: This assumes the HEF has the NMS layer included (standard for official RPi5 examples)
        # Official Hailo YOLOv8 NMS output usually has streams named like 'yolov8_nms_layer'
        # Or it might be individual streams for boxes, scores, etc.
        # For simplicity, we search for the 'detection' key or similar
        
        boxes, scores, classes = [], [], []
        h_orig, w_orig = orig_shape[:2]

        def _iter_nms_arrays(tensor):
            if isinstance(tensor, (list, tuple)):
                if len(tensor) == 1:
                    yield from _iter_nms_arrays(tensor[0])
                    return
                try:
                    stacked = np.stack([np.asarray(t) for t in tensor], axis=0)
                    yield stacked
                except Exception:
                    for item in tensor:
                        yield from _iter_nms_arrays(item)
                return

            arr = np.asarray(tensor)
            if arr.dtype == object:
                for item in arr.tolist():
                    yield from _iter_nms_arrays(item)
            else:
                yield arr

        if self.is_nms_model:
            # Hailo NMS results are usually in a single output named 'detection_layer' or similar
            # Format: [ymin, xmin, ymax, xmax, score, class_id]
            for key, tensor in outputs.items():
                if 'nms' in key.lower() or 'detection' in key.lower():
                    for arr in _iter_nms_arrays(tensor):
                        if arr.size == 0:
                            continue
                        if arr.ndim >= 4 and arr.shape[0] == 1:
                            arr = arr[0]
                        # Common Hailo NMS format: (classes, 5, max_det)
                        if arr.ndim == 3 and arr.shape[1] == 5:
                            num_classes, _, max_det = arr.shape
                            for class_id in range(num_classes):
                                for det_idx in range(max_det):
                                    score = arr[class_id, 4, det_idx]
                                    if score > self.conf_threshold:
                                        ymin, xmin, ymax, xmax = arr[class_id, 0:4, det_idx]
                                        boxes.append([int(xmin * w_orig), int(ymin * h_orig), int(xmax * w_orig), int(ymax * h_orig)])
                                        scores.append(float(score))
                                        classes.append(int(class_id))
                            continue
                        # Generic formats: (num_det, 6) or (batch, num_det, 6)
                        if arr.ndim == 3 and arr.shape[-1] >= 6:
                            arr = arr.reshape(-1, arr.shape[-1])
                        if arr.ndim == 2 and arr.shape[1] >= 6:
                            for det in arr:
                                score = det[4]
                                if score > self.conf_threshold:
                                    ymin, xmin, ymax, xmax = det[0:4]
                                    boxes.append([int(xmin * w_orig), int(ymin * h_orig), int(xmax * w_orig), int(ymax * h_orig)])
                                    scores.append(float(score))
                                    classes.append(int(det[5]))
            return boxes, scores, classes

        # Fallback for raw models (Manual YOLO decoding - though we aim for NMS-integrated)
        # ... existing fallback code if needed, but we'll focus on NMS ...
        return boxes, scores, classes

# ==============================================================================
# Custom Sack Counter (Line Crossing)
# ==============================================================================
class SackCounter:
    def __init__(self, line_points, classes_to_count=[1]):
        self.line_points = line_points # [(x1,y1), (x2,y2)]
        self.classes_to_count = classes_to_count
        self.tracker = CentroidTracker()
        self.in_count = 0
        self.out_count = 0
        self.track_history = {} # {id: last_centroid}

    def is_crossing(self, p1, p2):
        # Line from (x1, y1) to (x2, y2)
        (lx1, ly1), (lx2, ly2) = self.line_points
        (px1, py1) = p1
        (px2, py2) = p2
        
        # Determine if points are on opposite sides of the line (Simple cross product approach)
        def side(lp1, lp2, p):
            return (lp2[0] - lp1[0]) * (p[1] - lp1[1]) - (lp2[1] - lp1[1]) * (p[0] - lp1[0])

        s1 = side((lx1, ly1), (lx2, ly2), p1)
        s2 = side((lx1, ly1), (lx2, ly2), p2)
        
        if s1 * s2 < 0: # Opposite signs = crossing
            # Optional: check direction
            if s2 > 0: return "in"
            else: return "out"
        return None

    def update(self, boxes, class_ids):
        # Filter boxes by classes_to_count
        filtered_boxes = []
        filtered_classes = []
        for i, cid in enumerate(class_ids):
            if cid in self.classes_to_count:
                filtered_boxes.append(boxes[i])
                filtered_classes.append(cid)
        
        objects = self.tracker.update(filtered_boxes, filtered_classes)
        
        for object_id, (centroid, class_id) in objects.items():
            if object_id in self.track_history:
                last_centroid = self.track_history[object_id]
                direction = self.is_crossing(last_centroid, centroid)
                if direction == "in": self.in_count += 1
                elif direction == "out": self.out_count += 1
            
            self.track_history[object_id] = centroid
        
        return objects

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="picamera0", help="Source: usb0, picamera0, or video path")
    parser.add_argument("--conf", default=0.35, type=float, help="Confidence threshold")
    parser.add_argument("--resolution", default="640x480", help="Resolution WxH")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--line", default="vertical", choices=["vertical", "horizontal"], help="Line orientation")
    args = parser.parse_args()

    # 1. Setup Stream
    resW, resH = map(int, args.resolution.split("x"))
    stream = StreamLoader(args.source, resW, resH)
    while not stream.grabbed:
        time.sleep(0.1)
        if stream.stopped: break
    ret, frame = stream.read()
    h, w = frame.shape[:2]
    print(f"✅ Stream Ready: {w}x{h}")

    # 2. Setup Logic
    if args.line == "vertical":
        line_points = [(w // 2, 0), (w // 2, h)]
    else:
        line_points = [(0, h // 2), (w, h // 2)]
    
    target_classes = [1] if args.headless else [0, 1]
    counter = SackCounter(line_points, classes_to_count=target_classes)

    # 3. Setup Recording
    writer = None
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("records", exist_ok=True)
        video_path = f"records/Hailo_{timestamp}.mp4"
        writer = AsyncVideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        print(f"💾 Recording to: {video_path}")

    # 4. Setup MQTT (Async)
    mq_host = os.environ.get('MQTT_HOST', 'mqtt')
    mq_port = int(os.environ.get('MQTT_PORT', 1883))
    mq_user = os.environ.get('MQTT_USER')
    mq_pass = os.environ.get('MQTT_PASS')
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client() # Fallback for older paho-mqtt
    if mq_user and mq_pass:
        client.username_pw_set(mq_user, mq_pass)
    try:
        client.connect_async(mq_host, mq_port, 60)
        client.loop_start()
        print(f"📡 MQTT Configured: {mq_host}")
    except: pass

    # 5. Inference Loop
    print(f"🔄 Initializing Hailo-8L with: {HEF_PATH}")
    if not HAILO_AVAILABLE:
        print("❌ HailoRT not found. Exiting.")
        return

    try:
        with HailoInference(str(HEF_PATH), conf_threshold=args.conf) as model:
            t0 = time.time()
            frame_cnt = 0
            
            while True:
                if stream.stopped: break
                ret, frame = stream.read()
                if not ret: continue

                # Inference & Post-Process
                boxes, scores, classes = model.run(frame)
                
                # Tracking & Counting
                tracked_objects = counter.update(boxes, classes)

                # Summary text for logs and overlays
                info = f"In: {counter.in_count} Out: {counter.out_count}"

                # Annotation (only if NOT headless or saving)
                if not args.headless or args.save:
                    # Draw Line
                    cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 2)
                    
                    # Draw Detections
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        label = f"Class {classes[i]} {scores[i]:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Draw Tracked Counts
                    cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Save / Display
                if writer: writer.write(frame)
                if not args.headless:
                    cv2.imshow("Hailo-8L Edge Detector", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break

                # Stats & MQTT
                frame_cnt += 1
                if frame_cnt % 30 == 0:
                    fps = frame_cnt / (time.time() - t0)
                    print(f"Hailo-8L FPS: {fps:.2f} | {info}")
                    if client:
                        client.publish("sack/stats", json.dumps({"fps": round(fps, 2), "in": counter.in_count, "out": counter.out_count, "engine": "hailo"}))
                    t0 = time.time()
                    frame_cnt = 0

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print("👋 Hailo Engine Shutdown.")

if __name__ == "__main__":
    main()
