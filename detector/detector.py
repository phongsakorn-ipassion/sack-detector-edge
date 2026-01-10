import os
import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, solutions
from datetime import datetime
from threading import Thread
from queue import Queue

# Picamera2 import for Raspberry Pi
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

# =========================
# UTILS: Threaded Video Writer
# =========================
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
        if self.queue.full(): return # Drop frame if writing is too slow
        self.queue.put(frame)

    def update(self):
        while not self.stopped:
            if not self.queue.empty():
                frame = self.queue.get()
                self.writer.write(frame)
                self.queue.task_done()
            else:
                time.sleep(0.01)
        
        # Flush queue on stop
        while not self.queue.empty():
            frame = self.queue.get()
            self.writer.write(frame)
            self.queue.task_done()
            
        self.writer.release()

    def release(self):
        self.stopped = True
        self.thread.join()

# =========================
# UTILS: Threaded Stream Loader
# =========================
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

        # Determine source type
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

        # Start thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def config_picamera(self):
        config = self.cap.create_video_configuration(
            main={"format": "XRGB8888", "size": (self.width, self.height)},
            controls={"FrameRate": 30} # Request 30fps
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
                # Picamera2 capture is simplified here. 
                # For high performance, using request_sequence is better, 
                # but capture_array is safe for basic usage.
                frame = self.cap.capture_array()
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self.grabbed = True
            
            # For file video execution speed control could be added here
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

# =========================
# CLI Arguments
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=Path("models/detection.onnx"), help="Path to YOLO model")
parser.add_argument("--source", default="picamera0", help="Source: usb0, picamera0, or video path")
parser.add_argument("--conf", default=0.35, type=float, help="Confidence threshold")
parser.add_argument("--resolution", default="640x480", help="Resolution WxH")
parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config (e.g. bytetrack.yaml)")
parser.add_argument("--iou", default=0.7, type=float, help="IOU threshold")
parser.add_argument("--max_det", default=50, type=int, help="Max detections")
parser.add_argument("--save", action="store_true", help="Save output video")
parser.add_argument("--headless", action="store_true", help="Run without GUI (faster)") # Optimization
parser.add_argument("--line", default="vertical", choices=["vertical", "horizontal"], help="Region line orientation")
args = parser.parse_args()

# =========================
# Main
# =========================
def main():
    # Parse Resolution
    resW, resH = map(int, args.resolution.split("x"))
    
    # Init Model
    MODEL_PATH = Path(args.model)
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        sys.exit(1)
        
    print(f"🔄 Loading Model: {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))
    
    # Init Stream
    print(f"📷 Opening Stream: {args.source}")
    stream = StreamLoader(args.source, resW, resH)
    
    # Wait for first frame
    print("Waiting for camera...")
    while not stream.grabbed:
        time.sleep(0.1)
        if stream.stopped: break
        
    ret, frame = stream.read()
    h, w = frame.shape[:2]
    print(f"✅ Stream Ready: {w}x{h}")

    # Init Output Writer (Lazy init)
    writer = None
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"records/Output_{timestamp}.mp4"
        # We assume 15fps avg for writing to be safe, or dynamic? 
        # Let's fix at 30 for safety or use dynamic logic.
        # Ideally, we start writer after FPS stabilizes, but for simplicity:
        writer = AsyncVideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        print(f"💾 Recording enabled: {video_path}")

    # Logic Config
    if args.line == "vertical":
        region_points = [(w // 2, 0), (w // 2, h)]
    else:
        region_points = [(0, h // 2), (w, h // 2)]
    counter = solutions.ObjectCounter(
        model=model,
        region=region_points,
        classes=[0, 1],
        conf=args.conf,
        tracker=args.tracker,
        iou=args.iou,
        max_det=args.max_det,
        show=False,
        verbose=False
    )

    t0 = time.time()
    frame_count = 0
    
    print("🚀 Start Loop")
    try:
        while True:
            # 1. Get Frame (Non-blocking access to latest frame)
            if stream.stopped: break
            ret, frame = stream.read()
            if frame is None: continue

            # 2. Inference
            # For optimal speed, we might want to resize just for inference?
            # ObjectCounter handles it internally usually.
            results = counter(frame)
            if results is None: continue
            
            # 3. Output
            out_frame = results.plot_im
            
            # 4. Async Write
            if writer:
                writer.write(out_frame)
            
            # 5. Display (Skip if headless)
            if not args.headless:
                cv2.imshow("YOLO Edge", out_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # FPS Calc
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t0
                fps = frame_count / elapsed
                print(f"FPS: {fps:.2f} | In: {counter.in_count} Out: {counter.out_count}")
                t0 = time.time()
                frame_count = 0

    except KeyboardInterrupt:
        print("Stopping...")
        
    finally:
        stream.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print("👋 Exited cleanly")

if __name__ == "__main__":
    main()
