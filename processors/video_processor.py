# processors/video_processor.py
import cv2
import time
from utils.box_utils import is_inside, compress_box, iou

class VideoProcessor:
    """
    Processes a video using a YOLO model to detect workers and mobiles.
    Tracks when workers are using mobile phones and overlays visual information on the output video.
    """

    def __init__(self, model, input_size=(1088, 1088), buffer_frames=26, iou_thresh=0.3, compression_ratio=0.1):
        self.model = model
        self.input_size = input_size
        self.buffer_frames = buffer_frames
        self.iou_thresh = iou_thresh
        self.compression_ratio = compression_ratio

    def draw_transparent_rect(self, frame, x, y, w, h, color=(0, 0, 0), alpha=0.6):
        """Draw a semi-transparent rectangle on the video frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w + 80, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def process(self, video_path: str, output_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # State variables
        frame_count = 0
        phone_usage_periods = []      # (start_frame, end_frame)
        phone_active = False
        period_start = None
        active_boxes = []             # list of (worker_box, mobile_box, mobile_conf)
        last_active_boxes = []
        buffer_counter = 0
        phone_usage_flags = []
        frames_with_phone = 0         # NEW: counter for frames where phone was detected
        prev_time = time.time()
        video_name = video_path.split("/")[-1]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Resize to model input size and run inference
            orig_h, orig_w = frame.shape[:2]
            frame_resized = cv2.resize(frame, self.input_size)
            results = self.model(frame_resized)[0]

            # Separate detections into workers and mobiles (with confidences)
            mobiles, workers = [], []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Rescale back to original frame size
                scale_x = orig_w / self.input_size[0]
                scale_y = orig_h / self.input_size[1]
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                if self.model.names[cls] == "mobile":
                    mobiles.append(((x1, y1, x2, y2), conf))
                elif self.model.names[cls] == "worker":
                    workers.append(((x1, y1, x2, y2), conf))

            # Check if any mobile is inside a (compressed) worker box
            phone_detected_in_frame = False
            current_active_boxes = []
            for (mobile_box, mobile_conf) in mobiles:
                for (worker_box, _wconf) in workers:
                    compressed_worker_box = compress_box(worker_box, compression=self.compression_ratio)
                    if is_inside(mobile_box, compressed_worker_box):
                        current_active_boxes.append((worker_box, mobile_box, mobile_conf))
                        phone_detected_in_frame = True
                        break
                if phone_detected_in_frame:
                    break

            # Buffer logic
            if phone_detected_in_frame:
                phone_active = True
                period_start = period_start or frame_count
                active_boxes = current_active_boxes
                last_active_boxes = current_active_boxes
                buffer_counter = 0
            else:
                if phone_active and buffer_counter < self.buffer_frames:
                    if last_active_boxes:
                        for (lw, lm, _lc) in last_active_boxes:
                            ref_pair = [(lw, lm, _lc)]
                            for tup in (current_active_boxes or ref_pair):
                                w, m, _c = tup
                                if iou(lm, m) > self.iou_thresh or iou(lw, w) > self.iou_thresh:
                                    phone_detected_in_frame = True
                                    active_boxes = last_active_boxes
                                    buffer_counter += 1
                                    break
                else:
                    if phone_active:
                        phone_active = False
                        phone_usage_periods.append((period_start, frame_count - 1))
                        period_start = None
                        active_boxes = []
                        buffer_counter = 0

            phone_usage_flags.append(phone_detected_in_frame)

            # NEW: count frames with phone
            if phone_detected_in_frame:
                frames_with_phone += 1

            # Draw worker & mobile boxes + RED label above worker with confidence
            for worker_box, mobile_box, mobile_conf in active_boxes:
                wx1, wy1, wx2, wy2 = worker_box
                mx1, my1, mx2, my2 = mobile_box
                cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 5)      # worker (red)

                label = f"Using mobile {mobile_conf:.2f}"
                text_x = max(0, wx1)
                text_y = max(30, wy1 - 10)
                cv2.putText(frame, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 255), 3, cv2.LINE_AA)

            # Overlay: phone usage secs + instantaneous FPS
            total_phone_sec = sum((end - start + 1) / fps for start, end in phone_usage_periods)
            if phone_active and period_start:
                total_phone_sec += (frame_count - period_start + 1) / fps

            current_time = time.time()
            current_fps = 1 / (current_time - prev_time) if (time.time() - prev_time) > 0 else 0.0
            prev_time = current_time

            phone_text = f"Phone Usage: {total_phone_sec:.2f} sec"
            fps_text = f"FPS: {round(current_fps)}"

            self.draw_transparent_rect(frame, 45, 30, 500, 140)
            cv2.putText(frame, phone_text, (60, 90),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, fps_text,   (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)

            out.write(frame)

        if phone_active and period_start:
            phone_usage_periods.append((period_start, frame_count))

        cap.release()
        out.release()
        # Now returning frames_with_phone as well
        return video_name, phone_usage_periods, frame_count, fps, frames_with_phone

