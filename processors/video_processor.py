import cv2
import time
from utils.box_utils import is_inside, compress_box, iou

class VideoProcessor:
    """
    Processes a video using a YOLO model to detect workers and mobiles.
    Tracks when workers are using mobile phones and overlays visual information on the output video.
    """

    def __init__(self, model, input_size=(1088, 1088), buffer_frames=26, iou_thresh=0.3, compression_ratio=0.1):
        """
        Initialize the VideoProcessor.

        Args:
            model: YOLO model used for detection.
            input_size (tuple): Input resolution for the model.
            buffer_frames (int): Number of frames to tolerate before ending a phone usage event.
            iou_thresh (float): IoU threshold for comparing bounding boxes in buffer mode.
            compression_ratio (float): Compression ratio applied to worker boxes (for stricter containment check).
        """
        self.model = model
        self.input_size = input_size
        self.buffer_frames = buffer_frames
        self.iou_thresh = iou_thresh
        self.compression_ratio = compression_ratio

    def draw_transparent_rect(self, frame, x, y, w, h, color=(0, 0, 0), alpha=0.6):
        """Draw a semi-transparent rectangle on the video frame (used for overlays)."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def process(self, video_path: str, output_path: str):
        """
        Process a single video file:
          - Run YOLO detections per frame
          - Detect if workers are using mobile phones
          - Track phone usage periods with buffering
          - Draw overlays (bounding boxes, labels, FPS, usage time)
          - Save processed video to output_path

        Returns:
            video_name (str): Name of the video file
            phone_usage_periods (list): List of (start_frame, end_frame) where phone usage occurred
            frame_count (int): Total number of frames processed
            fps (float): Frames per second of input video
            frames_with_phone (int): Number of frames with phone detected
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # State variables for tracking
        frame_count = 0
        phone_usage_periods = []      # List of (start_frame, end_frame) intervals
        phone_active = False          # Whether phone usage is ongoing
        period_start = None           # Start frame of current phone usage period
        active_boxes = []             # List of current worker-mobile pairs
        last_active_boxes = []        # Last seen worker-mobile pairs (used for buffering)
        buffer_counter = 0            # Counts frames inside buffer tolerance
        phone_usage_flags = []        # Per-frame boolean flags (phone detected or not)
        frames_with_phone = 0         # Counter of frames with phone usage detected
        prev_time = time.time()       # For FPS calculation
        video_name = video_path.split("/")[-1]  # Extract filename from path

        # Main video loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Resize frame to model input size and run YOLO inference
            orig_h, orig_w = frame.shape[:2]
            frame_resized = cv2.resize(frame, self.input_size)
            results = self.model(frame_resized)[0]

            # Separate detections into mobiles and workers
            mobiles, workers = [], []
            for box in results.boxes:
                cls = int(box.cls[0])              # class index
                conf = float(box.conf[0])          # confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # predicted coordinates

                # Rescale coordinates back to original frame size
                scale_x = orig_w / self.input_size[0]
                scale_y = orig_h / self.input_size[1]
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                # Separate based on class labels
                if self.model.names[cls] == "mobile":
                    mobiles.append(((x1, y1, x2, y2), conf))
                elif self.model.names[cls] == "worker":
                    workers.append(((x1, y1, x2, y2), conf))

            # Check if a mobile is inside a compressed worker bounding box
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

            # Buffer logic to smooth short gaps in detection
            if phone_detected_in_frame:
                phone_active = True
                period_start = period_start or frame_count
                active_boxes = current_active_boxes
                last_active_boxes = current_active_boxes
                buffer_counter = 0
            else:
                # If no phone detected but still within buffer range
                if phone_active and buffer_counter < self.buffer_frames:
                    if last_active_boxes:
                        for (lw, lm, _lc) in last_active_boxes:
                            ref_pair = [(lw, lm, _lc)]
                            for tup in (current_active_boxes or ref_pair):
                                w, m, _c = tup
                                # Compare overlap with last active boxes using IoU
                                if iou(lm, m) > self.iou_thresh or iou(lw, w) > self.iou_thresh:
                                    phone_detected_in_frame = True
                                    active_boxes = last_active_boxes
                                    buffer_counter += 1
                                    break
                else:
                    # End of phone usage period
                    if phone_active:
                        phone_active = False
                        phone_usage_periods.append((period_start, frame_count - 1))
                        period_start = None
                        active_boxes = []
                        buffer_counter = 0

            # Track per-frame phone detection
            phone_usage_flags.append(phone_detected_in_frame)

            # Increment frame counter for phone usage
            if phone_detected_in_frame:
                frames_with_phone += 1

            # Draw bounding boxes and labels
            for worker_box, mobile_box, mobile_conf in active_boxes:
                wx1, wy1, wx2, wy2 = worker_box
                mx1, my1, mx2, my2 = mobile_box
                cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 5)      # Red box for worker
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 255), 5)    # Yellow box for mobile

                # Label above worker box
                label = f"Using mobile {mobile_conf:.2f}"
                text_x = max(0, wx1)
                text_y = max(30, wy1 - 10)
                cv2.putText(frame, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 255), 3, cv2.LINE_AA)

            # Overlay total phone usage time + instantaneous FPS
            total_phone_sec = sum((end - start + 1) / fps for start, end in phone_usage_periods)
            if phone_active and period_start:
                total_phone_sec += (frame_count - period_start + 1) / fps

            current_time = time.time()
            current_fps = 1 / (current_time - prev_time) if (time.time() - prev_time) > 0 else 0.0
            prev_time = current_time

            phone_text = f"Phone Usage: {total_phone_sec:.2f} sec"
            fps_text = f"FPS: {round(current_fps)}"

            # Draw semi-transparent overlay box for text
            self.draw_transparent_rect(frame, 45, 30, 500, 140)
            cv2.putText(frame, phone_text, (60, 90),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, fps_text,   (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4, cv2.LINE_AA)

            # Write processed frame to output video
            out.write(frame)

        # If video ended while still active, close the last usage period
        if phone_active and period_start:
            phone_usage_periods.append((period_start, frame_count))

        # Release resources
        cap.release()
        out.release()

        # Return all collected data
        return video_name, phone_usage_periods, frame_count, fps, frames_with_phone
