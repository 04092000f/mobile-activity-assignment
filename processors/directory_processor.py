# processors/directory_processor.py
import os
import csv
from .video_processor import VideoProcessor

class DirectoryProcessor:
    def __init__(self, model, input_size=(1088, 1088)):
        self.video_processor = VideoProcessor(model, input_size=input_size)

    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        all_usage_summaries = []

        for filename in os.listdir(input_dir):
            if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"processed_{filename}")
                print(f"Processing {input_path} -> {output_path}")

                # Expect VideoProcessor.process to return frames_with_phone also
                video_name, phone_usage_periods, total_frames, video_fps, frames_with_phone = \
                    self.video_processor.process(input_path, output_path)

                usage_percentage = (frames_with_phone / total_frames * 100) if total_frames > 0 else 0

                for start_frame, end_frame in phone_usage_periods:
                    start_sec = start_frame / video_fps
                    end_sec = end_frame / video_fps
                    all_usage_summaries.append((
                        video_name, start_frame, end_frame, start_sec, end_sec,
                        total_frames, frames_with_phone, usage_percentage
                    ))

        # Save CSV summary
        csv_file = os.path.join(output_dir, "phone_usage_summary.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_name", "start_frame", "end_frame",
                "start_sec", "end_sec", "total_frames",
                "frames_with_phone", "usage_percentage"
            ])
            writer.writerows(all_usage_summaries)
        print(f"Phone usage summary saved to {csv_file}")
