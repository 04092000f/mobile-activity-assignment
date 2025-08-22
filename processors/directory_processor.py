import os
import csv
from .video_processor import VideoProcessor

class DirectoryProcessor:
    """
    Handles batch processing of videos in a directory using the VideoProcessor.
    Generates phone usage summaries (frame ranges, seconds, and percentages) and
    saves results into a CSV file.
    """

    def __init__(self, model, input_size=(1088, 1088)):
        """
        Initialize the DirectoryProcessor with a given YOLO model.

        Args:
            model: The YOLO model used for detection.
            input_size (tuple): The input image size (width, height) for inference.
        """
        # Create a VideoProcessor instance to process individual videos
        self.video_processor = VideoProcessor(model, input_size=input_size)

    def process_directory(self, input_dir, output_dir):
        """
        Process all video files in the given directory and generate a CSV summary.

        Args:
            input_dir (str): Path to the input directory containing video files.
            output_dir (str): Path to the output directory for processed videos and summary CSV.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # List to store phone usage summaries for all videos
        all_usage_summaries = []

        # Iterate over all files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
                # Construct input and output file paths
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"processed_{filename}")
                print(f"Processing {input_path} -> {output_path}")

                # Process the video and collect results
                # The VideoProcessor.process() method should return:
                #   video_name: Name of the video file
                #   phone_usage_periods: List of (start_frame, end_frame) where phone usage was detected
                #   total_frames: Total number of frames in the video
                #   video_fps: Frame rate of the video
                #   frames_with_phone: Number of frames with phone usage detected
                video_name, phone_usage_periods, total_frames, video_fps, frames_with_phone = \
                    self.video_processor.process(input_path, output_path)

                # Calculate the percentage of frames where phone usage was detected
                usage_percentage = (frames_with_phone / total_frames * 100) if total_frames > 0 else 0

                # Convert frame ranges into seconds and prepare summary rows
                for start_frame, end_frame in phone_usage_periods:
                    start_sec = start_frame / video_fps
                    end_sec = end_frame / video_fps
                    all_usage_summaries.append((
                        video_name, start_frame, end_frame, start_sec, end_sec,
                        total_frames, frames_with_phone, usage_percentage
                    ))

        # Save all usage summaries into a CSV file
        csv_file = os.path.join(output_dir, "phone_usage_summary.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Write CSV header
            writer.writerow([
                "video_name", "start_frame", "end_frame",
                "start_sec", "end_sec", "total_frames",
                "frames_with_phone", "usage_percentage"
            ])
            # Write data rows
            writer.writerows(all_usage_summaries)

        print(f"Phone usage summary saved to {csv_file}")
