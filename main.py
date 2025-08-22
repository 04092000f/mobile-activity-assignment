# main.py
import argparse
import os
from processors.model_handler import ModelHandler
from processors.directory_processor import DirectoryProcessor
from processors.video_processor import VideoProcessor

if __name__ == "__main__":
    # ---------------------------
    # Argument parsing
    # ---------------------------
    parser = argparse.ArgumentParser(description="Phone Usage Detection in Videos")

    # Paths to models (PyTorch and ONNX versions)
    parser.add_argument("--pt_model", type=str, default="models/mobile_activity.pt")
    parser.add_argument("--onnx_model", type=str, default="models/mobile_activity.onnx")

    # Input can be a single video file or a directory containing multiple videos
    parser.add_argument("--input", type=str, required=True, help="Input video or directory")

    # Output directory where processed videos/results will be saved
    parser.add_argument("--output", type=str, default="output_videos", help="Output directory")

    # IoU threshold for pseudo-tracking across frames
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="IoU threshold for pseudo-tracking")

    # Number of frames to buffer for smoothing phone detection results
    parser.add_argument("--buffer_frames", type=int, default=26, help="Number of buffer frames for phone detection smoothing")

    # Compression ratio applied to detected worker bounding boxes (shrinks box area)
    parser.add_argument("--compression", type=float, default=0.1, help="Compression ratio for worker bounding box")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # ---------------------------
    # Load detection model
    # ---------------------------
    handler = ModelHandler(args.pt_model, args.onnx_model)
    model = handler.load_model()

    # ---------------------------
    # Process single video file
    # ---------------------------
    if os.path.isfile(args.input):
        vp = VideoProcessor(
            model,
            iou_thresh=args.iou_thresh,
            buffer_frames=args.buffer_frames,
            compression_ratio=args.compression
        )

        # Build output path for processed video
        output_path = os.path.join(args.output, f"processed_{os.path.basename(args.input)}")

        # Process video: returns metadata (video name, phone usage periods, total frames, fps)
        video_name, phone_usage_periods, total_frames, fps = vp.process(args.input, output_path)

        print(f"Processed single video: {video_name}")

    # ---------------------------
    # Process all videos in a directory
    # ---------------------------
    elif os.path.isdir(args.input):
        dp = DirectoryProcessor(model)

        # Update video processor parameters
        dp.video_processor.iou_thresh = args.iou_thresh
        dp.video_processor.buffer_frames = args.buffer_frames
        dp.video_processor.compression_ratio = args.compression

        # Process all videos in directory
        dp.process_directory(args.input, args.output)

    # ---------------------------
    # Handle invalid input path
    # ---------------------------
    else:
        print("Input path does not exist:", args.input)