"""
Create summary video from selected frame indices.
Extracts relevant frames and creates a new video showing only the important segments.
"""

import json
import h5py
import argparse
import numpy as np
import subprocess
from pathlib import Path

def get_video_metadata(h5_path, video_key):
    """Get video metadata from H5 dataset."""
    with h5py.File(h5_path, 'r') as f:
        video_data = f[video_key]
        n_frames = video_data['n_frames'][()]
    
    # Derive video name from key (e.g., "video_14" -> "Video_14.mp4")
    video_num = video_key.split('_')[1]
    video_name = f"Video_{video_num}.mp4"
    
    return {
        'n_frames': n_frames,
        'video_name': video_name
    }

def get_video_duration(video_path):
    """Get video duration using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return float(result.stdout.strip())
    else:
        print(f"Warning: Could not get duration from video")
        return None

def frames_to_time_ranges(frame_indices):
    """Convert frame indices to contiguous time ranges."""
    if len(frame_indices) == 0:
        return []
    
    # Sort indices
    sorted_indices = sorted(frame_indices)
    
    # Group into contiguous segments
    ranges = []
    start = sorted_indices[0]
    end = sorted_indices[0]
    
    for idx in sorted_indices[1:]:
        if idx == end + 1:
            # Contiguous
            end = idx
        else:
            # Gap found, save current range
            ranges.append((start, end))
            start = idx
            end = idx
    
    # Don't forget the last range
    ranges.append((start, end))
    
    return ranges

def create_filter_complex(time_ranges, fps):
    """Create ffmpeg filter_complex for extracting and concatenating segments."""
    segments = []
    
    for i, (start_frame, end_frame) in enumerate(time_ranges):
        start_time = start_frame / fps
        end_time = (end_frame + 1) / fps  # +1 to include the end frame
        duration = end_time - start_time
        
        segments.append({
            'index': i,
            'start': start_time,
            'duration': duration
        })
    
    return segments

def create_summary_video(video_path, frame_indices, metadata, output_path):
    """Create summary video using ffmpeg."""
    fps = metadata['fps']
    time_ranges = frames_to_time_ranges(frame_indices)
    
    print(f"\nVideo: {metadata['video_name']}")
    print(f"Total frames: {metadata['n_frames']}")
    print(f"Selected frames: {len(frame_indices)}")
    print(f"FPS: {fps}")
    print(f"Original duration: {metadata['duration']:.2f}s")
    print(f"\nContiguous segments: {len(time_ranges)}")
    
    # Create temporary files for each segment
    temp_files = []
    temp_list_file = "temp_concat_list.txt"
    
    try:
        for i, (start_frame, end_frame) in enumerate(time_ranges):
            start_time = start_frame / fps
            duration = (end_frame - start_frame + 1) / fps
            temp_file = f"temp_segment_{i}.mp4"
            temp_files.append(temp_file)
            
            print(f"  Segment {i+1}: frames {start_frame}-{end_frame} ({start_time:.2f}s, {duration:.2f}s)")
            
            # Extract segment
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', str(video_path),
                '-c', 'copy',
                '-y',
                temp_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Failed to extract segment {i}: {result.stderr}")
        
        # Create concat list file
        with open(temp_list_file, 'w') as f:
            for temp_file in temp_files:
                f.write(f"file '{temp_file}'\n")
        
        # Concatenate all segments
        print(f"\nConcatenating {len(temp_files)} segments...")
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list_file,
            '-c', 'copy',
            '-y',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\n✓ Summary video created: {output_path}")
            
            # Get output duration
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                summary_duration = float(result.stdout.strip())
                compression_ratio = (summary_duration / metadata['duration']) * 100
                print(f"Summary duration: {summary_duration:.2f}s ({compression_ratio:.1f}% of original)")
        else:
            print(f"Error concatenating: {result.stderr}")
    
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            Path(temp_file).unlink(missing_ok=True)
        Path(temp_list_file).unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Create summary video from frame indices')
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to frame_indices_with_scores.json')
    parser.add_argument('--h5_path', type=str, required=True,
                       help='Path to TVSum dataset H5 file')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing original videos')
    parser.add_argument('--video_key', type=str, default='video_14',
                       help='Video key (e.g., video_14)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (default: {video_key}_summary.mp4)')
    
    args = parser.parse_args()
    
    # Load frame indices from JSON
    print(f"Loading frame indices from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        results = json.load(f)
    
    if args.video_key not in results:
        print(f"Error: {args.video_key} not found in JSON file")
        print(f"Available videos: {list(results.keys())}")
        return
    
    frame_indices = results[args.video_key]['indices']
    
    # Get metadata from H5
    print(f"Loading metadata from {args.h5_path}...")
    metadata = get_video_metadata(args.h5_path, args.video_key)
    
    # Find video file
    video_path = Path(args.video_dir) / metadata['video_name']
    if not video_path.exists():
        # Try common extensions
        for ext in ['.mp4', '.avi', '.mkv', '.mov']:
            test_path = Path(args.video_dir) / (metadata['video_name'] + ext)
            if test_path.exists():
                video_path = test_path
                break
        
        # Try without extension
        if not video_path.exists():
            name_without_ext = Path(metadata['video_name']).stem
            for ext in ['.mp4', '.avi', '.mkv', '.mov']:
                test_path = Path(args.video_dir) / (name_without_ext + ext)
                if test_path.exists():
                    video_path = test_path
                    break
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        print(f"Tried: {metadata['video_name']}")
        return
    
    # Calculate FPS from video duration and frame count
    print(f"Calculating FPS from video duration and frame count...")
    duration = get_video_duration(video_path)
    
    if duration:
        fps = metadata['n_frames'] / duration
        print(f"  Video duration: {duration:.2f}s")
        print(f"  Frames in H5: {metadata['n_frames']}")
        print(f"  Calculated FPS: {fps:.2f}")
    else:
        # Fallback to default
        fps = 30.0
        duration = metadata['n_frames'] / fps
        print(f"  Warning: Using default 30 FPS")
    
    metadata['fps'] = fps
    metadata['duration'] = duration
    
    # Set output path
    output_path = args.output if args.output else f"{args.video_key}_summary.mp4"
    
    # Create summary video
    create_summary_video(video_path, frame_indices, metadata, output_path)

if __name__ == '__main__':
    main()
