1. My first working version was very slow on Colab because it defaulted to the CPU. It took over an hour for just 5 files. The final version forced the GPU (`device=0`) and used multithreading for decoding, which drastically sped up inference. The single change that saved the most time was forcing the YOLO model to run on the GPU instead of the CPU. (65 words)

2. The script uses PyArrow to read Parquet files in batches. It extracts the video bytes and puts them in a queue. Decoder threads use PyAV to open the video from memory, extract frames, and apply rotation if needed. A single inference thread batches 32 frames at a time and runs YOLO11l. Finally, the script groups detections by second, checks if an object appears in >=50% of frames, calculates the median people count, and writes to JSONL. (76 words)

3. Video decoding takes the most time. I measured this by observing the Colab CPU warning and seeing the huge delay before GPU inference happened. PyAV decoding from Parquet bytes is bottlenecked by the CPU. The YOLO model itself runs very fast on the GPU once the frames are extracted. To optimize, future versions should use hardware-accelerated video decoding. (58 words)

4. I checked correctness by writing a small test script that processed a standard 50-second MP4 video file of people walking. I manually watched the video and compared it to the JSON output printed by the script. The script correctly identified the timestamps where people entered and left the frame. (50 words)

5. The AI suggested exporting the model to TensorRT for maximum speed. I tried this, but it crashed on the free Colab T4 because it ran out of memory during the complex conversion process. I rejected this idea and forced the script to fall back to the standard PyTorch model, which is slower but 100% reliable on Colab. (57 words)

6. If I had one more week, I would figure out how to build PyAV with CUDA hardware decoding. The biggest bottleneck is the CPU decoding the video frames. If I could offload that to the GPU using `h264_cuvid`, the script would be infinitely faster and could process all 50 files in an hour. (55 words)
