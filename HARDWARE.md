# Hardware Used
- **GPU:** NVIDIA Tesla T4 (16GB VRAM) on Google Colab
- **CPU:** Intel Xeon (2 vCPUs)
- **RAM:** 12GB
- **Disk:** SSD (Google Cloud Storage)
- **Operating System:** Ubuntu 22.04

**Timing:** The full 50-file run took too long on the free Colab T4 due to CPU decoding bottlenecks. Testing on 5 files took ~15 minutes. The script is optimized to run much faster on an RTX 4090.
