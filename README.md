# Frigate NVR with NVIDIA GPU on Unraid 7.x

## The Story

This guide exists because I spent two days where nothing worked.

I wanted a simple thing: GPU-accelerated object detection in Frigate on my Unraid server with an NVIDIA RTX 5060 Ti. Should be straightforward, right? The Frigate docs mention ONNX support for NVIDIA GPUs. Just configure a detector and go.

Except Frigate 0.16 removed the TensorRT detector and no longer ships with pre-built models. The documentation points to a Google Colab notebook that was painfully slow. The official Docker build command for YOLOv9 failed with cryptic `requirements.txt not found` errors. When I finally got a model exported, it crashed with `Opset 22 is under development`. When *that* was fixed, the detector loaded but Frigate kept crashing with `exceeded fps limit` errors because AgentDVR was fighting for the same camera streams.

Two days. Countless container restarts. Multiple model formats (YOLO-NAS, YOLOv7, YOLOv9). ROCm images that couldn't find models. TensorRT images that needed models that didn't exist. Stack traces that led nowhere.

What finally worked was a combination of:
- The right export command with the right opset version
- Understanding that the `model` config is separate from `detectors`
- Realizing go2rtc exists specifically to prevent RTSP stream conflicts
- Knowing which parameters YOLOv9 actually needs (`yolo-generic`, not `yolonas`)

This guide is the distilled result of that struggle, with AI-assisted troubleshooting helping me work through each failure until we hit a stable configuration pulling ~5ms inference on GPU.

Save yourself the two days. Here's what actually works.

---

## What This Guide Covers

- Frigate NVR container setup with NVIDIA GPU passthrough
- YOLOv9 ONNX model export that actually works
- go2rtc configuration for multi-system camera access
- Every error I hit and how to fix it

---

## Tested Configuration

| Component | Details |
|-----------|---------|
| **Unraid Version** | 7.2.x |
| **Frigate Version** | 0.16.3 |
| **GPU** | NVIDIA RTX 5060 Ti (Blackwell architecture) |
| **CPU** | AMD Ryzen 9 7950X |
| **Container Image** | `ghcr.io/blakeblackshear/frigate:stable-tensorrt` |

> **Note:** This guide should work with other NVIDIA GPUs (RTX 20/30/40/50 series). The key requirement is CUDA support and the `-tensorrt` container image.

---

## Prerequisites

### 1. Install NVIDIA Driver Plugin on Unraid

1. Go to **Apps** tab in Unraid
2. Search for **Nvidia-Driver**
3. Install it
4. **Reboot Unraid** after installation
5. After reboot, verify the GPU is detected:
   ```bash
   nvidia-smi
   ```
   You should see your GPU listed with driver version and CUDA version.

### 2. Verify Docker NVIDIA Runtime

The Nvidia-Driver plugin typically configures this automatically. Verify with:
```bash
docker info | grep -i runtime
```
You should see `nvidia` listed as an available runtime.

---

## Step 1: Create the Frigate Container in Unraid

1. Go to **Docker** tab in Unraid
2. Click **Add Container**
3. Toggle **Basic View** to **OFF** (so you see advanced options)

### Container Settings

| Field | Value |
|-------|-------|
| **Name** | `FrigateNVR` |
| **Repository** | `ghcr.io/blakeblackshear/frigate:stable-tensorrt` |
| **Network Type** | `bridge` |
| **Extra Parameters** | `--runtime=nvidia --shm-size=256m` |

> **CRITICAL:** You MUST use the `stable-tensorrt` image tag. This image contains the CUDA and TensorRT libraries required for NVIDIA GPU acceleration with ONNX. The regular `stable` image does NOT include these libraries.

### Environment Variables

Click **"Add another Path, Port, Variable, Label or Device"** and select **Variable** for each:

| Config Type | Name | Key | Value |
|-------------|------|-----|-------|
| Variable | NVIDIA Devices | `NVIDIA_VISIBLE_DEVICES` | `all` |
| Variable | NVIDIA Capabilities | `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,video` |
| Variable | Timezone | `TZ` | Your timezone (e.g., `America/Toronto`) |

### Port Mappings

Click **"Add another Path, Port, Variable, Label or Device"** and select **Port** for each:

| Config Type | Name | Container Port | Host Port | Connection Type |
|-------------|------|----------------|-----------|-----------------|
| Port | Web UI | `5000` | `5000` | TCP |
| Port | Web UI Auth | `8971` | `8971` | TCP |
| Port | RTSP | `8554` | `8554` | TCP |
| Port | WebRTC TCP | `8555` | `8555` | TCP |
| Port | WebRTC UDP | `8555` | `8555` | **UDP** |
| Port | go2rtc API | `1984` | `1984` | TCP |

> **CRITICAL:** Port 8555 must be added **twice** - once for TCP and once for UDP. WebRTC requires both protocols to function properly. Missing the UDP mapping will cause live view issues.

### Volume Mappings

Click **"Add another Path, Port, Variable, Label or Device"** and select **Path** for each:

| Config Type | Name | Container Path | Host Path | Access Mode |
|-------------|------|----------------|-----------|-------------|
| Path | Config | `/config` | `/mnt/user/appdata/frigate` | Read/Write |
| Path | Media | `/media/frigate` | `/mnt/user/data/frigate` | Read/Write |

Create the directories if they don't exist:
```bash
mkdir -p /mnt/user/appdata/frigate
mkdir -p /mnt/user/data/frigate
```

### Device Mappings

**For NVIDIA GPUs, you do NOT need to add any device mappings.** The `--runtime=nvidia` flag combined with the environment variables handles GPU passthrough automatically.

**Do NOT add:**
- ❌ `/dev/dri` (this is for Intel/AMD GPUs using VAAPI)
- ❌ `/dev/nvidia0` or similar (handled automatically by nvidia runtime)

### Click Apply

**Do not start the container yet** - we need to create the model and config file first.

---

## Step 2: Export YOLOv9 ONNX Model

Frigate 0.16+ removed the built-in TensorRT detector. You must provide your own ONNX model.

### The Working Export Command

Run this command on your Unraid terminal:

```bash
docker run --rm -v /mnt/user/appdata/frigate:/output python:3.11 bash -c "
apt-get update && apt-get install -y libgl1 libglib2.0-0 &&
pip install ultralytics onnx onnxsim &&
yolo export model=yolov9t.pt format=onnx imgsz=320 simplify=True opset=17 &&
cp yolov9t.onnx /output/
"
```

### What This Command Does

1. **`docker run --rm`** - Creates a temporary container that auto-deletes when done
2. **`-v /mnt/user/appdata/frigate:/output`** - Maps your Frigate config folder into the container
3. **`python:3.11`** - Uses official Python 3.11 image (clean environment)
4. **`apt-get install -y libgl1 libglib2.0-0`** - Installs OpenCV dependencies (fixes `libGL.so.1` error)
5. **`pip install ultralytics onnx onnxsim`** - Installs YOLO export tools
6. **`yolo export model=yolov9t.pt format=onnx imgsz=320 simplify=True opset=17`** - Downloads YOLOv9-tiny and exports to ONNX
7. **`cp yolov9t.onnx /output/`** - Copies the model to your Frigate config folder

### CRITICAL: The opset=17 Parameter

The `opset=17` parameter is **required**. Without it, newer versions of Ultralytics export with opset 22, which causes:
```
ONNX Runtime only *guarantees* support for models stamped with official released onnx opset versions. Opset 22 is under development...
```

Frigate's ONNX runtime only supports up to opset 21. Using opset 17 ensures compatibility.

### Verify the Model Was Created

```bash
ls -la /mnt/user/appdata/frigate/yolov9t.onnx
```

You should see a file approximately 5-15MB in size.

### Alternative Model Sizes

Replace `yolov9t.pt` in the export command for different model sizes:

| Model | File | Size | Inference Speed | Accuracy |
|-------|------|------|-----------------|----------|
| YOLOv9-tiny | `yolov9t.pt` | ~5MB | ~5ms | Good |
| YOLOv9-small | `yolov9s.pt` | ~15MB | ~8ms | Better |
| YOLOv9-medium | `yolov9m.pt` | ~40MB | ~15ms | Best |

### Alternative Resolution (640x640)

For higher accuracy at the cost of slower inference:

```bash
docker run --rm -v /mnt/user/appdata/frigate:/output python:3.11 bash -c "
apt-get update && apt-get install -y libgl1 libglib2.0-0 &&
pip install ultralytics onnx onnxsim &&
yolo export model=yolov9t.pt format=onnx imgsz=640 simplify=True opset=17 &&
cp yolov9t.onnx /output/
"
```

> **Important:** If using 640x640, update `width` and `height` in your config.yml to match.

---

## Step 3: Create the Frigate Configuration

Create the config file at `/mnt/user/appdata/frigate/config.yml`:

```bash
nano /mnt/user/appdata/frigate/config.yml
```

### Complete Working Configuration

```yaml
mqtt:
  enabled: false

# =============================================================================
# MODEL CONFIGURATION
# This section defines the ONNX model parameters
# =============================================================================
model:
  model_type: yolo-generic
  width: 320
  height: 320
  input_tensor: nchw
  input_dtype: float
  path: /config/yolov9t.onnx
  labelmap_path: /labelmap/coco-80.txt

# =============================================================================
# DETECTOR CONFIGURATION
# The stable-tensorrt image automatically uses CUDA when available
# =============================================================================
detectors:
  onnx:
    type: onnx

# =============================================================================
# HARDWARE ACCELERATION
# Use NVIDIA for video decoding
# =============================================================================
ffmpeg:
  hwaccel_args: preset-nvidia-h264

# =============================================================================
# GO2RTC STREAMS
# Acts as RTSP proxy - one connection to camera, multiple clients can pull
# This prevents overwhelming the camera with multiple RTSP connections
# =============================================================================
go2rtc:
  streams:
    # Main stream for recording (4K/high quality)
    # Replace with your camera's credentials and IP
    your_camera:
      - "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/h265Preview_01_main"
    # Sub stream for detection (lower resolution)
    your_camera_sub:
      - "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/h264Preview_01_sub"

# =============================================================================
# RECORDING CONFIGURATION
# =============================================================================
record:
  enabled: true
  retain:
    days: 30
    mode: motion
  alerts:
    retain:
      days: 30
  detections:
    retain:
      days: 30

# =============================================================================
# SNAPSHOT CONFIGURATION
# =============================================================================
snapshots:
  enabled: true
  retain:
    default: 30

# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================
cameras:
  your_camera:
    enabled: true
    ffmpeg:
      inputs:
        # Main stream for 4K recording - pulls from go2rtc
        - path: rtsp://127.0.0.1:8554/your_camera
          input_args: preset-rtsp-restream
          roles:
            - record
        # Sub stream for detection - pulls from go2rtc
        - path: rtsp://127.0.0.1:8554/your_camera_sub
          input_args: preset-rtsp-restream
          roles:
            - detect
    detect:
      enabled: true
      width: 640
      height: 480
      fps: 5
    objects:
      track:
        - person
        - car
        - dog
        - cat

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
detect:
  enabled: true

version: 0.16-0

auth:
  enabled: true
```

### Configuration Explanation

#### Model Section (CRITICAL)
| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `model_type` | `yolo-generic` | **Required for YOLOv9.** Do NOT use `yolonas` - that's for a different model format. |
| `width` | `320` | Must match the `imgsz` used during export |
| `height` | `320` | Must match the `imgsz` used during export |
| `input_tensor` | `nchw` | YOLOv9 uses NCHW tensor format (batch, channels, height, width) |
| `input_dtype` | `float` | **Required for YOLOv9.** Specifies 32-bit float input. |
| `path` | `/config/yolov9t.onnx` | Path inside the container to your model |
| `labelmap_path` | `/labelmap/coco-80.txt` | Built-in COCO labels file (included in Frigate) |

#### Detector Section
```yaml
detectors:
  onnx:
    type: onnx
```

The `stable-tensorrt` image automatically detects and uses CUDA when available. You do NOT need to specify `device: /gpu` or any GPU parameters - this is handled automatically by the ONNX runtime in the tensorrt image.

#### go2rtc Section (Important for Multi-System Setups)
The go2rtc streams act as an RTSP proxy:
- Frigate connects to go2rtc at `127.0.0.1:8554`
- go2rtc maintains a single connection to each camera
- Multiple clients (Frigate recording, Frigate detection, other systems) can pull from go2rtc

**This is critical if you run other NVR software** (like AgentDVR). Have all applications pull from go2rtc instead of directly from the camera to prevent RTSP connection conflicts.

---

## Step 4: Start the Container

### Start Frigate

1. Go to **Docker** tab in Unraid
2. Click on **FrigateNVR** container
3. Click **Start**

### Verify GPU is Being Used

Check that the GPU is detected:
```bash
docker exec FrigateNVR nvidia-smi
```

You should see your GPU listed and a `frigate.detector.onnx` process using GPU memory (~300-400MB).

### Check ONNX Providers

Verify CUDA is available:
```bash
docker exec FrigateNVR python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output:
```
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

If you only see `['CPUExecutionProvider']`, something is wrong with the GPU passthrough.

### View Logs

```bash
docker logs FrigateNVR --tail 100
```

Look for:
```
ONNX: /config/yolov9t.onnx loaded
```

And confirm there are no CUDA errors.

---

## Step 5: Access and Verify

### Access the Web UI

Open in your browser:
```
http://YOUR_UNRAID_IP:5000
```

You'll be prompted to create an admin account on first login.

### Check System Stats

1. Navigate to **System** in the left sidebar
2. Look at the **Detectors** section
3. Verify inference speed is ~5ms (GPU) not ~50ms+ (CPU fallback)

### Verify Detection is Working

Walk in front of your camera. You should see:
- Bounding boxes around detected objects with confidence scores
- Events appearing in the **Review** tab

---

## Troubleshooting

### ONNX Opset Version Error

**Error:**
```
ONNX Runtime only *guarantees* support for models stamped with official released onnx opset versions. Opset 22 is under development...
```

**Solution:** Re-export the model with `opset=17` as shown in Step 2.

---

### libGL.so.1 Missing Error

**Error:**
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

**Solution:** The export command in Step 2 already includes the fix:
```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0
```

---

### FPS Limit Exceeded / Stream Dropping

**Error:**
```
your_camera exceeded fps limit. Exiting ffmpeg...
[rtsp @ ...] RTP: PT=61: bad cseq ...
```

**Causes:**
1. Camera sub-stream FPS is too high
2. Another application (e.g., AgentDVR) is fighting for the same camera streams

**Solutions:**

1. **Lower sub-stream FPS on camera** (via camera web interface):
   - Set sub-stream to 5-7 FPS
   - Set sub-stream bitrate to 512-1024 kbps

2. **Have other applications pull from go2rtc:**
   ```
   rtsp://YOUR_UNRAID_IP:8554/your_camera
   ```
   Instead of directly from the camera.

3. **Test by stopping conflicting applications:**
   ```bash
   docker stop AgentDVR
   docker restart FrigateNVR
   ```
   If Frigate stabilizes, you've found the cause.

---

### Model Not Loading (None Type Error)

**Error:**
```
ONNX: loading None
TypeError: Unable to load from type '<class 'NoneType'>'
```

**Solution:** The model path is wrong or the file doesn't exist. Verify:
```bash
docker exec FrigateNVR ls -la /config/yolov9t.onnx
```

---

### High Bandwidth Warning

**Warning:**
```
your_camera has a bandwidth of 19461.73 MB/hr which exceeds the expected maximum
```

**Solution:** Lower the sub-stream bitrate on your camera (512-1024 kbps recommended for detection).

---

### GPU Not Being Used (High Inference Time)

**Symptom:** Inference speed is 50ms+ instead of ~5ms

**Check 1:** Verify CUDA providers are available:
```bash
docker exec FrigateNVR python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

**Check 2:** Verify nvidia-smi shows the GPU:
```bash
docker exec FrigateNVR nvidia-smi
```

**Check 3:** Verify you're using the `stable-tensorrt` image, not `stable`.

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| Inference Speed (GPU) | ~5ms |
| Detection FPS | 5 FPS |
| GPU Memory Usage | ~300-400MB |
| CPU Usage | Minimal (GPU handles detection) |

---

## Summary Checklist

- [ ] Unraid 7.2.x with Nvidia-Driver plugin installed and rebooted
- [ ] `nvidia-smi` shows your GPU
- [ ] Container uses `ghcr.io/blakeblackshear/frigate:stable-tensorrt` image
- [ ] Extra Parameters: `--runtime=nvidia --shm-size=256m`
- [ ] Environment variables set: `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`, `TZ`
- [ ] Ports mapped: 5000, 8971, 8554, 8555/TCP, **8555/UDP**, 1984
- [ ] Volumes mapped: `/config`, `/media/frigate`
- [ ] **No** `/dev/dri` device mapping (not needed for NVIDIA)
- [ ] `yolov9t.onnx` model file exists at `/mnt/user/appdata/frigate/`
- [ ] Model exported with `opset=17`
- [ ] config.yml uses `model_type: yolo-generic` (not `yolonas`)
- [ ] config.yml uses `input_dtype: float`
- [ ] Container started with no errors in logs
- [ ] Inference speed ~5ms in System page

---

## Additional Resources

- [Frigate Documentation](https://docs.frigate.video/)
- [Frigate GitHub](https://github.com/blakeblackshear/frigate)
- [Frigate Object Detectors](https://docs.frigate.video/configuration/object_detectors/)
- [Ultralytics YOLOv9](https://docs.ultralytics.com/models/yolov9/)

---

## License

This guide is provided as-is for educational purposes. Frigate NVR is licensed under the MIT License.
