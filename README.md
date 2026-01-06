# Frigate NVR with NVIDIA GPU on Unraid

## The Story

This guide exists because I spent two days and nothing worked... right up until it did. 

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

### Tested Configuration

| Component | Details |
|-----------|---------|
| Unraid Version | 7.2.x |
| Frigate Version | 0.16.3 (tensorrt)|
| GPU | NVIDIA RTX 5060 Ti (Blackwell) |
| CPU | AMD Ryzen 9 7950X |
| Container Image | `ghcr.io/blakeblackshear/frigate:stable-tensorrt` |

> **Note:** This guide should work with other NVIDIA GPUs (RTX 20/30/40/50 series). The key requirement is CUDA support.

---

## Prerequisites

1. **Unraid with NVIDIA Plugin**
   - Install the "Nvidia-Driver" plugin from Community Applications
   - Reboot Unraid after installation
   - Verify GPU is detected: `nvidia-smi`

2. **Docker configured for NVIDIA runtime**
   - The Nvidia-Driver plugin typically configures this automatically

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

### Environment Variables

Click **"Add another Path, Port, Variable, Label or Device"** and select **Variable** for each:

| Config Type | Name | Key | Value |
|-------------|------|-----|-------|
| Variable | NVIDIA Devices | `NVIDIA_VISIBLE_DEVICES` | `all` |
| Variable | NVIDIA Capabilities | `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,video` |
| Variable | Timezone | `TZ` | Your timezone (e.g., `America/New_York`) |

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

> **Important:** Port 8555 must be added twice - once for TCP and once for UDP. WebRTC requires both protocols.

### Volume Mappings

Click **"Add another Path, Port, Variable, Label or Device"** and select **Path** for each:

| Config Type | Name | Container Path | Host Path | Access Mode |
|-------------|------|----------------|-----------|-------------|
| Path | Config | `/config` | `/mnt/user/appdata/frigate` | Read/Write |
| Path | Media | `/media/frigate` | `/mnt/user/data/frigate` | Read/Write |

> Create the media directory if it doesn't exist: `mkdir -p /mnt/user/data/frigate`

### Device Mappings

**For NVIDIA GPUs, you do NOT need to add any device mappings.** The `--runtime=nvidia` flag with the environment variables handles GPU passthrough automatically.

Do NOT add:
- ❌ `/dev/dri` (this is for Intel/AMD GPUs)
- ❌ `/dev/nvidia0` or similar (handled by runtime)

### Click Apply

Do not start the container yet - we need to create the model and config file first.

---

## Step 2: Export YOLOv9 ONNX Model

Frigate 0.16+ requires you to provide your own ONNX model. The built-in TensorRT detector has been removed.

### Export Command

Run this command on your Unraid terminal to generate the YOLOv9-tiny model:

```bash
docker run --rm -v /mnt/user/appdata/frigate:/output python:3.11 bash -c "
apt-get update && apt-get install -y libgl1 libglib2.0-0 &&
pip install ultralytics onnx onnxsim &&
yolo export model=yolov9t.pt format=onnx imgsz=320 simplify=True opset=17 &&
cp yolov9t.onnx /output/
"
```

#### What this does:
1. Creates a temporary Python container
2. Installs required dependencies (libGL for OpenCV)
3. Downloads YOLOv9-tiny pretrained weights
4. Exports to ONNX format at 320x320 resolution with opset 17
5. Copies the model to your Frigate config folder
6. Auto-deletes the temporary container

#### Critical: ONNX Opset Version

The `opset=17` parameter is **required**. Frigate's ONNX runtime only supports up to opset 21, and newer versions of Ultralytics default to opset 22 which causes the error:
```
ONNX Runtime only *guarantees* support for models stamped with official released onnx opset versions. Opset 22 is under development...
```

### Alternative Model Sizes

Replace `yolov9t.pt` in the export command:

| Model | File | Size | Speed |
|-------|------|------|-------|
| YOLOv9-tiny | `yolov9t.pt` | ~5MB | Fastest |
| YOLOv9-small | `yolov9s.pt` | ~15MB | Balanced |
| YOLOv9-medium | `yolov9m.pt` | ~40MB | More accurate |

### Alternative Resolution (640x640)

For higher accuracy (slower inference), change `imgsz=320` to `imgsz=640` and update your config to match.

---

## Step 3: Create the COCO Labels File

Create the labelmap file at `/mnt/user/appdata/frigate/coco-80.txt`:

```
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
```

---

## Step 4: Configure Frigate

Create or edit `/mnt/user/appdata/frigate/config.yml`:

```yaml
mqtt:
  enabled: false

model:
  model_type: yolo-generic
  width: 320
  height: 320
  input_tensor: nchw
  input_dtype: float
  path: /config/yolov9t.onnx
  labelmap_path: /labelmap/coco-80.txt

detectors:
  onnx:
    type: onnx

ffmpeg:
  hwaccel_args: preset-nvidia-h264

go2rtc:
  streams:
    your_camera:
      - "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/h265Preview_01_main"
    your_camera_sub:
      - "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/h264Preview_01_sub"

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

snapshots:
  enabled: true
  retain:
    default: 30

cameras:
  your_camera:
    enabled: true
    ffmpeg:
      inputs:
        - path: rtsp://127.0.0.1:8554/your_camera
          input_args: preset-rtsp-restream
          roles:
            - record
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

detect:
  enabled: true

version: 0.16-0

auth:
  enabled: true
```

### Configuration Notes

#### Model Section
- `model_type: yolo-generic` - Required for YOLOv9 (not `yolonas`)
- `input_dtype: float` - Required for YOLOv9
- `width/height: 320` - Must match the export resolution

#### go2rtc Streams
- Acts as an RTSP proxy - one connection to camera, multiple clients can pull from go2rtc
- Prevents overwhelming the camera with multiple simultaneous RTSP connections
- **Critical for multi-system setups** - if you run other NVR software (like AgentDVR), have them pull from go2rtc instead of directly from the camera

#### Camera Stream Separation
- **Main stream** (`your_camera`): High resolution (4K) for recording only
- **Sub stream** (`your_camera_sub`): Lower resolution for detection only
- This separation means detection settings don't affect recording quality

---

## Step 5: Start Frigate

```bash
docker start FrigateNVR
```

### Verify GPU Detection

Check that the GPU is being used:

```bash
docker exec FrigateNVR nvidia-smi
```

You should see `frigate.detector.onnx` in the process list using GPU memory (~300-400MB).

### Check Available ONNX Providers

```bash
docker exec FrigateNVR python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output:
```
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### View Logs

```bash
docker logs FrigateNVR --tail 100
```

Look for:
```
ONNX: /config/yolov9t.onnx loaded
```

---

## Step 6: Verify Operation

### Access the Web UI

Open `http://YOUR_UNRAID_IP:5000` in your browser.

### Check System Stats

1. Navigate to **System** in the left sidebar
2. Look at the **Detectors** section
3. Verify inference speed (~5ms with GPU)

### Test Detection

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

**Solution:** Re-export the model with `opset=17` parameter as shown in Step 2.

### libGL.so.1 Missing Error

**Error:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solution:** The export command in Step 2 already includes the fix:
```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0
```

### FPS Limit Exceeded / Stream Dropping

**Error:**
```
your_camera exceeded fps limit. Exiting ffmpeg...
[rtsp @ ...] RTP: PT=61: bad cseq ...
```

**Causes:**
1. Camera sub-stream FPS too high
2. Another application (e.g., AgentDVR) competing for the camera's RTSP streams

**Solutions:**
1. Lower the sub-stream FPS on the camera itself via its web interface (5-7 FPS recommended)
2. Have other applications pull from go2rtc instead of directly from the camera:
   ```
   rtsp://YOUR_UNRAID_IP:8554/your_camera
   ```
3. Stop conflicting applications to test:
   ```bash
   docker stop AgentDVR
   docker restart FrigateNVR
   ```

### Model Not Loading (None Type Error)

**Error:**
```
ONNX: loading None
TypeError: Unable to load from type '<class 'NoneType'>'
```

**Solution:** Ensure the model path is correct and the file exists:
```bash
ls -la /mnt/user/appdata/frigate/yolov9t.onnx
```

### High Bandwidth Warning

**Warning:**
```
your_camera has a bandwidth of 19461.73 MB/hr which exceeds the expected maximum
```

**Solution:** Lower the sub-stream bitrate on your camera (512-1024 kbps recommended for detection).

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| Inference Speed (GPU) | ~5ms |
| Inference Speed (CPU) | ~6-10ms |
| Detection FPS | 5 FPS |
| GPU Memory Usage | ~300-400MB |

---

## Additional Resources

- [Frigate Documentation](https://docs.frigate.video/)
- [Frigate GitHub](https://github.com/blakeblackshear/frigate)
- [Frigate Object Detectors](https://docs.frigate.video/configuration/object_detectors/)
- [Ultralytics YOLOv9](https://docs.ultralytics.com/models/yolov9/)

---

## License

This guide is provided as-is for educational purposes. Frigate NVR is licensed under the MIT License.

---

## Contributing

Found an issue or have an improvement? Feel free to open a pull request or issue.
