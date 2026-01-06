# Frigate NVR with NVIDIA GPU on Unraid 7.x

## What is Frigate and Why Use It?

Frigate is an open-source Network Video Recorder (NVR) built specifically for real-time AI object detection. Unlike traditional NVRs that simply record everything or use basic motion detection, Frigate analyzes your camera feeds and can distinguish between a person walking up your driveway, a car pulling in, your dog running across the yard, or just a tree branch swaying in the wind.

**Why this matters for your home security:**

- **Smarter alerts** – Get notified when a person or vehicle appears, not every time a shadow moves
- **Searchable footage** – Find "all clips with a person in the driveway from last Tuesday" in seconds
- **Reduced storage** – Record only events that matter, or keep full recordings but highlight what's important
- **Home Assistant integration** – Trigger automations based on specific objects (turn on lights when a person is detected, alert when an unknown car enters)
- **Local processing** – Your camera feeds never leave your network; no cloud subscriptions required

**Why GPU acceleration?**

Object detection is computationally expensive. Running it on your CPU means:
- High CPU usage (often 50-100% per camera)
- Limited to 1-2 cameras before performance degrades
- Slower inference times (50-100ms+), which can miss fast-moving objects

With an NVIDIA GPU handling detection:
- ~5ms inference time (10-20x faster)
- Minimal CPU impact
- Scale to many cameras without breaking a sweat
- Your CPU stays free for other tasks

---

## Why This Guide Exists (The Hard Way)

Getting NVIDIA GPU acceleration working in Frigate on Unraid *should* be straightforward. The Frigate docs mention ONNX support for NVIDIA GPUs. Just configure a detector and go, right?

I spent two full days discovering how wrong that assumption was.

**The model export nightmare:**

Frigate 0.16 removed the built-in TensorRT detector and no longer ships with pre-built models. You need to bring your own. The documentation points to a Google Colab notebook for exporting models—which runs painfully slow on free-tier GPUs and times out constantly. Fine, I'll build locally.

The official Docker build command for YOLOv9? Failed immediately with `requirements.txt not found`. The ultralytics export command? Worked, but produced a model with ONNX opset 22. Frigate's runtime only supports up to opset 21. The error message? A vague warning buried in startup logs that took an hour to track down.

**The configuration maze:**

Once I had a working model, Frigate wouldn't load it. Turns out the `model` configuration section is completely separate from the `detectors` section—something the docs gloss over. I tried `model_type: yolonas` because YOLOv9 seemed close to YOLO-NAS. Wrong. It needs `yolo-generic`. I tried omitting `input_dtype`. The model loaded but produced garbage detections. It needs `input_dtype: float` explicitly set.

**The stream conflict disaster:**

Model finally loaded. Detector running at 5ms on GPU. Victory! Then Frigate started crash-looping with `exceeded fps limit` errors. Cameras would connect, run for 30 seconds, then die. The culprit? AgentDVR was also pulling from the same cameras. Most IP cameras only support 2-3 simultaneous RTSP connections. Frigate uses two streams per camera (main for recording, sub for detection). Add another NVR and you're over the limit.

The solution—go2rtc as an RTSP proxy—is mentioned in the Frigate docs but presented as optional. It's not optional if you run multiple systems. And configuring cameras to pull from `127.0.0.1:8554` instead of the camera IP directly isn't obvious.

**The image tag gotcha:**

Even with everything else perfect, inference was running at 50ms—CPU speeds, not GPU. Turns out the standard `frigate:stable` image doesn't include CUDA libraries. You need `frigate:stable-tensorrt`. This is mentioned in the docs, but it's easy to miss when you're following an Unraid Community Apps template that defaults to the wrong image.

**Two days. Countless container restarts. Multiple model formats. Stack traces that led nowhere.**

This guide is everything I learned, distilled into a reproducible process. Every step has been tested. Every error message has a solution. The ONNX export command works. The configuration is complete and commented. The go2rtc setup prevents stream conflicts.

Save yourself the two days. Here's what actually works.

---

## What This Guide Covers
