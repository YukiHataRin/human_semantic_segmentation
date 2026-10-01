# Human Mask Studio

Local browser tool for visualizing YOLO person instance-segmentation masks on images and videos, with real-time multi-person ID tracking.

## Run it

Terminal 1 (API; first run downloads `yolo11n-seg.pt`):

```bash
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --reload --port 8001
```

The API uses `ffmpeg` to produce browser-compatible H.264 video, so it must be available on `PATH`.

Terminal 2 (UI):

```bash
npm install
npm run dev
```

Open the URL Vite prints (normally `http://localhost:5173`). Upload an image or video. The service only selects COCO class `person`, then blends each individual mask into the original media.

## Current scope

- Image: uploads and returns a YOLO-masked result immediately.
- Video: processes the selected video frame-by-frame then shows a masked MP4.
- Video tracking: BoT-SORT uses lightweight ReID embeddings to keep a stable ID and color for each person.
- Camera input is intentionally not included yet.

For long videos, this first prototype waits until processing finishes; streaming progress and webcam input are the next upgrades.
