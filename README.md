# Intelligent Football Highlights

An AI-powered system that automatically detects and extracts the most exciting moments from full football match videos. Upload a complete game and get a condensed highlights reel in minutes.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## About

This project uses audio and video analysis to intelligently identify highlight-worthy moments in football matches - including goals, penalties, big chances, shots, fouls, corners, and general exciting play. Instead of manually scrubbing through 90+ minutes of footage, the system analyzes crowd reactions, commentary intensity, motion patterns, scene changes, and on-field activity to automatically clip and compile the key moments.

The system employs a **fused audio-video heuristic pipeline (v6)** that combines:
- **Audio analysis** (Librosa): Detects crowd roars, sustained excitement, spikes in energy, and vocal intensity
- **Video analysis** (OpenCV): Tracks motion, optical flow, scene changes, and foreground density (player activity)
- **Smart event classification**: Distinguishes between different types of events (goals, penalties, chances, fouls, corners)
- **Adaptive clipping**: Dynamically adjusts pre/post-roll based on event type and when the action actually begins

## ✨ Features

- **🎯 Intelligent Detection**: Multi-signal fusion detects goals, penalties, big chances, regular chances, fouls/cards, corners, and general highlights
- **🎬 Smart Clip Boundaries**: Automatically finds action onset (not just peak noise) to create cleaner, more watchable clips
- **⚡ Adaptive Highlight Limits**: Adjusts max events based on match duration (more clips for longer games, fewer for short clips)
- **🔀 Smart Merging**: Merges overlapping or closely spaced events to avoid duplicate/redundant clips
- **🎨 Modern Web UI**: Clean, responsive interface with drag-and-drop upload, real-time progress tracking, and beautiful results visualization
- **📊 Detailed Insights**: See event timestamps, confidence scores, event types, and compression ratio for your highlights
- **⚙️ Fully Configurable**: Tune detection sensitivity, weights, clip durations, and more via `config.py`
- **💾 Job-Based Processing**: Async background processing with status polling - upload large videos without timeouts

## 🏗️ Architecture

```
intelligent-football-highlights/
├── backend/
│   ├── main.py           # FastAPI app - upload, status, results, download endpoints
│   ├── processor.py      # Core orchestration pipeline
│   ├── audio.py          # Audio feature extraction (RMS, onset, crowd energy)
│   ├── video_features.py # Video analysis (motion, optical flow, scene changes, density)
│   ├── detector.py       # Event detection & classification with multi-signal fusion
│   ├── clipper.py        # Clip window building, merging, extraction
│   ├── ffmpeg_tools.py   # FFmpeg/FFprobe wrappers for video/audio processing
│   ├── jobs.py           # In-memory job tracking & background processing
│   ├── config.py         # All tunable parameters & paths
│   └── utils.py          # Utility helpers
├── frontend/
│   ├── index.html        # Web UI
│   ├── style.css         # Modern dark theme styling
│   └── app.js            # Frontend logic & API calls
├── data/
│   ├── uploads/          # Uploaded videos (temporary)
│   ├── temp/             # Intermediate files (WAV, concat lists)
│   └── outputs/          # Generated clips + final highlights video
└── requirements.txt
```

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) |
| **Audio Analysis** | [Librosa](https://librosa.org/), [SoundFile](https://pysoundfile.readthedocs.io/) |
| **Computer Vision** | [OpenCV](https://opencv.org/) (frame differencing, optical flow, histogram comparison) |
| **Numerical** | [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) |
| **Video Processing** | [FFmpeg](https://ffmpeg.org/) (audio extraction, precise clipping, concatenation) |
| **Frontend** | Vanilla HTML/CSS/JavaScript (no framework dependencies) |

## 📋 Prerequisites

Before running this project, make sure you have:

1. **Python 3.9+** installed
2. **FFmpeg** installed and available in your PATH
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows (via Chocolatey)
   choco install ffmpeg
   ```
3. **Sufficient disk space** - Videos are processed temporarily; a 90-min match can use ~2-5GB during processing
4. **Decent CPU/RAM** - Video feature extraction (optical flow) is CPU-intensive but optimized to sample at 2 FPS by default

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/EngSenku/intelligent-football-highlights-mvp.git
cd intelligent-football-highlights-mvp
```

### 2. Create a virtual environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using uv
uv venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `librosa` and `opencv-python` can take a few minutes to install. If you're on Linux, you may also need system packages for audio/video codecs.

## 💻 Usage

### Start the backend server

```bash
# From the project root
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Interactive API docs (Swagger): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Open the frontend

Simply open `frontend/index.html` in your browser, or serve it with any static server:

```bash
# Using Python
python -m http.server 8080 --directory frontend
```

Then navigate to http://localhost:8080

### Process a video

1. Open the web UI in your browser
2. Drag & drop your football match video (MP4, MKV, MOV, AVI, WEBM supported)
3. Click "Analyse Match" to start processing
4. Watch real-time progress as it extracts audio, analyzes frames, detects events, and clips highlights
5. Once complete, view detected events with timestamps and confidence scores
6. Download the final highlights MP4

**Processing time**: Expect ~10-20% of video duration for a typical match (varies by CPU, resolution). A 90-min match usually takes ~10-15 minutes.

## 🔧 Configuration

All detection parameters are tunable in [`backend/config.py`](backend/config.py). Key settings:

| Parameter | Default | Description |
|---|---|---|
| `FRAME_SAMPLE_FPS` | 2.0 | Frames sampled per second (lower = faster, less accurate; higher = slower, more accurate) |
| `WEIGHT_AUDIO` | 0.55 | Weight of audio cues in fusion |
| `WEIGHT_MOTION` | 0.14 | Weight of motion detection |
| `WEIGHT_FLOW` | 0.08 | Weight of optical flow |
| `WEIGHT_DENSITY` | 0.11 | Weight of foreground/player density |
| `WEIGHT_SCENE` | 0.12 | Weight of scene changes |
| `PEAK_PERCENTILE` | 75.0 | Percentile threshold for peak detection |
| `PEAK_DISTANCE_SECONDS` | 18 | Minimum gap between detected events |
| `CLIP_PRE_SECONDS` | 35 | Default seconds before peak |
| `CLIP_POST_SECONDS` | 35 | Default seconds after peak |
| `MAX_CLIP_DURATION_SECONDS` | 130 | Cap for merged long events |

Event-specific clip rules (per event type) are defined in [`backend/detector.py`](backend/detector.py) under `CLIP_RULES`.

## 📊 How It Works

The pipeline processes videos in stages:

1. **Metadata Extraction** (FFprobe) - Gets duration, FPS, audio presence
2. **Audio Extraction** (FFmpeg) - Extracts mono WAV at 22.05kHz
3. **Audio Analysis** - Computes RMS energy, onset strength, sustained energy (15s/30s windows), crowd ratio (high-frequency energy), spikes, and energy deltas
4. **Video Frame Sampling** (OpenCV) - Samples at 2 FPS to balance accuracy vs. performance
5. **Video Feature Extraction**:
   - **Motion**: Grayscale frame differencing (captures general activity)
   - **Scene Changes**: Histogram Bhattacharyya distance (replays, cuts, crowd shots)
   - **Optical Flow**: Lucas-Kanade sparse flow on high-motion windows (tracks player movement intensity)
   - **Foreground Density**: Non-green pixel ratio in central ROI (estimates on-field activity)
6. **Multi-Signal Fusion** - Normalizes and weights all signals into a unified excitement score
7. **Peak Detection** - Finds local maxima using adaptive percentile + std-dev thresholds
8. **Event Classification** - Classifies peaks into event types using heuristics based on sustained energy, crowd reaction, motion, density, and scene changes
9. **Action Onset Refinement** - Looks backward from peaks to find when action actually started (reduces "late" clips)
10. **Clip Window Building** - Creates windows with event-type-specific pre/post durations
11. **Smart Merging** - Merges events within `MERGE_EVENTS_CLOSER_THAN_SECONDS` to avoid fragmentation
12. **Clip Extraction** (FFmpeg) - Extracts individual highlight clips with re-encoding for smooth cuts
13. **Concatenation** (FFmpeg) - Stitches all clips into final `final_highlights.mp4` with faststart

## 🎯 Event Types

| Event Type | Typical Triggers | Clip Style |
|---|---|---|
| `goal_candidate` | High sustained energy + strong crowd reaction + elevated density | Longer (55-95s), captures buildup + celebration |
| `penalty_candidate` | Very high sustained + crowd + density | Longest (90-125s) |
| `big_chance_candidate` | Strong audio + high motion/density + some sustained energy | Medium (35-65s) |
| `chance_candidate` | Good audio/motion balance | Short-medium (28-55s) |
| `foul_or_card_candidate` | Audio spike + scene change (stop in play) | Short (22-42s) |
| `corner_or_set_piece_candidate` | High density + scene change | Short (32-55s) |
| `highlight_candidate` | General excitement not matching specific patterns | Compact (22-40s) |

## 🧪 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload a video file. Returns `job_id` and starts background processing |
| `GET` | `/status/{job_id}` | Get processing status (queued/running/completed/failed), current stage, progress % |
| `GET` | `/results/{job_id}` | Get full results JSON (events, metadata, durations, compression ratio) |
| `GET` | `/download/{job_id}` | Download the final highlights MP4 |

Full API schema available at http://localhost:8000/docs when running.

## ⚠️ Limitations

- **Heuristic-based, not ML**: Uses signal processing + rules rather than trained models. Detection quality depends on broadcast audio mix (some stadiums/mixes are quieter)
- **English/Global broadcast assumptions**: Tuned for typical TV football broadcasts with commentary + crowd mics
- **CPU-intensive**: Optical flow on high-motion frames is the main bottleneck. Reduce `FRAME_SAMPLE_FPS` to 1.0 if running on slower hardware
- **No player tracking/ball detection**: Uses activity proxies (motion, density) rather than object detection
- **Single-camera broadcast only**: Works best with standard TV coverage, not multi-angle raw footage
- **Memory usage**: While frames aren't all kept in memory at once (streaming), 4K/very high res + high FPS sampling can be demanding

## 🗂️ Data & Git Hygiene

- `data/uploads/`, `data/temp/`, `data/outputs/` contain runtime artifacts. The `.gitignore` excludes all generated files except `.gitkeep` placeholders to preserve directory structure.
- Large video files (especially `final_highlights.mp4`) should **never** be committed to git. If you have processed outputs locally, they're expected to be git-ignored.
- If you want to share a sample, host it externally (e.g., GitHub Releases, Drive) and link it in docs.

## 🛣️ Roadmap Ideas

- [ ] Add ML-based event detection (e.g., YOLO for goals/players, audio classification models)
- [ ] Support custom highlight durations per user preference
- [ ] Add timeline visualization with scrubbing
- [ ] Export individual clips as ZIP
- [ ] Add Docker support for easier deployment
- [ ] GPU acceleration for optical flow (CUDA)
- [ ] Multi-language commentary handling
- [ ] Configurable API base URL in frontend (env-based)
- [ ] Add authentication for production deployments
- [ ] Support more sports (basketball, hockey, rugby)

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, improved heuristics, UI enhancements, or documentation - feel free to open an issue or submit a PR.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Librosa](https://librosa.org/) for audio feature extraction tools
- [OpenCV](https://opencv.org/) for computer vision primitives
- [FFmpeg](https://ffmpeg.org/) for robust video processing
- [FastAPI](https://fastapi.tiangolo.com/) for the clean API framework

## 📧 Contact

Created by [EngSenku](https://github.com/EngSenku)  
Project Link: [https://github.com/EngSenku/intelligent-football-highlights-mvp](https://github.com/EngSenku/intelligent-football-highlights-mvp)
