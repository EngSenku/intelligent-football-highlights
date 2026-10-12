# Intelligent Football Highlights

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-0.10+-orange?style=flat-square)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-007808?style=flat-square&logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An AI-powered system that automatically detects and extracts the most exciting moments from full football match videos. Upload a complete game and get a condensed highlights reel in minutes.

Ce projet a été réalisé dans le cadre du module Traitement d'Image à l'École Nationale des Sciences Appliquées de Tanger (ENSAT), sous l'encadrement de M. LACHKAR ABDELMOUNAIM.

## About


It uses audio and video analysis to intelligently identify highlight-worthy moments in football matches - including goals, penalties, big chances, shots, fouls, corners, and general exciting play. Instead of manually scrubbing through 90+ minutes of footage, the system analyzes crowd reactions, commentary intensity, motion patterns, scene changes, and on-field activity to automatically clip and compile the key moments.

The system employs a fused audio-video heuristic pipeline that combines:
- Audio analysis (Librosa): Detects crowd roars, sustained excitement, spikes in energy, and vocal intensity
- Video analysis (OpenCV): Tracks motion, optical flow, scene changes, and foreground density (player activity)
- Smart event classification: Distinguishes between different types of events
- Adaptive clipping: Dynamically adjusts pre/post-roll based on event type and when the action actually begins

## Features

- Intelligent detection of highlight-worthy moments using audio and video fusion
- Smart clip boundary refinement to capture action onset
- Adaptive event limits based on match duration
- Smart merging of overlapping events
- Web-based interface with real-time progress tracking
- Detailed event information with timestamps and confidence scores
- Support for common video formats

## Architecture

The project is structured as follows:

```
intelligent-football-highlights/
├── backend/
│   ├── main.py           # FastAPI application
│   ├── processor.py      # Core processing pipeline
│   ├── audio.py          # Audio feature extraction
│   ├── video_features.py # Video analysis
│   ├── detector.py       # Event detection and classification
│   ├── clipper.py        # Clip generation
│   ├── ffmpeg_tools.py   # Video processing utilities
│   ├── jobs.py           # Job management
│   ├── config.py         # Configuration
│   └── utils.py          # Helper functions
├── frontend/
│   ├── index.html        # User interface
│   ├── style.css         # Styling
│   └── app.js            # Frontend logic
├── data/
│   ├── uploads/          # Uploaded videos
│   ├── temp/             # Temporary files
│   └── outputs/          # Generated highlights
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.9 or higher
- FFmpeg installed and available in PATH

### Setup

1. Clone the repository:
```bash
git clone https://github.com/EngSenku/intelligent-football-highlights.git
cd intelligent-football-highlights
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the backend server:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

2. Open the frontend by opening `frontend/index.html` in a web browser, or serve it with:
```bash
python -m http.server 8080 --directory frontend
```

3. Upload your football match video, wait for processing to complete, and download the generated highlights.

## Topics
- football-highlights
- video-processing
- audio-analysis
- computer-vision
- fastapi
- opencv
- librosa
- ffmpeg
- sports-analytics
- highlight-detection

## Topics
- football-highlights
- video-processing
- audio-analysis
- computer-vision
- fastapi
- opencv
- librosa
- ffmpeg
- sports-analytics
- highlight-detection

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
