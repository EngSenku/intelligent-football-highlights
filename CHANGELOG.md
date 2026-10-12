# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-06-11

### Added
- Initial MVP release of Intelligent Football Highlights
- Audio-based highlight detection using Librosa (RMS, onset, sustained energy, crowd ratio)
- Video feature extraction with OpenCV (motion, optical flow, scene changes, foreground density)
- Multi-signal fusion pipeline (audio + video) for improved accuracy
- Event classification into 7 categories: goals, penalties, big chances, chances, fouls/cards, corners, general highlights
- Smart action onset refinement to improve clip boundaries
- Adaptive event limits based on match duration
- Event merging to avoid overlapping/redundant clips
- FastAPI backend with job-based async processing
- Modern web UI with drag-and-drop upload, real-time progress tracking
- Downloadable final highlights video with concatenated clips
- REST API endpoints: upload, status, results, download
- Configurable parameters via `config.py`
- FFmpeg integration for precise audio extraction, clipping, and concatenation
- Support for multiple video formats (MP4, MKV, MOV, AVI, WEBM)

### Changed
- N/A (initial release)

### Fixed
- N/A (initial release)
