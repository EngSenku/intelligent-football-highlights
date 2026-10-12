# Contributing to Intelligent Football Highlights

Thank you for your interest in contributing! This project aims to make football highlight generation more accessible and accurate. We welcome contributions from everyone.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/intelligent-football-highlights-mvp.git
   cd intelligent-football-highlights-mvp
   ```
3. **Set up the development environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Verify FFmpeg is installed**:
   ```bash
   ffmpeg -version
   ffprobe -version
   ```

## Development Workflow

### Branching

- Create a feature branch from `main`: `git checkout -b feature/your-feature-name`
- Use descriptive branch names: `feature/improve-detection`, `fix/clip-boundaries`, `docs/readme-update`

### Code Style

- Follow PEP 8 guidelines for Python code
- Use type hints where appropriate (the codebase already does this)
- Keep functions focused and well-documented with docstrings
- Use meaningful variable and function names
- Avoid adding unnecessary dependencies

### Testing

Before submitting, test your changes:

1. **Run the app**:
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Test with a small video clip** if possible - even a 30-second clip is enough to verify the pipeline runs end-to-end
3. **Check for errors** in the logs
4. **Verify the frontend** still works correctly

### Areas to Contribute

We especially welcome contributions in:

- **Detection accuracy**: Improve heuristics in `detector.py` (event classification, peak refinement)
- **Performance**: Optimize video frame processing, optical flow, or memory usage
- **UI/UX**: Enhance the frontend (better visualizations, mobile responsiveness, accessibility)
- **Configuration**: Make more parameters configurable or add presets (e.g., "aggressive" vs "conservative" modes)
- **Documentation**: Improve README, add examples, create tutorials
- **Testing**: Add unit tests for core logic (audio/video feature extraction, detection)
- **Deployment**: Docker files, docker-compose, cloud deployment guides
- **New sports**: Adapt the heuristics for other sports

## Submitting Changes

1. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "feat: improve action onset detection in clipper"
   ```
   Use conventional commit prefixes if possible: `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `chore:`

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request** on GitHub with:
   - A clear title and description of what changed and why
   - Screenshots/videos if UI changes
   - Notes about testing performed
   - Any related issue numbers (e.g., "Fixes #123")

## Code Review Process

- All PRs require at least one review
- Be responsive to feedback and questions
- Keep PRs focused - avoid bundling unrelated changes
- Update documentation if you change behavior or add features

## Reporting Issues

Found a bug or have a suggestion? Please open an issue on GitHub with:

- **Bug reports**: Include steps to reproduce, expected vs actual behavior, logs, video format details (codec, resolution, duration)
- **Feature requests**: Describe the use case, proposed solution, and why it would be useful
- **Questions**: Feel free to ask - we're happy to help!

## Development Notes

### Important Files

- `backend/detector.py`: Core detection logic - tweak thresholds here for accuracy
- `backend/config.py`: Global configuration - weights, durations, limits
- `backend/video_features.py`: Video analysis - optical flow is most CPU-intensive
- `backend/processor.py`: Orchestrates the full pipeline

### Performance Tips

- Use `FRAME_SAMPLE_FPS=1.0` for very long matches or slower machines
- The optical flow only runs on high-motion windows (optimization already in place)
- Processing is single-threaded by default (`MAX_WORKERS=1`) for memory stability

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
