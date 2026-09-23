# OmniSign Dataset Collection Prototype

This repository contains an early data-collection utility developed for the
OmniSign sign-language recognition research project. It captures webcam frames,
extracts MediaPipe hand landmarks, records session metadata, and applies basic
image augmentation.

OmniSign is an actively developed research and engineering platform focused on
Lebanese Sign Language, with basic American Sign Language experiments. The
canonical project history, team attribution, current scope, and limitations are
documented in the [OmniSign case study](https://laythayache.com/projects/omnisign/).

## What this repository contains

- A Tkinter interface for guided capture sessions
- Webcam capture with OpenCV
- Hand-landmark extraction with MediaPipe
- Session-oriented file and metadata organization
- Validation and augmentation utilities

The repository does **not** contain the full OmniSign application, the trained
production models, or the community-collected Lebanese Sign Language dataset.
Its file count must not be used as evidence for OmniSign's dataset size.

## Status and limitations

This is a historical research prototype, not a maintained collection service.
The code includes local path assumptions and experimental workflows that need
review before reuse. It does not itself implement participant consent,
de-identification, access control, retention policy, or secure dataset storage.

Anyone adapting the collector is responsible for an appropriate collection
protocol, informed participant consent, data minimization, secure storage,
access controls, retention and deletion procedures, and any required ethical or
legal review. Participant recordings should not be committed to a public
repository.

## Local setup

The prototype was written for Python with OpenCV, MediaPipe, NumPy, Pillow,
Tkinter, and Albumentations. Review the paths and configuration in the scripts
before running them; the current code should not be treated as a turnkey
installer.

```bash
python -m pip install -r requirements.txt
python main.py
```

## Public evidence

- [OmniSign case study and engineering scope](https://laythayache.com/projects/omnisign/)
- [Rafik Hariri University award announcement](https://www.rhu.edu.lb/media-room/news/rhu-team-wins-public-choice-award-at-national-fyp-demo-day-2025)

## License

No license file is currently included. Public availability does not by itself
grant permission to reuse the code or other repository content.
