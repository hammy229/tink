# Changelog

## 2.0.0

### Added

- The whole app runs in the browser with no server: ONNX in WebAssembly in a worker, hash-pinned and cached.
- Hands-free listening with voice activity detection that learns the room's noise floor.
- A WebSocket stream for devices with no screen, and an ONNX serving backend that doesn't need torch.
- A benchmark suite on ReCANVo and TORGO with paired comparisons, person-level bootstrap, an exact sign-flip test, calibration, and conformal checks.
- A model gate that re-scores the benchmarks with the exact shipped file.
- End-to-end tests with the real model in Chromium, parity tests between Python and JS, a load test, and CI.
- A model card and architecture decision records.

### Fixed

- Inference ran on the event loop, so one person's model call froze the server for everyone.
- A crash during a write could corrupt a person's profile.
- A small compressed upload could decode to hours of audio and exhaust memory.
- Forged `X-Forwarded-For` headers could dodge the rate limit, and its memory grew without bound.

### Changed

- Restarts reuse cached embeddings instead of re-embedding every take.
- Profile links use the URL fragment, so they stay out of server logs.

### Removed

- Learning from confirmed sounds, added and removed in this release. In a corrected simulation it hurt deliberate sounds by 7.0 points. See ADR 5.

## 1.0.0

The HackTX 26 project (October 2026). Hackathon documents are in `docs/archive`.
