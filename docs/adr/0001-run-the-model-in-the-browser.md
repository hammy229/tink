# 1. Run the model in the browser

**Status:** accepted

## Context

Nobody could try Tink without cloning the repo and downloading 360 MB, as the torch app doesn't fit on free hosting. The site just gave instructions. There's also a privacy reason: this is the voice of someone with a disability, and the best upload is no upload.

## Decision

Export the encoder to ONNX, quantize it to int8, and run it with ONNX Runtime Web in a Web Worker when no server answers. The server stays too, for hosted use and for streaming from devices with no screen. The browser engine answers the same API as the server, so the UI barely changes.

## Why not the other options

- **A paid GPU server:** it costs money forever for a portfolio project, and it means uploading voices.
- **Sending clips to a hosted inference API:** same privacy problem, and it adds a dependency that can disappear.
- **Transformers.js:** it wraps the same runtime. I only need one model and one tensor, so the wrapper adds little.
- **WebGPU:** dynamic int8 models don't run well on it, and WASM was fast enough in testing.

## Consequences

- There are now two implementations of scoring, one in Python and one in JS, which is the main risk. A fixture generated from the Python code is checked by both test suites, so neither can change alone.
- The download is 122 MB, once. It starts only when someone asks, and the Cache API keeps it for later visits.
- The model comes from another host, so the browser checks its SHA-256 before loading it, and a test checks that a wrong hash is refused.
- Threads need a cross-origin isolated page, so `vercel.json` sets COOP and COEP. COEP is `credentialless` so the icon CDN keeps working.
- The end-to-end test serves the same headers as production. That paid off right away: it caught my first content security policy blocking the runtime.
