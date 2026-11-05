// Holds the browser engine to what the Python code does. Run: node --test web/engine/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { MODEL } from "./config.js";
import * as core from "./core.js";

const fixture = JSON.parse(readFileSync(new URL("../../tests/fixtures/parity.json", import.meta.url)));
const close = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
const NOW = Date.UTC(2026, 10, 5, 14, 0);
const pendingAt = (p) => (p ? { options: p.options, i: p.i, at: NOW + p.at_seconds * 1000 } : null);

test("scores match the Python store", () => {
  const { vecs, query, scores } = fixture.scoring;
  const out = core.scores(vecs, query);
  assert.deepEqual(Object.keys(out), Object.keys(scores));
  for (const n of Object.keys(scores)) close(out[n], scores[n], 0.0015, n);
});

test("self-calibration matches the Python store", () => {
  const out = core.calibrate(fixture.scoring.vecs), want = fixture.scoring.calibrate;
  assert.equal(out.ready, want.ready);
  assert.equal(out.n, want.n);
  assert.equal(out.right, want.right);
  assert.equal(out.sure_n, want.sure_n);
  assert.equal(out.sure_right, want.sure_right);
  close(out.temp, want.temp, 1e-3, "temp");
  close(out.threshold, want.threshold, 2e-3, "threshold");
});

test("calibration waits for enough takes", () => {
  assert.equal(core.calibrate({ A: [[[1, 0]]], B: [[[0, 1]]] }).ready, false);
});

test("posterior matches the Python code", () => {
  for (const { args, out } of fixture.posterior) {
    const [sc, th, usage, cues, seen, hour, temp] = args;
    const got = core.posterior(sc, th, usage, cues, seen, hour, temp);
    assert.equal(got.part_of_day, out.part_of_day);
    assert.deepEqual(got.cues, out.cues);
    assert.deepEqual(got.ranked.map((r) => r.name), out.ranked.map((r) => r.name));
    got.ranked.forEach((r, i) => {
      close(r.p, out.ranked[i].p, 0.0015, `p ${r.name}`);
      close(r.audio_p, out.ranked[i].audio_p, 0.0015, `audio_p ${r.name}`);
      close(r.prior, out.ranked[i].prior, 0.0015, `prior ${r.name}`);
    });
  }
});

test("every turn decision matches the Python code", () => {
  fixture.turn.forEach((c, i) => {
    const sounds = { phrase: c.sounds.phrase, roles: c.sounds.roles, urgent: new Set(c.sounds.urgent) };
    const pending = pendingAt(c.pending);
    const o = core.decideWithoutWording(c.ranked, sounds, pending, NOW) ?? core.finishTurn(c.ranked, sounds, NOW);
    assert.deepEqual(o.fields, c.fields, `case ${i}`);
    assert.deepEqual(o.pending, pendingAt(c.next_pending), `case ${i} pending`);
    assert.equal(o.said, c.said, `case ${i} said`);
  });
});

test("resampling matches scipy.signal.resample_poly", () => {
  const wav = Float32Array.from(fixture.audio.wav);
  for (const [ratio, want] of Object.entries(fixture.audio.resample)) {
    const [up, down] = ratio.split("/").map(Number);
    const got = core.resamplePoly(wav, up, down);
    assert.equal(got.length, want.length, ratio);
    got.forEach((v, i) => close(v, want[i], 2e-3, `${ratio}[${i}]`));
  }
});

test("trim and silence checks match", () => {
  const { trim_len, trim_input_len, short_trim_len, silent } = fixture.audio;
  const padded = new Float32Array(trim_input_len);
  const start = (trim_input_len - 4000) / 2;
  for (let i = 0; i < 4000; i++) padded[start + i] = 0.4 * Math.sin((2 * Math.PI * 220 * i) / 16000);
  assert.equal(core.trim(padded).length, trim_len);
  assert.equal(core.trim(new Float32Array(100).fill(0.5)).length, short_trim_len);
  assert.equal(core.isSilent(new Float32Array(16000)), silent);
});

test("normalize gives zero mean and unit variance", () => {
  const out = core.normalize(Float32Array.from({ length: 1000 }, (_, i) => Math.sin(i / 10) * 0.3 + 0.1));
  const mean = out.reduce((a, b) => a + b, 0) / out.length;
  const variance = out.reduce((a, b) => a + (b - mean) ** 2, 0) / out.length;
  close(mean, 0, 1e-3, "mean");
  close(variance, 1, 1e-3, "variance");
});

test("WAV encoding round-trips", () => {
  const wav = Float32Array.from({ length: 2000 }, (_, i) => Math.sin(i / 10) * 0.5);
  const { samples, rate } = core.decodeWav(core.encodeWav(wav, 16000));
  assert.equal(rate, 16000);
  assert.equal(samples.length, 2000);
  samples.forEach((v, i) => close(v, wav[i], 1e-3, `sample ${i}`));
});

test("a file that is not a WAV is refused", () => {
  assert.throws(() => core.decodeWav(new ArrayBuffer(100)), /Not a WAV/);
});

test("augmentation is deterministic and makes four variants", () => {
  const wav = Float32Array.from({ length: 4000 }, (_, i) => Math.sin(i / 5) * 0.3);
  const a = core.augment(wav), b = core.augment(wav);
  assert.equal(a.length, 4);
  assert.deepEqual(a[2], b[2]);
  assert.notEqual(a[0].length, wav.length);
});

test("voice activity detection finds the same segments as Python", () => {
  const levels = { quiet: 0.001, sound: 0.2, tick: 0.2, fan: 0.02, "fan+sound": 0.3, long: 0.2 };
  const frames = [];
  for (const [kind, n] of fixture.vad.plan) {
    for (let i = 0; i < n; i++) frames.push(new Float32Array(320).fill(levels[kind] * (i % 2 ? 1 : -1)));
  }
  const stream = new Float32Array(frames.length * 320);
  frames.forEach((f, i) => stream.set(f, i * 320));
  const det = new core.SegmentDetector(), lengths = [];
  for (let i = 0; i < stream.length; i += 1777) lengths.push(...det.push(stream.subarray(i, i + 1777)).map((s) => s.length));
  lengths.push(...det.flush().map((s) => s.length));
  assert.deepEqual(lengths, fixture.vad.segment_lengths);
});

test("the shipped config pins the model by hash, so the integrity check can't be skipped", () => {
  assert.match(MODEL.sha256, /^[0-9a-f]{64}$/);
  assert.match(MODEL.url, /^https:\/\//);
});
