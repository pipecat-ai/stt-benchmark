"""Experiment: what does an STT service's end-of-transcript signal mean?

Plays benchmark samples through the benchmark's own pipeline (Silero VAD, then
the STT service) and logs every message the service sends, when it arrived, and
the audio times of its words. Nothing is written to the results database: each
sample is one line of `messages.jsonl` in the output directory.

Each service's messages are reduced to *segments*: a piece of final text, and
the signal that closed it.

- deepgram (nova-3-general): a final `Results` message. Signal: `speech_final`,
  `from_finalize`, or none.
- soniox_v4 / soniox_v6: the final tokens up to an `<end>` token (Soniox's own
  endpoint detection) or a `<fin>` token (the answer to `finalize`).
- assemblyai (assemblyai-stt-3-7-realtime-agent): a `Turn` message with
  `end_of_turn`.

Two things are varied:

- `--finalize` / `--no-finalize`: whether the service is told that the turn
  ended at each VAD stop (Deepgram `Finalize`, Soniox `finalize`, AssemblyAI
  `ForceEndpoint`), as the benchmark does. Without it, only the service's own
  endpointing closes a segment.
- `--endpointing`: Deepgram's server-side endpointing, in ms. 10 is its
  default. Ignored for the other services.

Run, then summarise:

    uv run python scripts/end_signals.py run --service soniox_v6 -n 150
    uv run python scripts/end_signals.py report <output-dir>
"""

import argparse
import asyncio
import json
import os
import sqlite3
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.workers.runner import WorkerRunner

from stt_benchmark.config import get_config
from stt_benchmark.evals.bot import build_worker

# Two finals whose words are closer together than this were split inside
# continuous speech, not at a pause.
SPLICE_GAP_SECS = 0.3
LEAD_IN_SECS = 1.0


class SampleAudio(FrameProcessor):
    """Plays a sample in real time, marks its end, then sends silence."""

    CHUNK_MS = 20

    def __init__(self, audio: bytes, sample_rate: int, log: dict):
        super().__init__()
        self._audio = audio
        self._sample_rate = sample_rate
        self._log = log
        self._playing: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._playing = self.create_task(self._play())

    async def stop(self):
        if self._playing:
            await self.cancel_task(self._playing)
            self._playing = None

    async def _play(self):
        chunk = int(self._sample_rate * self.CHUNK_MS / 1000) * 2
        # Silence first, so the service is connected before the speech starts.
        audio = b"\x00" * int(LEAD_IN_SECS * self._sample_rate) * 2 + self._audio
        offset = 0
        marked = False
        next_at = time.monotonic()
        self._log["stream_start"] = next_at
        while True:
            if offset >= len(audio) and not marked:
                self._log["audio_end"] = time.monotonic()
                await self.push_frame(InputDTMFFrame(button=KeypadEntry.POUND))
                marked = True
            piece = audio[offset : offset + chunk].ljust(chunk, b"\x00")
            offset += chunk
            await self.push_frame(
                InputAudioRawFrame(audio=piece, sample_rate=self._sample_rate, num_channels=1)
            )
            next_at += self.CHUNK_MS / 1000
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))


class LoggingDeepgramSTT(DeepgramSTTService):
    """Deepgram, with every Results message and VAD event written to a log."""

    def __init__(self, *, log: dict, send_finalize: bool, **kwargs):
        super().__init__(**kwargs)
        self._log = log
        self._send_finalize = send_finalize

    async def _on_message(self, message):
        channel = getattr(message, "channel", None)
        if channel is not None and channel.alternatives:
            alternative = channel.alternatives[0]
            self._log["messages"].append(
                {
                    "at": time.monotonic(),
                    "is_final": bool(getattr(message, "is_final", False)),
                    "speech_final": bool(getattr(message, "speech_final", False)),
                    "from_finalize": bool(getattr(message, "from_finalize", False)),
                    "start": getattr(message, "start", None),
                    "duration": getattr(message, "duration", None),
                    "transcript": alternative.transcript,
                    "words": [[w.word, w.start, w.end] for w in alternative.words or []],
                }
            )
        await super()._on_message(message)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, VADUserStoppedSpeakingFrame) and not self._send_finalize:
            # Everything the service does on a VAD stop except Finalize.
            await STTService.process_frame(self, frame, direction)
            return
        await super().process_frame(frame, direction)


class VADLog(FrameProcessor):
    """Sits after the STT service and notes when the VAD's events pass."""

    def __init__(self, log: dict):
        super().__init__()
        self._log = log

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._log["vad"].append(["start", time.monotonic()])
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._log["vad"].append(["stop", time.monotonic()])
        await self.push_frame(frame, direction)


class LoggingWebsocket:
    """A websocket that notes each message it yields."""

    def __init__(self, websocket, on_message):
        self._websocket = websocket
        self._on_message = on_message

    def __getattr__(self, name):
        return getattr(self._websocket, name)

    async def __aiter__(self):
        async for message in self._websocket:
            self._on_message(message)
            yield message


def _deepgram(log: dict, endpointing: int, send_finalize: bool) -> FrameProcessor:
    return LoggingDeepgramSTT(
        log=log,
        send_finalize=send_finalize,
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(
            model="nova-3-general",
            smart_format=False,
            profanity_filter=False,
            language=Language.EN,
            endpointing=endpointing,
        ),
    )


def _soniox(model: str, log: dict, send_finalize: bool) -> FrameProcessor:
    from pipecat.services.soniox.stt import SonioxSTTService

    stt = SonioxSTTService(
        api_key=os.environ["SONIOX_API_KEY"],
        settings=SonioxSTTService.Settings(
            model=model,
            language_hints=[Language.EN],
            language_hints_strict=True,
        ),
        vad_force_turn_endpoint=send_finalize,
    )

    def on_message(message):
        try:
            tokens = json.loads(message).get("tokens") or []
        except (ValueError, AttributeError):
            return
        if tokens:
            log["messages"].append(
                {
                    "at": time.monotonic(),
                    "tokens": [
                        [t.get("text"), t.get("start_ms"), t.get("end_ms"), bool(t.get("is_final"))]
                        for t in tokens
                    ],
                }
            )

    get_websocket = stt._get_websocket
    stt._get_websocket = lambda: LoggingWebsocket(get_websocket(), on_message)
    return stt


def _assemblyai(log: dict, send_finalize: bool) -> FrameProcessor:
    from pipecat.services.assemblyai.stt import AssemblyAISTTService

    stt = AssemblyAISTTService(
        api_key=os.environ["ASSEMBLYAI_API_KEY"],
        settings=AssemblyAISTTService.Settings(
            model="assemblyai-stt-3-7-realtime-agent",
            min_turn_silence=50,
            vad_threshold=0.2,
            language_codes=[Language.EN],
        ),
        vad_force_turn_endpoint=send_finalize,
    )
    handle_message = stt._handle_message

    async def logged(message):
        if isinstance(message, dict) and message.get("type") in ("Turn", "SpeechStarted"):
            log["messages"].append({"at": time.monotonic(), **message})
        await handle_message(message)

    stt._handle_message = logged
    return stt


SERVICES = {
    "deepgram": lambda log, e, f: _deepgram(log, e, f),
    "soniox_v4": lambda log, e, f: _soniox("stt-rt-v4", log, f),
    "soniox_v6": lambda log, e, f: _soniox("stt-rt-v6", log, f),
    "assemblyai": lambda log, e, f: _assemblyai(log, f),
}


async def run_sample(sample: dict, service: str, endpointing: int, send_finalize: bool) -> dict:
    config = get_config()
    log: dict = {
        "sample_id": sample["sample_id"],
        "duration_seconds": sample["duration_seconds"],
        "service": service,
        "endpointing": endpointing,
        "finalize": send_finalize,
        "vad_stop_secs": config.vad_stop_secs,
        "messages": [],
        "vad": [],
    }
    stt = SERVICES[service](log, endpointing, send_finalize)
    audio = SampleAudio(Path(sample["audio_path"]).read_bytes(), config.sample_rate, log)
    worker, report_sample = build_worker(
        audio,
        stt,
        VADLog(log),
        vad_stop_secs=config.vad_stop_secs,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        result = await asyncio.wait_for(report_sample(), timeout=90)
        log["transcription"] = result.transcription
        log["ttfs_seconds"] = result.ttfb_seconds
        log["error"] = result.error
    except Exception as e:
        log["error"] = f"{type(e).__name__}: {e}"
    finally:
        await audio.stop()
        await runner.cancel()
        await asyncio.gather(running, return_exceptions=True)
    if "audio_end" in log:
        # How far behind real time the audio was played. Late audio reaches
        # Deepgram in bursts, which is not what is being measured.
        played = log["audio_end"] - log["stream_start"]
        log["playback_lag"] = played - LEAD_IN_SECS - sample["duration_seconds"]
    return log


RESULT_MARK = "@@result "


def run_one() -> None:
    """Run the one sample given on stdin, and print its log. Used by `run`."""
    job = json.loads(sys.stdin.read())
    log = asyncio.run(
        run_sample(job["sample"], job["service"], job["endpointing"], job["finalize"])
    )
    print(RESULT_MARK + json.dumps(log), flush=True)


def load_samples(limit: int, offset: int) -> list[dict]:
    db = get_config().results_db
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT sample_id, audio_path, duration_seconds FROM samples "
            "ORDER BY dataset_index LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


async def _run_jobs(jobs: list[tuple], out: Path, concurrency: int) -> None:
    """Run each sample in a process of its own.

    Samples sharing one event loop delay each other's audio, and a process
    that has run many samples slows down.
    """
    semaphore = asyncio.Semaphore(concurrency)
    finished = 0

    async def one(sample: dict, service: str, endpointing: int, send_finalize: bool, repeat: int):
        nonlocal finished
        job = {
            "sample": sample,
            "service": service,
            "endpointing": endpointing,
            "finalize": send_finalize,
        }
        async with semaphore:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                __file__,
                "one",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(json.dumps(job).encode()), timeout=180
                )
            except TimeoutError:
                process.kill()
                stdout = b""
        log = None
        for line in stdout.decode(errors="replace").splitlines():
            if line.startswith(RESULT_MARK):
                log = json.loads(line[len(RESULT_MARK) :])
        if log is None:
            log = {
                "sample_id": sample["sample_id"],
                "service": service,
                "endpointing": endpointing,
                "finalize": send_finalize,
                "messages": [],
                "vad": [],
                "error": "the sample's process returned no result",
            }
        log["repeat"] = repeat
        with out.open("a") as f:
            f.write(json.dumps(log) + "\n")
        finished += 1
        if finished % 25 == 0 or finished == len(jobs):
            print(f"  {finished}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(*job) for job in jobs))


def run(args) -> Path:
    samples = load_samples(args.num_samples, args.offset)
    if args.sample_ids:
        wanted = set(Path(args.sample_ids).read_text().split())
        samples = [s for s in load_samples(100000, 0) if s["sample_id"] in wanted]
    out_dir = Path(
        args.output_dir
        or get_config().data_dir / "end_signals" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "messages.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add(
                (
                    r["sample_id"],
                    r.get("service", "deepgram"),
                    r["endpointing"],
                    r["finalize"],
                    r.get("repeat", 0),
                )
            )

    jobs = [
        (s, service, e, f, rep)
        for rep in range(args.repeats)
        for service in args.service
        for e in (args.endpointing if service == "deepgram" else [0])
        for f in args.finalize
        for s in samples
        if (s["sample_id"], service, e, f, rep) not in done
    ]
    print(f"{len(jobs)} runs to do ({len(done)} already in {out})")
    asyncio.run(_run_jobs(jobs, out, args.concurrency))
    return out_dir


def segments(record: dict) -> list[dict]:
    """A sample's final text, as the pieces the service closed.

    Each segment has its text, the signal that closed it (None for a Deepgram
    final without one), the audio times of its first and last word, when its
    last text arrived (``text_at``) and when its signal did (``at``).
    """
    service = record.get("service", "deepgram")
    out = []
    if service == "deepgram":
        for m in record["messages"]:
            if not m["is_final"]:
                continue
            signal = (
                "from_finalize"
                if m["from_finalize"]
                else "speech_final"
                if m["speech_final"]
                else None
            )
            words = m["words"]
            out.append(
                {
                    "text": m["transcript"],
                    "signal": signal,
                    "start": words[0][1] if words else None,
                    "end": words[-1][2] if words else None,
                    "text_at": m["at"],
                    "at": m["at"],
                }
            )
    elif service.startswith("soniox"):
        pending: list = []
        text_at = None
        for m in record["messages"]:
            for text, start_ms, end_ms, is_final in m["tokens"]:
                if not is_final:
                    continue
                if text in ("<end>", "<fin>"):
                    timed = [t for t in pending if t[1] is not None]
                    out.append(
                        {
                            "text": "".join(t[0] for t in pending).strip(),
                            "signal": text.strip("<>"),
                            "start": timed[0][1] / 1000 if timed else None,
                            "end": timed[-1][2] / 1000 if timed else None,
                            "text_at": text_at if pending else m["at"],
                            "at": m["at"],
                        }
                    )
                    pending = []
                else:
                    pending.append((text, start_ms, end_ms))
                    text_at = m["at"]
        if pending:
            # Final tokens that no end token ever closed.
            timed = [t for t in pending if t[1] is not None]
            out.append(
                {
                    "text": "".join(t[0] for t in pending).strip(),
                    "signal": None,
                    "start": timed[0][1] / 1000 if timed else None,
                    "end": timed[-1][2] / 1000 if timed else None,
                    "text_at": text_at,
                    "at": text_at,
                }
            )
    elif service == "assemblyai":
        turns: dict[int, dict] = {}
        for m in record["messages"]:
            if m.get("type") != "Turn" or not m.get("end_of_turn"):
                continue
            words = m.get("words") or []
            # A turn's end can arrive twice, unformatted and then formatted:
            # keep the first arrival and the last text.
            first = turns.get(m["turn_order"])
            turns[m["turn_order"]] = {
                "text": m.get("transcript", ""),
                "signal": "end_of_turn",
                "start": words[0]["start"] / 1000 if words else None,
                "end": words[-1]["end"] / 1000 if words else None,
                "text_at": first["text_at"] if first else m["at"],
                "at": first["at"] if first else m["at"],
            }
        out = [turns[k] for k in sorted(turns)]
    return out


def analyse(record: dict) -> dict | None:
    """What one sample's segments say about the service's end signal."""
    if record.get("error") or "stream_start" not in record:
        return None
    segs = [s for s in segments(record) if s["text"]]
    if not segs:
        return None
    stops = [t for kind, t in record["vad"] if kind == "stop"]
    # When the speech ended, as the benchmark defines it for TTFS.
    speech_end = stops[-1] - record["vad_stop_secs"] if stops else None

    # A segment closed by an end signal, followed by more text.
    trailing = []
    for i, seg in enumerate(segs[:-1]):
        nxt = segs[i + 1]
        if not seg["signal"] or seg["end"] is None or nxt["start"] is None:
            continue
        trailing.append(
            {
                "after": seg["signal"],
                "gap": nxt["start"] - seg["end"],
                "before": seg["text"],
                "next": nxt["text"],
                "next_words": len(nxt["text"].split()),
                "is_last": i + 1 == len(segs) - 1,
            }
        )
    last = segs[-1]
    return {
        "sample_id": record["sample_id"],
        "segments": len(segs),
        "trailing": trailing,
        "splices": [t for t in trailing if t["gap"] < SPLICE_GAP_SECS],
        # Only the last word or two of the sample came after a signal.
        "last_words_trail": any(t["is_last"] and t["next_words"] <= 2 for t in trailing),
        "last_is_signalled": bool(last["signal"]),
        # Latency from the end of speech to the sample's last text, and to the
        # signal that closed it.
        "to_last_text": last["text_at"] - speech_end if speech_end else None,
        "to_last_signal": last["at"] - speech_end if speech_end and last["signal"] else None,
        "text": " ".join(s["text"] for s in segs),
    }


def _ms(values: list[float], q: float) -> str:
    if not values:
        return "-"
    values = sorted(values)
    return f"{values[min(len(values) - 1, int(q * len(values)))] * 1000:.0f}"


def report(out_dir: Path, examples: int) -> None:
    records = [
        json.loads(line) for line in (out_dir / "messages.jsonl").read_text().splitlines() if line
    ]
    arms: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record.get("service", "deepgram"), record["endpointing"], record["finalize"])
        arms.setdefault(key, []).append(record)

    print(
        f"{'service':<11} {'endpt':>5} {'finalize':>8} {'n':>4} {'err':>3} {'segs':>5} "
        f"{'trailing':>9} {'splice':>9} {'last1-2':>8} {'signalled':>9} "
        f"{'text p50':>8} {'p95':>5} {'signal p50':>10} {'p95':>5} {'lag p95':>7}"
    )
    for (service, endpointing, finalize), group in sorted(arms.items()):
        rows = [a for a in map(analyse, group) if a]
        errors = sum(1 for r in group if r.get("error"))
        n = len(rows)
        head = f"{service:<11} {endpointing:>5} {finalize!s:>8} {n:>4} {errors:>3}"
        if not n:
            print(head)
            continue
        trailing = sum(1 for r in rows if r["trailing"])
        splices = sum(1 for r in rows if r["splices"])
        last_words = sum(1 for r in rows if r["last_words_trail"])
        text = [r["to_last_text"] for r in rows if r["to_last_text"] is not None]
        signal = [r["to_last_signal"] for r in rows if r["to_last_signal"] is not None]
        lag = [r["playback_lag"] for r in group if "playback_lag" in r]
        print(
            f"{head} {statistics.mean(r['segments'] for r in rows):>5.2f} "
            f"{trailing:>4} {trailing / n:>4.0%} {splices:>4} {splices / n:>4.0%} "
            f"{last_words / n:>8.0%} {sum(r['last_is_signalled'] for r in rows) / n:>9.0%} "
            f"{_ms(text, 0.5):>8} {_ms(text, 0.95):>5} "
            f"{_ms(signal, 0.5):>10} {_ms(signal, 0.95):>5} {_ms(lag, 0.95):>7}"
        )
        shown = 0
        for r in rows:
            for t in r["splices"]:
                if shown >= examples:
                    break
                shown += 1
                print(
                    f"      {r['sample_id'][:8]} after {t['after']}, gap {t['gap'] * 1000:.0f}ms: "
                    f"...{t['before'][-45:]!r} | {t['next'][:45]!r}"
                )

    print(
        "\nsegs: pieces of final text per sample.\n"
        "trailing: samples where a segment closed by an end signal was followed by more text.\n"
        f"splice: of those, the next text began under {SPLICE_GAP_SECS * 1000:.0f}ms of audio "
        "after the last word.\n"
        "last1-2: samples where only the last one or two words came after a signal.\n"
        "signalled: samples whose last text was closed by an end signal.\n"
        "text / signal: ms from the end of speech (VAD stop minus stop_secs) to the sample's "
        "last text, and to the signal that closed it.\n"
        "lag: how far behind real time the audio was played, ms."
    )


def wer(out_dir: Path, limit: int | None) -> None:
    """Score each run's transcript with the benchmark's judge, and report it with TTFS.

    The transcript and TTFS are the ones the benchmark would have stored for
    the sample. Judgments go to `judgments.jsonl` in the output directory and
    are reused on the next call.
    """
    from judge_common import judgment_cost, run_judgments

    from stt_benchmark.evaluation.semantic_wer import (
        DEFAULT_JUDGE_EFFORT,
        DEFAULT_JUDGE_MODEL,
    )

    config = get_config()
    conn = sqlite3.connect(f"file:{config.results_db}?mode=ro", uri=True)
    try:
        truth = dict(conn.execute("SELECT sample_id, text FROM ground_truth"))
    finally:
        conn.close()

    records = [
        json.loads(line) for line in (out_dir / "messages.jsonl").read_text().splitlines() if line
    ]
    arms: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record.get("service", "deepgram"), record["endpointing"], record["finalize"])
        arms.setdefault(key, []).append(record)

    pairs = []
    for (service, endpointing, finalize), group in arms.items():
        for record in group[:limit]:
            if record.get("transcription") and record["sample_id"] in truth:
                pairs.append(
                    {
                        "service_name": f"{service}/{endpointing}/{finalize}",
                        "model_name": "",
                        "sample_id": record["sample_id"],
                        "reference": truth[record["sample_id"]],
                        "transcription": record["transcription"],
                    }
                )
    runs, evaluator, made = asyncio.run(
        run_judgments(
            DEFAULT_JUDGE_MODEL,
            DEFAULT_JUDGE_EFFORT,
            pairs,
            1,
            out_dir / "judgments.jsonl",
            config.data_dir / "test_results.db",
        )
    )
    print(f"{made} judgments made now, ${judgment_cost(evaluator):.2f}. Judge: {evaluator.judge}")

    print(
        f"{'service':<11} {'endpt':>5} {'finalize':>8} {'n':>4} {'scored':>6} {'perfect':>7} "
        f"{'WER mean':>8} {'pooled':>7} {'TTFS p50':>8} {'p95':>5} {'p99':>5}"
    )
    for (service, endpointing, finalize), group in sorted(arms.items()):
        group = group[:limit]
        name = f"{service}/{endpointing}/{finalize}"
        scored = [
            run[0]
            for record in group
            if (run := runs.get(f"{name}||{record['sample_id']}")) and run[0] is not None
        ]
        ttfs = [r["ttfs_seconds"] for r in group if r.get("ttfs_seconds") is not None]
        head = f"{service:<11} {endpointing:>5} {finalize!s:>8} {len(group):>4} {len(scored):>6}"
        if not scored:
            print(head)
            continue
        words = sum(r["reference_words"] for r in scored)
        print(
            f"{head} {sum(1 for r in scored if r['error_count'] == 0) / len(scored):>7.1%} "
            f"{statistics.mean(r['wer'] for r in scored):>8.2%} "
            f"{sum(r['error_count'] for r in scored) / words:>7.2%} "
            f"{_ms(ttfs, 0.5):>8} {_ms(ttfs, 0.95):>5} {_ms(ttfs, 0.99):>5}"
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--service", nargs="+", default=["deepgram"], choices=list(SERVICES))
    run_parser.add_argument("--endpointing", type=int, nargs="+", default=[10])
    run_parser.add_argument("--finalize", action="store_true", default=None)
    run_parser.add_argument("--no-finalize", dest="finalize", action="store_false")
    run_parser.add_argument("-n", "--num-samples", type=int, default=100)
    run_parser.add_argument("--offset", type=int, default=0)
    run_parser.add_argument("--sample-ids", help="File of sample ids to run instead")
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("-c", "--concurrency", type=int, default=12)
    run_parser.add_argument("--output-dir", help="Resume into, or write to, this directory")

    commands.add_parser("one")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("output_dir", type=Path)
    report_parser.add_argument("--examples", type=int, default=5)

    wer_parser = commands.add_parser("wer")
    wer_parser.add_argument("output_dir", type=Path)
    wer_parser.add_argument("--limit", type=int, help="Score only the first N runs of each arm")

    args = parser.parse_args()
    logger.remove()
    logger.add(lambda m: print(m, end=""), level="ERROR")
    if args.command == "one":
        run_one()
    elif args.command == "wer":
        wer(args.output_dir, args.limit)
    elif args.command == "run":
        # Neither flag given: run both arms.
        args.finalize = [True, False] if args.finalize is None else [args.finalize]
        out_dir = run(args)
        report(out_dir, examples=5)
        print(f"\nOutput: {out_dir}")
    else:
        report(args.output_dir, args.examples)


if __name__ == "__main__":
    main()
