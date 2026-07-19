# audio_module.py
#
# Offline TTS coaching cues via pyttsx3 — no internet needed, uses the
# OS speech engine (SAPI5 on Windows).
#
# runAndWait() blocks, so speaking happens on a dedicated daemon thread
# fed through a queue — the main video loop never waits on it. Each cue
# type has its own cooldown so it doesn't talk constantly, and low-
# priority cues get dropped if the queue backs up rather than piling up
# and playing late. If pyttsx3 isn't installed, everything just no-ops.

import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not installed — audio coaching disabled. "
                   "Run: pip install pyttsx3")

_COOLDOWNS = {
    "rep":        1.5,
    "stroke":     1.2,
    "lunge":      2.0,
    "split_step": 2.5,
    "jump":       2.5,
    "stance":     4.0,
    "rally":      1.5,
    "generic":    1.0,
}

_QUEUE_BACKLOG_LIMIT = 2   # low-priority cues get dropped past this


class AudioCoach:
    """
    Thread-safe TTS coach. Call speak() from any thread.
    Call start() once, stop() on shutdown.
    Toggle enabled/disabled at any time with .enabled.
    """
    def __init__(self, rate: int = 175, volume: float = 0.9):
        self.enabled  = False   # starts OFF — user presses [A] to enable
        self._rate    = rate
        self._volume  = volume
        self._queue: queue.Queue = queue.Queue()
        self._last_spoken: dict  = {}   # cue_type -> timestamp
        self._thread: threading.Thread = None
        self._running = False

    def start(self) -> bool:
        """
        Verifies pyttsx3 is available and starts the worker thread.
        NOTE: the actual TTS engine is intentionally NOT created here and
        held persistently — see _worker() for why.
        """
        if not _PYTTSX3_AVAILABLE:
            return False
        try:
            # Quick sanity check that init works at all on this system,
            # then immediately release it — the worker creates its own
            # fresh instance per utterance (see _worker docstring).
            test_engine = pyttsx3.init()
            test_engine.stop()
            del test_engine
        except Exception as e:
            logger.warning(f"AudioCoach init check failed: {e}")
            self._running = False
            return False

        self._running = True
        self._thread  = threading.Thread(target=self._worker,
                                         daemon=True, name="AudioCoach")
        self._thread.start()
        logger.info("AudioCoach started.")
        return True

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)   # sentinel to unblock worker
        except Exception:
            pass

    def speak(self, text: str, cue_type: str = "generic",
              priority: int = 2) -> None:
        """
        Queue a spoken cue. Respects cooldown and queue backlog limit.
        priority: 1=high (always queued), 2=normal, 3=low (dropped if busy)
        """
        if not self.enabled or not self._running:
            return

        now = time.time()
        cooldown = _COOLDOWNS.get(cue_type, _COOLDOWNS["generic"])
        if now - self._last_spoken.get(cue_type, 0) < cooldown:
            return   # still in cooldown for this cue type

        # Drop low-priority if queue is backing up
        if priority >= 3 and self._queue.qsize() >= _QUEUE_BACKLOG_LIMIT:
            return

        self._last_spoken[cue_type] = now
        try:
            self._queue.put_nowait((text, priority))
        except queue.Full:
            pass

    def _worker(self) -> None:
        # pyttsx3 on Windows has a known issue: reusing one engine
        # instance across multiple say()+runAndWait() calls from a
        # background thread speaks the first utterance fine then goes
        # silent on every call after that. Creating a fresh engine per
        # utterance avoids it — costs ~50-150ms per cue, which is fine
        # since cues are already rate-limited to 1.2-4s apart anyway.
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:   # stop sentinel
                break
            text, _ = item
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate',   self._rate)
                engine.setProperty('volume', self._volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                logger.debug(f"TTS speak error: {e}")


# ── Cue wording — kept here so main.py stays clean and phrasing can be
# tweaked in one place ──────────────────────────────────────────────────

def cue_rep(coach: AudioCoach, mode_name: str, count: int) -> None:
    coach.speak(f"{count}", cue_type="rep", priority=1)

def cue_stroke(coach: AudioCoach, side: str) -> None:
    label = side if side not in ("--", "") else "stroke"
    coach.speak(label, cue_type="stroke", priority=2)

def cue_lunge(coach: AudioCoach, direction: str) -> None:
    label = f"lunge {direction}" if direction not in ("--", "") else "lunge"
    coach.speak(label, cue_type="lunge", priority=2)

def cue_split_step(coach: AudioCoach) -> None:
    coach.speak("split", cue_type="split_step", priority=3)

def cue_jump(coach: AudioCoach) -> None:
    coach.speak("jump", cue_type="jump", priority=3)

def cue_stance(coach: AudioCoach, cues: dict) -> None:
    """cues is the dict returned by check_ready_stance()."""
    if not cues:
        return
    # Only speak the cues that need fixing — "good" ones are silent by design
    msgs = [text for _, (qual, text) in cues.items() if qual != "good"]
    if msgs:
        coach.speak(". ".join(msgs), cue_type="stance", priority=3)

def cue_rally_end(coach: AudioCoach, length: int) -> None:
    coach.speak(f"rally, {length}", cue_type="rally", priority=2)