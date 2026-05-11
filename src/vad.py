import collections
import time

import numpy as np

try:
    import webrtcvad

    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False

try:
    import pyaudio

    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

# Audio constants
SAMPLE_RATE_16K = 16000
BYTES_PER_SAMPLE = 2
MAX_INT16 = 32767

# VAD defaults
DEFAULT_FRAME_MS = 30
DEFAULT_AGGRESSIVENESS = 2
DEFAULT_SPEECH_PAD_MS = 300
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_MIN_SILENCE_MS = 500
TRIGGER_THRESHOLD = 0.9


class VADConfig:
    """VAD configuration."""

    def __init__(
        self,
        sample_rate=SAMPLE_RATE_16K,
        frame_duration_ms=DEFAULT_FRAME_MS,
        aggressiveness=DEFAULT_AGGRESSIVENESS,
        speech_pad_ms=DEFAULT_SPEECH_PAD_MS,
        min_speech_duration_ms=DEFAULT_MIN_SPEECH_MS,
        min_silence_duration_ms=DEFAULT_MIN_SILENCE_MS,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.aggressiveness = aggressiveness
        self.speech_pad_ms = speech_pad_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms


class VADState:
    """VAD state constants."""

    SILENCE = "silence"
    SPEECH = "speech"
    SPEECH_END = "speech_end"


class VoiceActivityDetector:
    """Real-time voice activity detector."""

    def __init__(self, cfg=None):
        if not HAS_WEBRTCVAD:
            raise ImportError("webrtcvad not installed")

        self.cfg = cfg or VADConfig()
        self.vad = webrtcvad.Vad(self.cfg.aggressiveness)

        self.frame_size = int(self.cfg.sample_rate * self.cfg.frame_duration_ms / 1000)

        num_pad = int(self.cfg.speech_pad_ms / self.cfg.frame_duration_ms)
        self.ring_buffer = collections.deque(maxlen=num_pad)

        self.triggered = False
        self.voiced_frames = []
        self.state = VADState.SILENCE
        self.last_state_change = time.time()

        self.on_speech_start = None
        self.on_speech_end = None

    def reset(self):
        """Reset detector state."""
        self.ring_buffer.clear()
        self.triggered = False
        self.voiced_frames = []
        self.state = VADState.SILENCE
        self.last_state_change = time.time()

    def process_frame(self, frame):
        """Process audio frame and return state."""
        is_speech = self.vad.is_speech(frame, self.cfg.sample_rate)

        if not self.triggered:
            self.ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in self.ring_buffer if speech])

            if num_voiced > TRIGGER_THRESHOLD * self.ring_buffer.maxlen:
                self.triggered = True
                self.state = VADState.SPEECH
                self.last_state_change = time.time()

                for f, _ in self.ring_buffer:
                    self.voiced_frames.append(f)
                self.ring_buffer.clear()

                if self.on_speech_start:
                    self.on_speech_start()
        else:
            self.voiced_frames.append(frame)
            self.ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])

            if num_unvoiced > TRIGGER_THRESHOLD * self.ring_buffer.maxlen:
                self.triggered = False
                self.state = VADState.SPEECH_END

                audio = b"".join(self.voiced_frames)

                if self.on_speech_end:
                    self.on_speech_end(audio)

                self.voiced_frames = []
                self.ring_buffer.clear()

                self.state = VADState.SILENCE
                self.last_state_change = time.time()

        return self.state

    def process_audio(self, audio):
        """Process audio chunk and return state transitions."""
        frames = self._split_frames(audio)
        events = []

        for frame in frames:
            prev = self.state
            new = self.process_frame(frame)

            if new != prev:
                if new == VADState.SPEECH:
                    events.append((VADState.SPEECH, None))
                elif new == VADState.SPEECH_END:
                    events.append((VADState.SPEECH_END, b"".join(self.voiced_frames)))

        return events

    def _split_frames(self, audio):
        """Split audio into frames."""
        frame_bytes = self.frame_size * BYTES_PER_SAMPLE
        frames = []

        for i in range(0, len(audio) - frame_bytes + 1, frame_bytes):
            frames.append(audio[i : i + frame_bytes])

        return frames

    def is_speaking(self):
        """Check if speaking."""
        return self.triggered

    def time_since_state_change(self):
        """Get seconds since last state change."""
        return time.time() - self.last_state_change


class MicrophoneVAD:
    """Real-time microphone VAD monitor."""

    def __init__(self, cfg=None):
        if not HAS_PYAUDIO:
            raise ImportError("pyaudio not installed")

        self.cfg = cfg or VADConfig()
        self.vad = VoiceActivityDetector(cfg)

        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.running = False

        self.on_speech_start = None
        self.on_speech_end = None
        self.on_interrupt = None

    def start(self):
        """Start monitoring microphone."""
        if self.running:
            return

        self.vad.on_speech_start = self._handle_speech_start
        self.vad.on_speech_end = self._handle_speech_end

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.cfg.sample_rate,
            input=True,
            frames_per_buffer=self.vad.frame_size,
            stream_callback=self._audio_callback,
        )

        self.running = True
        self.stream.start_stream()

    def stop(self):
        """Stop monitoring microphone."""
        if not self.running:
            return

        self.running = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def close(self):
        """Clean up resources."""
        self.stop()
        self.pa.terminate()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback."""
        if self.running and in_data:
            self.vad.process_frame(in_data)
        return (in_data, pyaudio.paContinue)

    def _handle_speech_start(self):
        """Handle speech start."""
        if self.on_speech_start:
            self.on_speech_start()
        if self.on_interrupt:
            self.on_interrupt()

    def _handle_speech_end(self, audio):
        """Handle speech end."""
        if self.on_speech_end:
            self.on_speech_end(audio)

    def is_speaking(self):
        """Check if speaking."""
        return self.vad.is_speaking()


def audio_energy(audio):
    """Calculate audio energy level (0.0 to 1.0)."""
    if len(audio) < BYTES_PER_SAMPLE:
        return 0.0

    samples = np.frombuffer(audio, dtype=np.int16)
    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

    return min(1.0, rms / MAX_INT16)


def detect_speech_energy(
    audio, threshold=0.02, sample_rate=SAMPLE_RATE_16K, frame_ms=DEFAULT_FRAME_MS
):
    """Energy-based speech detection."""
    frame_size = int(sample_rate * frame_ms / 1000) * BYTES_PER_SAMPLE
    segments = []
    in_speech = False
    speech_start = 0

    for i in range(0, len(audio) - frame_size + 1, frame_size):
        frame = audio[i : i + frame_size]
        energy = audio_energy(frame)
        time_ms = (i // BYTES_PER_SAMPLE) * 1000 // sample_rate

        if energy > threshold:
            if not in_speech:
                in_speech = True
                speech_start = time_ms
        else:
            if in_speech:
                in_speech = False
                segments.append(
                    {
                        "start": speech_start,
                        "end": time_ms,
                        "duration": time_ms - speech_start,
                    }
                )

    if in_speech:
        end_time = (len(audio) // BYTES_PER_SAMPLE) * 1000 // sample_rate
        segments.append(
            {
                "start": speech_start,
                "end": end_time,
                "duration": end_time - speech_start,
            }
        )

    return segments
