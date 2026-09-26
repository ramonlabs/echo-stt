import collections
import time

try:
    import webrtcvad

    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False

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
            num_voiced = sum(1 for _, speech in self.ring_buffer if speech)

            if num_voiced > TRIGGER_THRESHOLD * self.ring_buffer.maxlen:
                self.triggered = True
                self.state = VADState.SPEECH
                self.last_state_change = time.time()

                self.voiced_frames.extend(f for f, _ in self.ring_buffer)
                self.ring_buffer.clear()

                if self.on_speech_start:
                    self.on_speech_start()
        else:
            self.voiced_frames.append(frame)
            self.ring_buffer.append((frame, is_speech))
            num_unvoiced = sum(1 for _, speech in self.ring_buffer if not speech)

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
        return [
            audio[i : i + frame_bytes]
            for i in range(0, len(audio) - frame_bytes + 1, frame_bytes)
        ]

    def is_speaking(self):
        """Check if speaking."""
        return self.triggered

    def time_since_state_change(self):
        """Get seconds since last state change."""
        return time.time() - self.last_state_change
