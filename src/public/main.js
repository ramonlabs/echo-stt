const btn = document.getElementById("btn");
const status = document.getElementById("status");
const transcript = document.getElementById("transcript");

const FRAME_SAMPLES = 480; // 30ms at 16kHz
const MIN_SPEECH_MS = 400; // ignore speech bursts shorter than this

const State = {
  IDLE: 0,
  LISTENING: 1,
  STARTING_STT: 2,
  RECORDING: 3,
  TRANSCRIBING: 4,
};

let state = State.IDLE;
let stream = null;
let audioCtx = null;
let processor = null;
let vadWs = null;
let sttWs = null;
let recorder = null;
let speechTimer = null;

btn.addEventListener("click", () => {
  state === State.IDLE ? startListening() : stopListening();
});

async function startListening() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new AudioContext({ sampleRate: 16000 });

  const source = audioCtx.createMediaStreamSource(stream);
  processor = audioCtx.createScriptProcessor(512, 1, 1);
  let pcmBuffer = new Float32Array(0);

  processor.onaudioprocess = (e) => {
    if (vadWs?.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const combined = new Float32Array(pcmBuffer.length + input.length);
    combined.set(pcmBuffer);
    combined.set(input, pcmBuffer.length);
    pcmBuffer = combined;

    while (pcmBuffer.length >= FRAME_SAMPLES) {
      const frame = pcmBuffer.slice(0, FRAME_SAMPLES);
      pcmBuffer = pcmBuffer.slice(FRAME_SAMPLES);
      const int16 = new Int16Array(FRAME_SAMPLES);
      for (let i = 0; i < FRAME_SAMPLES; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, frame[i] * 32768));
      }
      vadWs.send(int16.buffer);
    }
  };

  source.connect(processor);
  processor.connect(audioCtx.destination);

  vadWs = new WebSocket(`ws://${location.host}/ws/vad`);
  vadWs.onmessage = onVadMessage;
  vadWs.onerror = (e) => console.error("VAD ws error", e);

  setState(State.LISTENING);
}

function onVadMessage(e) {
  const msg = JSON.parse(e.data);
  console.log(
    "VAD",
    msg.state,
    "speaking:",
    msg.is_speaking,
    "time:",
    msg.time_in_state?.toFixed(2),
  );

  if (msg.is_speaking) {
    if (state === State.LISTENING) {
      // debounce: only start STT if speech holds for MIN_SPEECH_MS
      if (!speechTimer) {
        speechTimer = setTimeout(() => {
          speechTimer = null;
          if (state === State.LISTENING) startSTT();
        }, MIN_SPEECH_MS);
      }
    }
  } else {
    // silence: cancel pending start or stop active recording
    if (speechTimer) {
      clearTimeout(speechTimer);
      speechTimer = null;
    }
    if (state === State.RECORDING) stopSTT();
  }
}

function startSTT() {
  setState(State.STARTING_STT);

  sttWs = new WebSocket(`ws://${location.host}/ws/stt`);

  sttWs.onopen = () => {
    sttWs.send(JSON.stringify({ type: "start" }));
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0 && sttWs?.readyState === WebSocket.OPEN)
        sttWs.send(e.data);
    };
    recorder.start(250);
    setState(State.RECORDING);
  };

  sttWs.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "transcription") {
      if (msg.text) {
        transcript.value += (transcript.value ? "\n" : "") + msg.text;
        transcript.scrollTop = transcript.scrollHeight;
      }
      if (msg.final) {
        sttWs.close();
        sttWs = null;
        recorder = null;
        setState(State.LISTENING);
      }
    }
  };

  sttWs.onerror = () => {
    sttWs = null;
    recorder = null;
    setState(State.LISTENING);
  };
}

function stopSTT() {
  setState(State.TRANSCRIBING);
  const r = recorder;
  recorder = null;
  r.onstop = () =>
    sttWs?.send(JSON.stringify({ type: "end", format: ".webm" }));
  r.stop();
}

function stopListening() {
  if (speechTimer) {
    clearTimeout(speechTimer);
    speechTimer = null;
  }
  vadWs?.close();
  sttWs?.close();
  if (recorder) {
    recorder.stop();
    recorder = null;
  }
  processor?.disconnect();
  audioCtx?.close();
  stream?.getTracks().forEach((t) => t.stop());
  vadWs = null;
  sttWs = null;
  audioCtx = null;
  processor = null;
  stream = null;
  setState(State.IDLE);
}

function setState(s) {
  state = s;
  const labels = {
    [State.IDLE]: ["start listening", "idle", ""],
    [State.LISTENING]: ["stop listening", "listening...", ""],
    [State.STARTING_STT]: ["stop listening", "speaking...", "recording"],
    [State.RECORDING]: ["stop listening", "speaking...", "recording"],
    [State.TRANSCRIBING]: ["stop listening", "transcribing...", ""],
  };
  const [btnText, statusText, statusClass] = labels[s];
  btn.textContent = btnText;
  status.textContent = statusText;
  status.className = statusClass;
}
