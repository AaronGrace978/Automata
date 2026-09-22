import { mount } from "./diatom3d.js";

// One body. The cellular automaton grows the frustule, and when it speaks,
// the same cells grow into the words of the reply, one at a time, then
// return to glass. The chat log is a transcript; the body is the display.

const morph = window.DiatomMorph;
const voice = window.DiatomVoice;
const glyph = window.DiatomGlyph;
const { DiatomBody } = window.DiatomNCA;

const SIZE = 48;
const MORPH_STEPS = 48;
const HOLD_STEPS = 30;
const REST_STEPS = 30;

const canvas = document.getElementById("view");
const log = document.getElementById("log");
const form = document.getElementById("composer");
const input = document.getElementById("say");
const statusEl = document.getElementById("status");
const lockline = document.getElementById("lockline");
const personaEl = document.getElementById("persona");
const meters = document.getElementById("meters");
const wordEl = document.getElementById("word");
const smoke = new URLSearchParams(location.search).has("smoke");

const body = new DiatomBody(SIZE, window.DIATOM_WEIGHTS || null);
const view = mount(canvas, body);

let rid = [0.05, 0.15, 0, 0.25, 0, 0.2];
let persona = "diatom_elder";
let memory = [];
let busy = false;
let prevMass = null;
let stepsPerFrame = 2;
let queue = [];
let current = null;
let restLeft = 0;
let woundedRecently = 0;

const METER_KEYS = [
  ["arousal", 0],
  ["valence", 1],
  ["pain", 2],
  ["hunger", 3],
  ["rhythm", 4],
  ["calm", 5],
];

function paintMeters() {
  const rows = METER_KEYS.map(([name, index]) => {
    const value = (rid[index] + 1) / 2;
    return `<div class="meter"><span>${name}</span><div class="bar"><div style="width:${Math.round(value * 100)}%"></div></div></div>`;
  });
  rows.push(
    `<p class="fold">${body.trained ? "trained rule" : "untrained soup"} · ${SIZE}×${SIZE} cells · ${queue.length + (current ? 1 : 0)} words waiting</p>`
  );
  meters.innerHTML = rows.join("");
}

function pushLine(role, text) {
  const item = document.createElement("p");
  item.className = role === "you" ? "you" : "glass";
  const who = document.createElement("span");
  who.textContent = role === "you" ? "You" : personaLabel();
  item.appendChild(who);
  item.appendChild(document.createTextNode(text));
  log.appendChild(item);
  while (log.children.length > 6) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
}

function personaLabel() {
  if (persona === "lab_assistant") return "Lab";
  if (persona === "feral_bloom") return "Bloom";
  return "Elder";
}

function setStatus(state) {
  lockline.textContent = `${state.name} · ${state.quant} · ${state.license} · ${state.gpu} · context ${state.numCtx}`;
  if (state.ready) {
    statusEl.textContent = `${state.name} is on this machine.`;
  } else if (state.ok) {
    statusEl.textContent = `Ollama is up. Until you run \`${state.pull}\`, the body speaks from its felt state.`;
  } else {
    statusEl.textContent = "Ollama is not running. The body still speaks from its felt state. Start Ollama, then pull the model.";
  }
}

async function refreshStatus() {
  try {
    setStatus(await window.diatom.status());
  } catch (err) {
    statusEl.textContent = "Status unavailable. The body still speaks from its felt state.";
  }
}

// Words become templates. The rule grows the body into each one.
function say(text) {
  for (const word of glyph.wordsOf(text)) queue.push(word);
}

function advanceWords() {
  if (current) {
    current.left -= 1;
    if (current.left > 0) return;
    current = null;
    body.setTemplate(null);
    restLeft = queue.length ? 6 : REST_STEPS;
    wordEl.textContent = "";
    return;
  }
  if (restLeft > 0) {
    restLeft -= 1;
    return;
  }
  if (!queue.length) return;
  const word = queue.shift();
  body.setTemplate(glyph.renderText(word, SIZE));
  current = { word, left: MORPH_STEPS + HOLD_STEPS };
  wordEl.textContent = word;
}

function feel() {
  const desc = body.describe();
  const growth = prevMass == null ? 0 : desc.mass - prevMass;
  let relDrop = 0;
  if (prevMass != null && prevMass > 1e-6 && growth < 0) relDrop = -growth / prevMass;
  prevMass = desc.mass;
  // While the body is rebuilding itself as a word, mass swings are speech, not wounds.
  const speaking = current != null || restLeft > 0;
  rid = morph.stepDrives(
    rid, desc, speaking ? growth * 0.25 : growth, body.audio[0], 0,
    speaking ? 0 : relDrop
  );
  if (woundedRecently > 0) woundedRecently -= 1;
  return { desc, growth, relDrop };
}

let frameNo = 0;
function tick() {
  frameNo += 1;
  for (let i = 0; i < stepsPerFrame; i++) {
    body.step();
    advanceWords();
  }
  if (frameNo % 6 === 0) {
    feel();
    paintMeters();
  }
  for (let k = 0; k < body.audio.length; k++) body.audio[k] *= 0.985;
  requestAnimationFrame(tick);
}

async function talk(userText) {
  const text = userText.trim();
  if (!text || busy) return "";
  busy = true;
  input.disabled = true;
  pushLine("you", text);
  const [fire, energy] = morph.modulationFromText(text);
  body.fireRate = Math.max(0.2, Math.min(0.95, 0.5 * fire));
  body.audio[0] = Math.max(body.audio[0], energy);
  const { desc, growth, relDrop } = feel();
  const instinct = voice.instinctText(rid);
  const preds = voice.predicates(desc, growth, woundedRecently > 0 ? 0.5 : relDrop, rid[2]);
  let said = voice.speakLocal({
    persona, instinct, predicates: preds, userText: text,
    seed: (Date.now() ^ (text.length * 997)) >>> 0,
  });
  try {
    const messages = voice.messagesFor({ persona, instinct, predicates: preds, userText: text, memory });
    const result = await window.diatom.chat({ messages });
    if (result && result.ok && result.text) {
      said = result.text.replace(/\s+/g, " ").trim().slice(0, 400);
    }
  } catch (err) {
    // The felt-state line above is already the answer.
  }
  const [fire2, energy2] = morph.modulationFromText(said);
  body.fireRate = Math.max(0.2, Math.min(0.95, 0.5 * fire2));
  body.audio[0] = Math.max(body.audio[0], energy2);
  view.setSpin(0.0015 + 0.004 * Math.max(rid[0], 0) + 0.003 * Math.max(rid[4], 0));
  memory.push(said);
  if (memory.length > 6) memory = memory.slice(-6);
  pushLine("glass", said);
  say(said);
  paintMeters();
  busy = false;
  input.disabled = false;
  input.focus();
  return said;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value;
  input.value = "";
  talk(text);
});

document.getElementById("cut").addEventListener("click", () => {
  queue = [];
  current = null;
  body.setTemplate(null);
  body.cutHalf();
  woundedRecently = 40;
  const desc = body.describe();
  rid = morph.stepDrives(rid, desc, -0.08, 0, 0, 0.8);
  prevMass = desc.mass;
  talk("A blade just took half of you.");
});

document.getElementById("reseed").addEventListener("click", () => {
  queue = [];
  current = null;
  body.setTemplate(null);
  body.newGenome(0.2);
  prevMass = null;
  rid = [0.2, 0.1, 0, 0.55, 0.1, -0.1];
  talk("Grow a new frustule.");
});

personaEl.addEventListener("change", () => {
  persona = personaEl.value;
});

document.querySelectorAll("[data-say]").forEach((button) => {
  button.addEventListener("click", () => talk(button.getAttribute("data-say")));
});

const speed = document.getElementById("speed");
if (speed) {
  speed.addEventListener("input", () => {
    stepsPerFrame = +speed.value;
  });
}

window.addEventListener("resize", () => view.resize());
paintMeters();
requestAnimationFrame(tick);

refreshStatus().then(async () => {
  if (smoke) {
    // Let the frustule grow, then speak, then wait for the body to be a word.
    await new Promise((r) => setTimeout(r, 3500));
    const reply = await talk("hello");
    // Capture once the body has finished becoming its first word.
    const started = Date.now();
    while (Date.now() - started < 20000) {
      if (current && current.left <= HOLD_STEPS * 0.5) break;
      await new Promise((r) => setTimeout(r, 60));
    }
    await new Promise((r) => setTimeout(r, 120));
    window.diatom.smokeReady({ reply, word: wordEl.textContent, trained: body.trained });
    return;
  }
  // Grow first; speak once the body has a shape.
  setTimeout(() => {
    const { desc, growth } = feel();
    const said = voice.speakLocal({
      persona,
      instinct: voice.instinctText(rid),
      predicates: voice.predicates(desc, growth, 0, rid[2]),
      userText: "",
      seed: 2,
    });
    memory.push(said);
    pushLine("glass", said);
    say(said);
  }, 2600);
});
