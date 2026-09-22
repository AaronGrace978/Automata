import { mount } from "./diatom3d.js";

const morph = window.DiatomMorph;
const voice = window.DiatomVoice;
  const canvas = document.getElementById("view");
  const log = document.getElementById("log");
  const form = document.getElementById("composer");
  const input = document.getElementById("say");
  const statusEl = document.getElementById("status");
  const lockline = document.getElementById("lockline");
  const personaEl = document.getElementById("persona");
  const meters = document.getElementById("meters");
  const smoke = new URLSearchParams(location.search).has("smoke");

  const view = mount(canvas);
  let rid = [0.05, 0.15, 0, 0.25, 0, 0.2];
  let fold = 8;
  let damagePulse = 0;
  let persona = "diatom_elder";
  let memory = [];
  let busy = false;
  let target = morph.poseFromState(rid, "", "", fold, 0);

  const METER_KEYS = [
    ["arousal", 0],
    ["valence", 1],
    ["pain", 2],
    ["hunger", 3],
    ["rhythm", 4],
    ["calm", 5],
  ];

  function bodyDesc(pose) {
    return {
      mass: 0.18 + 0.45 * (1 - pose.damage) * (0.35 + 0.65 * pose.pores),
      rgb: [0.62, 0.48, 0.28],
      spread: pose.girdle,
      pan: 0.5,
      symmetry: 1 - pose.asymmetry,
    };
  }

  function paintMeters() {
    const pose = target;
    const rows = METER_KEYS.map(([name, index]) => {
      const value = (rid[index] + 1) / 2;
      return `<div class="meter"><span>${name}</span><div class="bar"><div style="width:${Math.round(value * 100)}%"></div></div></div>`;
    });
    rows.push(
      `<div class="meter"><span>damage</span><div class="bar"><div style="width:${Math.round(pose.damage * 100)}%"></div></div></div>`
    );
    rows.push(`<p class="fold">genome ${pose.folds}-fold</p>`);
    meters.innerHTML = rows.join("");
  }

  function retarget(utterance, userText) {
    target = morph.poseFromState(rid, utterance, userText, fold, damagePulse);
    view.setTarget(target);
    paintMeters();
  }

  function pushLine(role, text) {
    const item = document.createElement("p");
    item.className = role === "you" ? "you" : "glass";
    const who = document.createElement("span");
    who.textContent = role === "you" ? "You" : personaLabel();
    item.appendChild(who);
    item.appendChild(document.createTextNode(text));
    log.appendChild(item);
    while (log.children.length > 8) log.removeChild(log.firstChild);
    log.scrollTop = log.scrollHeight;
  }

  function personaLabel() {
    if (persona === "lab_assistant") return "Lab";
    if (persona === "feral_bloom") return "Bloom";
    return "Elder";
  }

  function setStatus(state) {
    const spec = `${state.name} · ${state.quant} · ${state.license}`;
    lockline.textContent = `${spec} · ${state.gpu} · context ${state.numCtx}`;
    if (state.ready) {
      statusEl.textContent = `${state.name} is on this machine.`;
      return;
    }
    if (state.ok) {
      statusEl.textContent = `Ollama is up. Until you run \`${state.pull}\`, the glass speaks from its body.`;
      return;
    }
    statusEl.textContent = "Ollama is not running. The glass still speaks from its body. Start Ollama, then pull the model.";
  }

  async function refreshStatus() {
    try {
      setStatus(await window.diatom.status());
    } catch (err) {
      statusEl.textContent = "Status unavailable. The glass still speaks from its body.";
    }
  }

  async function talk(userText) {
    const text = userText.trim();
    if (!text || busy) return "";
    busy = true;
    input.disabled = true;
    pushLine("you", text);
    const [fire, energy] = morph.modulationFromText(text);
    const relDrop = damagePulse > 0.4 ? 0.5 : 0;
    rid = morph.stepDrives(rid, bodyDesc(target), (fire - 1) * 0.04, energy, 0, relDrop);
    retarget("", text);
    const instinct = voice.instinctText(rid);
    const preds = voice.predicates(bodyDesc(target), (fire - 1) * 0.04, relDrop, rid[2]);
    const local = voice.speakLocal({
      persona,
      instinct,
      predicates: preds,
      userText: text,
      seed: (Date.now() ^ (text.length * 997)) >>> 0,
    });
    let said = local;
    let via = "body";
    try {
      const messages = voice.messagesFor({
        persona,
        instinct,
        predicates: preds,
        userText: text,
        memory,
      });
      const result = await window.diatom.chat({ messages });
      if (result && result.ok && result.text) {
        said = result.text.replace(/\s+/g, " ").trim().slice(0, 600);
        via = result.model;
      }
    } catch (err) {
      via = "body";
    }
    const [fire2, energy2] = morph.modulationFromText(said);
    rid = morph.stepDrives(rid, bodyDesc(target), (fire2 - 1) * 0.03, energy2, 0, 0);
    if (damagePulse > 0) damagePulse *= 0.55;
    retarget(said, text);
    memory.push(said);
    if (memory.length > 6) memory = memory.slice(-6);
    pushLine("glass", said);
    statusEl.dataset.via = via;
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
    damagePulse = 1;
    rid = morph.stepDrives(rid, bodyDesc(target), -0.08, 0, 0, 0.8);
    retarget("Something tore me.", "");
    talk("A blade just took half of you.");
  });

  document.getElementById("reseed").addEventListener("click", () => {
    const folds = [6, 8, 10, 12, 16];
    fold = folds[Math.floor(Math.random() * folds.length)];
    damagePulse = 0;
    rid = [0.2, 0.1, 0, 0.55, 0.1, -0.1];
    retarget("I want to spread.", "");
    talk("Grow a new frustule.");
  });

  personaEl.addEventListener("change", () => {
    persona = personaEl.value;
  });

  document.querySelectorAll("[data-say]").forEach((button) => {
    button.addEventListener("click", () => talk(button.getAttribute("data-say")));
  });

  setInterval(() => {
    if (damagePulse > 0.001) damagePulse *= 0.86;
    if (rid[2] > 0.05 || damagePulse > 0.001) {
      rid = morph.stepDrives(rid, bodyDesc(target), 0, 0, 0, damagePulse > 0.3 ? 0.35 : 0);
      retarget("", "");
    }
  }, 700);

  retarget("", "");
  window.addEventListener("resize", () => view.resize());

  refreshStatus().then(async () => {
    if (smoke) {
      const reply = await talk("hello");
      await new Promise((resolve) => setTimeout(resolve, 800));
      window.diatom.smokeReady({ reply, folds: target.folds });
      return;
    }
    const instinct = voice.instinctText(rid);
    const said = voice.speakLocal({
      persona,
      instinct,
      predicates: voice.predicates(bodyDesc(target), 0, 0, rid[2]),
      userText: "",
      seed: 2,
    });
    memory.push(said);
    pushLine("glass", said);
    retarget(said, "");
  });
