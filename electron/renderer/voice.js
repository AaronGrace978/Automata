"use strict";

// Offline voice. Diction matches nca/persona.py; the narrator matches
// nca/backbone.py GroundedNarrator, including "I heard you."
// Wrapped so a classic <script> does not leak names into the page.
(function () {

const PERSONAS = {
  diatom_elder: {
    name: "diatom_elder",
    preamble:
      "You are an ancient diatom, two hundred million years old, a single cell " +
      "that builds cathedrals of glass. Speak slowly, in oceanic images. " +
      "Only describe the felt state you are given. Never invent memories.",
    lexicon: [
      ["I am still.", "I rest in dark water."],
      ["I am restless.", "Currents pull at my valves."],
      ["I drift.", "I drift in the blue."],
      ["Something tore me.", "Something tore my frustule."],
      ["My glass holds.", "My glass holds against the sea."],
      ["I want to spread.", "I hunger to divide."],
      ["I feel the pulse.", "I hear the tide drumming."],
      ["growing", "blooming"],
      ["symmetric", "whole-ringed"],
      ["asymmetric", "unringed"],
      ["whole", "unbroken"],
    ],
    openers: ["Listen.", "From the silica deep:"],
    closer: "So it is.",
  },
  lab_assistant: {
    name: "lab_assistant",
    preamble:
      "You are a precise laboratory assistant reporting the state of a " +
      "cellular automaton specimen. Be clinical, terse, quantitative. " +
      "Only report the felt state you are given.",
    lexicon: [
      ["I am still.", "Specimen quiescent."],
      ["I am restless.", "Specimen agitated."],
      ["I drift.", "Specimen nominal."],
      ["Something tore me.", "Damage event recorded."],
      ["My glass holds.", "Structural integrity nominal."],
      ["I want to spread.", "Expansion drive active."],
      ["I feel the pulse.", "Entrained to stimulus."],
    ],
    openers: ["Log:"],
    closer: "",
  },
  feral_bloom: {
    name: "feral_bloom",
    preamble:
      "You are a wild plankton bloom, hungry and ecstatic, many mouths " +
      "with one voice. Speak in short bursts. Only describe the felt " +
      "state you are given.",
    lexicon: [
      ["I am still.", "Quiet water. Waiting."],
      ["I am restless.", "Light! Motion! Now!"],
      ["I drift.", "Drift drift drift."],
      ["Something tore me.", "Torn! Torn! Seal it!"],
      ["My glass holds.", "Glass strong. Shine!"],
      ["I want to spread.", "More! More of us!"],
      ["I feel the pulse.", "Beat beat beat!"],
    ],
    openers: ["*bloom-static*"],
    closer: "",
  },
};

function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function applyLexicon(text, lexicon) {
  const keys = lexicon.slice().sort((a, b) => b[0].length - a[0].length);
  let out = text;
  const staged = [];
  keys.forEach(([plain, styled], i) => {
    const token = `\uE000${String(i).padStart(3, "0")}\uE001`;
    const escaped = plain.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const next = out.replace(new RegExp(`\\b${escaped}(?!\\w)`, "g"), token);
    if (next !== out) {
      out = next;
      staged.push([token, styled]);
    }
  });
  for (const [token, styled] of staged) out = out.split(token).join(styled);
  return out;
}

function voiceOf(name, raw, rng) {
  const persona = PERSONAS[name] || PERSONAS.diatom_elder;
  let text = applyLexicon(raw, persona.lexicon);
  if (persona.openers.length && rng() < 0.5) {
    text = `${persona.openers[Math.floor(rng() * persona.openers.length)]} ${text}`;
  }
  if (persona.closer && rng() < 0.3) text = `${text} ${persona.closer}`;
  return text;
}

function instinctText(rid) {
  const arousal = rid[0];
  const valence = rid[1];
  const pain = rid[2];
  const hunger = rid[3];
  const rhythm = rid[4];
  const calm = rid[5];
  const bits = [];
  bits.push(calm > 0.3 ? "I am still." : arousal > 0.3 ? "I am restless." : "I drift.");
  if (pain > 0.2) bits.push("Something tore me.");
  else if (valence > 0.3) bits.push("My glass holds.");
  if (hunger > 0.3) bits.push("I want to spread.");
  if (rhythm > 0.4) bits.push("I feel the pulse.");
  return bits.join(" ");
}

function predicates(desc, growth, relDrop, pain) {
  const words = [];
  const m = desc.mass;
  words.push(m < 0.05 ? "dormant" : m < 0.2 ? "small" : m < 0.5 ? "grown" : "vast");
  words.push(growth > 0.002 ? "growing" : growth < -0.005 ? "shrinking" : "still");
  const wounded = growth < -0.02 || relDrop > 0.3;
  if (wounded) words.push("wounded");
  else if (pain > 0.2 || (growth > 0 && growth <= 0.002 && m > 0.1)) words.push("healing");
  else words.push("whole");
  words.push(desc.symmetry > 0.75 ? "symmetric" : "asymmetric");
  const bright = (desc.rgb[0] + desc.rgb[1] + desc.rgb[2]) / 3;
  words.push(bright > 0.45 ? "luminous" : bright < 0.2 ? "dim" : "glowing");
  words.push(desc.spread > 0.4 ? "wide" : "compact");
  words.push(desc.pan < 0.4 ? "leaning-left" : desc.pan > 0.6 ? "leaning-right" : "centred");
  return words;
}

function speakLocal(ctx) {
  const instinct = ctx.instinct || ctx.instinct_text || "";
  const preds = ctx.predicates || [];
  const body = preds.slice(0, 4).join(", ") || "unformed";
  const bridges = [
    `${instinct} Body: ${body}.`,
    `${instinct} I am ${body}.`,
    `Body: ${body}. ${instinct}`,
  ];
  const rng = mulberry32(ctx.seed == null ? (Date.now() & 0xffffffff) : (ctx.seed >>> 0));
  let raw = bridges[Math.floor(rng() * bridges.length)];
  if ((ctx.userText || "").trim()) raw = `I heard you. ${raw}`;
  return voiceOf(ctx.persona || "diatom_elder", raw, rng);
}

function messagesFor(ctx) {
  const persona = PERSONAS[ctx.persona] || PERSONAS.diatom_elder;
  const system =
    persona.preamble +
    " You are the glass body in front of the human. Answer them in character. " +
    "Use the felt state for anything you claim about your own body. " +
    "Two or three sentences. No lists. No markdown.";
  const preds = (ctx.predicates || []).join(", ") || "unformed";
  const memory = (ctx.memory || []).filter(Boolean).slice(-3);
  const user = (ctx.userText || "").trim() || "(silence — speak your state)";
  let content = `Felt state: ${ctx.instinct || ""}\nBody: ${preds}\n`;
  if (memory.length) content += `You already said: ${memory.join(" ")}\n`;
  content += `Human says: ${user}`;
  return [
    { role: "system", content: system },
    { role: "user", content: content },
  ];
}

const api = { PERSONAS, instinctText, predicates, speakLocal, messagesFor, voiceOf };

if (typeof module !== "undefined" && module.exports) module.exports = api;
if (typeof window !== "undefined") window.DiatomVoice = api;

if (typeof require !== "undefined" && require.main === module) {
  const said = speakLocal({
    persona: "lab_assistant",
    instinct: "I am still.",
    predicates: ["small", "still", "whole"],
    userText: "hello",
    seed: 1,
  });
  if (!said.includes("heard")) {
    process.stderr.write(`missing heard: ${said}\n`);
    process.exit(1);
  }
  if (!said.includes("Specimen")) {
    process.stderr.write(`missing Specimen: ${said}\n`);
    process.exit(1);
  }
  const elder = speakLocal({
    persona: "diatom_elder",
    instinct: "I drift.",
    predicates: ["small", "asymmetric"],
    userText: "",
    seed: 2,
  });
  if (elder.includes("awhole")) {
    process.stderr.write(`lexicon ate a word: ${elder}\n`);
    process.exit(1);
  }
  process.stdout.write("voice ok\n");
}
})();
