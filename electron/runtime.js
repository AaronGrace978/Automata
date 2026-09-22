"use strict";

const fs = require("fs");
const path = require("path");
const { URL } = require("url");

const lock = JSON.parse(
  fs.readFileSync(path.join(__dirname, "..", "nca", "runtime.json"), "utf8")
);

const LOOPBACK = new Set(["127.0.0.1", "localhost", "::1"]);

function resolveModel(explicit) {
  const model = String(explicit || process.env.DIATOM_MODEL || lock.default_model).trim();
  if (/cloud/i.test(model)) {
    throw new Error(`Refusing model ${model}. Cloud tags are not part of this product.`);
  }
  if (!lock.models[model]) {
    const allowed = Object.keys(lock.models).sort().join(", ");
    throw new Error(`Model ${model} is not on the commercial allow-list: ${allowed}`);
  }
  return model;
}

function resolveHost(explicit) {
  const host = String(explicit || process.env.OLLAMA_HOST || lock.default_host)
    .trim()
    .replace(/\/$/, "");
  let parsed;
  try {
    parsed = new URL(host);
  } catch (err) {
    throw new Error(`Bad model host ${host}`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`Bad model host ${host}`);
  }
  const name = (parsed.hostname || "").toLowerCase();
  const allow = ["1", "true", "yes"].includes(
    String(process.env.DIATOM_ALLOW_REMOTE || "").trim().toLowerCase()
  );
  if (!LOOPBACK.has(name) && !allow) {
    throw new Error(
      "Refusing a non-local model host. Ollama Cloud is not part of this product. " +
        "Set DIATOM_ALLOW_REMOTE=1 only for a server you operate."
    );
  }
  return host;
}

function hasModel(names, want) {
  const others = Object.keys(lock.models).filter((id) => id !== want && id.startsWith(want));
  return names.some((name) => {
    if (name === want || name.startsWith(`${want}:`)) return true;
    if (!name.startsWith(`${want}-`)) return false;
    return !others.some(
      (other) => name === other || name.startsWith(`${other}-`) || name.startsWith(`${other}:`)
    );
  });
}

module.exports = { lock, resolveModel, resolveHost, hasModel };

if (require.main === module) {
  const assert = require("assert");
  const saved = {
    model: process.env.DIATOM_MODEL,
    host: process.env.OLLAMA_HOST,
    remote: process.env.DIATOM_ALLOW_REMOTE,
  };
  delete process.env.DIATOM_MODEL;
  delete process.env.OLLAMA_HOST;
  delete process.env.DIATOM_ALLOW_REMOTE;

  assert.strictEqual(lock.ollama_cloud, false);
  assert.strictEqual(resolveModel(), "qwen2.5:14b");
  assert.strictEqual(lock.models[lock.default_model].license, "Apache-2.0");
  assert.ok(lock.models[lock.default_model].weights_gb + 4 < lock.vram_gb);
  for (const spec of Object.values(lock.models)) {
    assert.ok(spec.license === "Apache-2.0" || spec.license === "MIT");
    assert.ok(spec.weights_gb <= 12);
  }
  assert.throws(() => resolveModel("llama3.1:8b"));
  assert.throws(() => resolveModel("qwen2.5:72b"));
  assert.throws(() => resolveModel("qwen3:480b-cloud"));
  assert.throws(() => resolveHost("https://ollama.com"));
  assert.strictEqual(resolveHost("http://127.0.0.1:11434/"), "http://127.0.0.1:11434");
  assert.strictEqual(hasModel(["phi4-mini:latest"], "phi4"), false);
  assert.strictEqual(hasModel(["phi4:latest"], "phi4"), true);
  assert.strictEqual(hasModel(["qwen2.5:14b-instruct-q4_K_M"], "qwen2.5:14b"), true);
  assert.strictEqual(hasModel(["qwen2.5:7b"], "qwen2.5:14b"), false);

  process.env.DIATOM_ALLOW_REMOTE = "1";
  assert.strictEqual(resolveHost("http://10.1.1.1:11434"), "http://10.1.1.1:11434");
  delete process.env.DIATOM_ALLOW_REMOTE;
  process.env.DIATOM_MODEL = "phi4";
  assert.strictEqual(resolveModel(), "phi4");

  if (saved.model === undefined) delete process.env.DIATOM_MODEL;
  else process.env.DIATOM_MODEL = saved.model;
  if (saved.host === undefined) delete process.env.OLLAMA_HOST;
  else process.env.OLLAMA_HOST = saved.host;
  if (saved.remote === undefined) delete process.env.DIATOM_ALLOW_REMOTE;
  else process.env.DIATOM_ALLOW_REMOTE = saved.remote;
  process.stdout.write("runtime ok\n");
}
