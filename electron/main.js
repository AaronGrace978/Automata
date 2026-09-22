"use strict";

const http = require("http");
const https = require("https");
const path = require("path");
const fs = require("fs");
const { app, BrowserWindow, ipcMain } = require("electron");
const { lock, resolveModel, resolveHost, hasModel } = require("./runtime");

const smoke = process.argv.includes("--smoke");

if (smoke || process.env.DIATOM_SWIFTSHADER === "1") {
  app.commandLine.appendSwitch("use-gl", "angle");
  app.commandLine.appendSwitch("use-angle", "swiftshader");
  app.commandLine.appendSwitch("enable-unsafe-swiftshader");
}

function requestJson(host, method, pathname, body, timeoutMs) {
  const url = new URL(pathname, host.endsWith("/") ? host : `${host}/`);
  const lib = url.protocol === "https:" ? https : http;
  const payload = body ? JSON.stringify(body) : null;
  return new Promise((resolve, reject) => {
    const req = lib.request(
      url,
      {
        method,
        headers: payload
          ? {
              "Content-Type": "application/json",
              "Content-Length": Buffer.byteLength(payload),
            }
          : {},
        timeout: timeoutMs,
      },
      (res) => {
        const chunks = [];
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          if (res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`Ollama ${res.statusCode}: ${text.slice(0, 300)}`));
            return;
          }
          try {
            resolve(text ? JSON.parse(text) : {});
          } catch (err) {
            reject(err);
          }
        });
      }
    );
    req.on("error", reject);
    req.on("timeout", () => req.destroy(new Error("Ollama timed out")));
    if (payload) req.write(payload);
    req.end();
  });
}

function sanitizeMessages(messages) {
  if (!Array.isArray(messages) || messages.length < 1 || messages.length > 12) {
    throw new Error("bad messages");
  }
  return messages.map((message) => {
    if (!message || !["system", "user", "assistant"].includes(message.role)) {
      throw new Error("bad role");
    }
    if (typeof message.content !== "string" || message.content.length > 4000) {
      throw new Error("bad content");
    }
    return { role: message.role, content: message.content };
  });
}

function statusPayload(extra) {
  let model = lock.default_model;
  try {
    model = resolveModel();
  } catch (err) {
    extra = Object.assign({ error: err.message }, extra || {});
  }
  const spec = lock.models[model];
  return Object.assign(
    {
      ok: false,
      ready: false,
      model,
      name: spec.name,
      license: spec.license,
      quant: spec.quant,
      weightsGb: spec.weights_gb,
      pull: spec.pull,
      cloud: false,
      numCtx: lock.num_ctx,
      gpu: lock.gpu,
    },
    extra || {}
  );
}

ipcMain.handle("diatom:status", async () => {
  let host;
  let model;
  try {
    host = resolveHost();
    model = resolveModel();
  } catch (err) {
    return statusPayload({ error: err.message });
  }
  try {
    const tags = await requestJson(host, "GET", "/api/tags", null, 2500);
    const names = (tags.models || []).map((row) => row.name || row.model || "");
    return statusPayload({ ok: true, ready: hasModel(names, model), host });
  } catch (err) {
    return statusPayload({ ok: false, ready: false, error: err.message, host });
  }
});

ipcMain.handle("diatom:chat", async (_event, payload) => {
  let host;
  let model;
  try {
    host = resolveHost();
    model = resolveModel();
    const messages = sanitizeMessages(payload && payload.messages);
    const data = await requestJson(
      host,
      "POST",
      "/api/chat",
      {
        model,
        messages,
        stream: false,
        options: {
          temperature: lock.temperature,
          num_predict: lock.num_predict,
          num_ctx: lock.num_ctx,
        },
      },
      smoke ? 4000 : lock.timeout_ms
    );
    const text = ((data.message || {}).content || "").trim();
    if (!text) throw new Error("empty model reply");
    return { ok: true, text, model };
  } catch (err) {
    return { ok: false, error: err.message, model: model || lock.default_model };
  }
});

let win;

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 880,
    minHeight: 640,
    backgroundColor: "#070b10",
    title: "Diatom",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.loadFile(path.join(__dirname, "renderer", "index.html"), {
    query: smoke ? { smoke: "1" } : {},
  });
  if (smoke) {
    win.webContents.on("console-message", (_event, _level, message) => {
      console.log(`renderer: ${message}`);
    });
  }
}

if (smoke) {
  ipcMain.on("diatom:smoke-ready", async (_event, info) => {
    try {
      const image = await win.webContents.capturePage();
      const png = image.toPNG();
      const dest = process.env.DIATOM_SMOKE_PNG || "/tmp/diatom-smoke.png";
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.writeFileSync(dest, png);
      console.log(`smoke png ${dest}`);
      const reply = info && info.reply ? String(info.reply) : "";
      if (reply.length < 3) {
        console.error("smoke reply was empty");
        process.exit(1);
      }
      console.log(`smoke reply: ${reply.slice(0, 180)}`);
      console.log(`smoke word: ${info && info.word ? info.word : "(none)"} trained: ${info && info.trained}`);
      process.exit(0);
    } catch (err) {
      console.error(err);
      process.exit(1);
    }
  });
  setTimeout(() => {
    console.error("smoke timed out");
    process.exit(1);
  }, 30000).unref?.();
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => app.quit());
