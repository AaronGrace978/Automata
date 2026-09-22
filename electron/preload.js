"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("diatom", {
  chat: (payload) => ipcRenderer.invoke("diatom:chat", payload),
  status: () => ipcRenderer.invoke("diatom:status"),
  emoji: (concept) => ipcRenderer.invoke("diatom:emoji", concept),
  smokeReady: (info) => ipcRenderer.send("diatom:smoke-ready", info),
});
