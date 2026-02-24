const { contextBridge, ipcRenderer } = require("electron");

function subscribe(channel, callback) {
  const listener = (_event, payload) => callback(payload);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

contextBridge.exposeInMainWorld("traderApi", {
  startBot: (config) => ipcRenderer.invoke("bot:start", config),
  stopBot: () => ipcRenderer.invoke("bot:stop"),
  manualEntry: (payload) => ipcRenderer.invoke("bot:manual-entry", payload),
  getSnapshot: () => ipcRenderer.invoke("bot:snapshot"),
  getLiveBankroll: () => ipcRenderer.invoke("bot:get-live-bankroll"),
  onStatus: (callback) => subscribe("bot:status", callback),
  onLog: (callback) => subscribe("bot:log", callback),
  onMetrics: (callback) => subscribe("bot:metrics", callback),
  onLiveBankroll: (callback) => subscribe("bot:live-bankroll", callback),
});
