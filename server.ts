import express from "express";
import { spawn } from "child_process";
import { createProxyMiddleware } from "http-proxy-middleware";
import http from "http";

const app = express();
const PORT = 3000;
const STREAMLIT_PORT = 8501;

console.log("Starting Streamlit Python server on port", STREAMLIT_PORT);

const streamlitProcess = spawn(
  "streamlit",
  [
    "run",
    "app.py",
    `--server.port=${STREAMLIT_PORT}`,
    "--server.address=127.0.0.1",
    "--server.headless=true",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false",
    "--browser.gatherUsageStats=false"
  ],
  {
    stdio: "inherit",
    env: { ...process.env, PYTHONUNBUFFERED: "1" }
  }
);

streamlitProcess.on("error", (err) => {
  console.error("Failed to start Streamlit process:", err);
});

streamlitProcess.on("exit", (code, signal) => {
  console.log(`Streamlit process exited with code ${code} and signal ${signal}`);
});

process.on("exit", () => {
  streamlitProcess.kill();
});

// Proxy all requests to Streamlit server
const streamlitProxy = createProxyMiddleware({
  target: `http://127.0.0.1:${STREAMLIT_PORT}`,
  changeOrigin: true,
  ws: true
});

app.use("/", streamlitProxy);

const server = http.createServer(app);

// Handle websocket upgrades for Streamlit live communication
server.on("upgrade", (req, socket, head) => {
  (streamlitProxy as any).upgrade(req, socket, head);
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Express proxy listening on port ${PORT} -> Streamlit on port ${STREAMLIT_PORT}`);
});
