// node >=18
import fetch from "node-fetch";
const BASE = "http://localhost:8000";

async function querySSE(q) {
  const res = await fetch(`${BASE}/query?stream=true`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: q, think_mode: "on", include_reasoning: true })
  });
  res.body.setEncoding("utf8");
  res.body.on("data", chunk => process.stdout.write(chunk));
}

querySSE(process.argv[2] || "각 단계 핵심?");