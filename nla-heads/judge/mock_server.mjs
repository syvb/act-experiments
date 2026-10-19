// Minimal stand-in for POST /v1/messages, to exercise judge.mjs without an API key.
import http from "node:http";
import fs from "node:fs";
let n = 0;
http.createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    const j = JSON.parse(body);
    if (n++ === 0) fs.writeFileSync("mock_first_request.json", JSON.stringify({ path: req.url, headers: req.headers, body: j }, null, 1));
    const prompt = j.messages[0].content;
    const afc = prompt.includes("<passage_A>");
    // every 97th call returns an out-of-range score, to exercise the validation path
    const text = afc ? JSON.stringify({ choice: n % 2 ? "A" : "B", confidence: "low" })
      : JSON.stringify({ matching_elements: [], score: n % 97 === 0 ? 7 : n % 4 });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({
      id: "msg_mock" + n, type: "message", role: "assistant", model: j.model,
      content: [{ type: "thinking", thinking: "", signature: "x" }, { type: "text", text }],
      stop_reason: "end_turn", stop_sequence: null,
      usage: { input_tokens: 900, output_tokens: 120 },
    }));
  });
}).listen(18765, () => console.log("mock listening"));
