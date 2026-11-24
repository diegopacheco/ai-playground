import { fileURLToPath } from "node:url";
import { connect } from "./connect.ts";

const serverEntry = fileURLToPath(new URL("../src/main.ts", import.meta.url));
const topic = Bun.argv[2] ?? "electric bikes europe";
const print = (line: string) => process.stdout.write(line + "\n");

const conn = await connect(serverEntry);
try {
  print(`protocol ${conn.client.getNegotiatedProtocolVersion()}`);
  const { skills } = await conn.skills.list();
  print(`skills published by the server: ${skills.length}`);
  for (const skill of skills) print(`  ${skill.uri}`);
  const first = skills[0];
  if (first) {
    const file = await conn.skills.read(first);
    print(`loaded ${file.uri} verified=${file.verified}`);
    print((file.body ?? "").trim().split("\n").slice(0, 3).join("\n"));
  }

  const handle = await conn.tasks.callTool("market_report", { topic, depth: "deep" });
  print(`task ${handle.taskId} status=${handle.status} pollIntervalMs=${handle.pollIntervalMs}`);
  let last = "";
  const done = await conn.tasks.waitUntilDone(handle, (t) => {
    const line = `  ${t.status}: ${t.statusMessage ?? ""}`;
    if (line !== last) print(line);
    last = line;
  });
  if (done.status === "completed") print(JSON.stringify(done.result, null, 2));
  else print(`task ended ${done.status}`);

  const doomed = await conn.tasks.callTool("market_report", { topic: "fax machines", depth: "deep" });
  await conn.tasks.cancel(doomed.taskId);
  print(`task ${doomed.taskId} after cancel: ${(await conn.tasks.get(doomed.taskId)).status}`);
} finally {
  await conn.close();
}
