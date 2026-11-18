import * as restate from "@restatedev/restate-sdk";
import { readConfig, servicePort } from "./config.ts";
import { agentWorkflow } from "./workflow.ts";

const config = readConfig();
const port = servicePort();
restate.serve({ services: [agentWorkflow], port });
console.log(`agent-service on port ${port} using ${config.provider} ${config.model}`);
