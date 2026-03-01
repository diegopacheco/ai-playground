import { Mastra } from "@mastra/core/mastra";
import { chefAgent } from "./agents/chef-agent";

export const mastra = new Mastra({
  agents: { chefAgent },
});
