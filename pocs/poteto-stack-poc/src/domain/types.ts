export type PlaybookId = string & { readonly brand: "PlaybookId" };

export type Playbook = {
  id: PlaybookId;
  title: string;
  steps: string[];
};

export type PrincipleGroup = "Core" | "Architecture" | "Verification" | "Delegation" | "Meta";

export type Principle = {
  name: string;
  group: PrincipleGroup;
  when: string;
};

export type Catalog = {
  version: string;
  skills: string[];
  playbooks: Playbook[];
  principles: Principle[];
};
