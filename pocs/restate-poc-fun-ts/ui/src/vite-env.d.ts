interface ImportMetaEnv {
  readonly VITE_RESTATE_ADMIN: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "*.css";
