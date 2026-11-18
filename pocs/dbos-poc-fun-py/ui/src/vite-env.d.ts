declare module "*.css";

interface ImportMetaEnv {
  readonly VITE_DASHBOARD_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
