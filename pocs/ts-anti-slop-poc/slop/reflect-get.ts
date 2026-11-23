type Config = { readonly port: number };

export function port(config: Config): number {
  return Reflect.get(config, "port");
}
