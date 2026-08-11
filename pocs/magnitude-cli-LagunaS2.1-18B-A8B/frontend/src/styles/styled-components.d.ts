import "styled-components";
import { AppTheme } from "@/styles/theme";

declare module "styled-components" {
  export interface DefaultTheme extends AppTheme {}
}
