import { ReactNode } from "react";
import {
  HeaderContainer,
  HeaderInner,
  HeaderTitle,
  HeaderSubtitle,
  HeaderInfo,
  HeaderActions,
} from "./header.styles";

export interface HeaderProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}

export function Header({ title, subtitle, actions }: HeaderProps) {
  return (
    <HeaderContainer>
      <HeaderInner>
        <HeaderInfo>
          <HeaderTitle>🛍️ {title}</HeaderTitle>
          {subtitle && <HeaderSubtitle>{subtitle}</HeaderSubtitle>}
        </HeaderInfo>
        {actions && <HeaderActions>{actions}</HeaderActions>}
      </HeaderInner>
    </HeaderContainer>
  );
}
