import styled from "styled-components";

export const TableWrapper = styled.div`
  overflow-x: auto;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  box-shadow: ${({ theme }) => theme.colors.shadow};
`;

export const StyledTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: ${({ theme }) => theme.fontSize.sm};

  th,
  td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  }

  th {
    background-color: ${({ theme }) => theme.colors.background};
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: ${({ theme }) => theme.fontSize.xs};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
  }

  tbody tr {
    transition: background-color 0.15s ease;
  }

  tbody tr:hover {
    background-color: ${({ theme }) => theme.colors.background};
  }

  tbody tr:last-child td {
    border-bottom: none;
  }
`;

export const StockBadge = styled.span<{ $inStock: boolean }>`
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.5rem;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  font-size: ${({ theme }) => theme.fontSize.xs};
  font-weight: 500;

  ${({ $inStock, theme }) =>
    $inStock
      ? `
        background-color: ${theme.colors.success}15;
        color: ${theme.colors.success};
      `
      : `
        background-color: ${theme.colors.danger}15;
        color: ${theme.colors.danger};
      `}
`;

export const ActionButton = styled.button`
  background: transparent;
  border: none;
  cursor: pointer;
  padding: 0.25rem 0.5rem;
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 0.875rem;
  transition: all 0.2s ease;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;

  &:hover {
    background-color: ${({ theme }) => theme.colors.background};
    color: ${({ theme }) => theme.colors.text};
  }

  &.danger:hover {
    color: ${({ theme }) => theme.colors.danger};
  }
`;

export const ActionCell = styled.td`
  display: flex;
  gap: 0.25rem;
  white-space: nowrap;
`;

export const PriceCell = styled.td`
  font-weight: 500;
  color: ${({ theme }) => theme.colors.text};
`;

export const EmptyState = styled.div`
  text-align: center;
  padding: 3rem 1rem;
  color: ${({ theme }) => theme.colors.textSecondary};

  svg {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.3;
  }
`;
