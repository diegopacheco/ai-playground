export function generateResponsiveCSS(): string {
  return `
@media (max-width: 1024px) {
  .grid-3 {
    grid-template-columns: repeat(2, 1fr) !important;
  }
  .grid-4 {
    grid-template-columns: repeat(2, 1fr) !important;
  }
}

@media (max-width: 768px) {
  .grid-3, .grid-2, .grid-4 {
    grid-template-columns: 1fr !important;
  }
  nav {
    flex-direction: column !important;
    gap: 1rem;
  }
  nav .nav-links {
    flex-direction: column;
    gap: 0.5rem;
    align-items: center;
  }
  .sidebar-layout {
    flex-direction: column !important;
  }
  .sidebar-layout aside {
    width: 100% !important;
    border-right: none !important;
    border-bottom: 1px solid #ddd;
  }
  .pricing-grid {
    grid-template-columns: 1fr !important;
  }
  .stats-row {
    grid-template-columns: repeat(2, 1fr) !important;
  }
  footer .footer-grid {
    grid-template-columns: 1fr !important;
  }
  .hero-section {
    padding: 40px 20px !important;
  }
  table {
    font-size: 0.85rem;
  }
}

@media (max-width: 480px) {
  .stats-row {
    grid-template-columns: 1fr !important;
  }
  body {
    font-size: 14px;
  }
}
`;
}
