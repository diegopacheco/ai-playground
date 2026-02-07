export function generateNavbar(): string {
  return `<nav style="display:flex;justify-content:space-between;align-items:center;padding:16px 32px;background:var(--bg);border-bottom:1px solid rgba(0,0,0,0.1);">
  <div class="brand" style="font-size:1.4rem;font-weight:700;color:var(--primary);">MyApp</div>
  <div class="nav-links" style="display:flex;gap:24px;">
    <a href="#" style="color:var(--text);text-decoration:none;">Home</a>
    <a href="#" style="color:var(--text);text-decoration:none;">About</a>
    <a href="#" style="color:var(--text);text-decoration:none;">Services</a>
    <a href="#" style="color:var(--text);text-decoration:none;">Contact</a>
  </div>
</nav>`;
}
