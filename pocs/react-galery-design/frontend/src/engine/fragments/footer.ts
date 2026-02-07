export function generateFooter(): string {
  return `<footer style="background:var(--primary);color:#fff;padding:48px 32px 24px;">
  <div class="footer-grid" style="max-width:1200px;margin:0 auto;display:grid;grid-template-columns:repeat(3,1fr);gap:32px;">
    <div>
      <h4 style="margin-bottom:12px;">About</h4>
      <p style="font-size:0.9rem;opacity:0.8;">We build tools that empower people to create amazing things. Our mission is to simplify the complex.</p>
    </div>
    <div>
      <h4 style="margin-bottom:12px;">Links</h4>
      <div style="display:flex;flex-direction:column;gap:8px;">
        <a href="#" style="color:#fff;opacity:0.8;font-size:0.9rem;">Home</a>
        <a href="#" style="color:#fff;opacity:0.8;font-size:0.9rem;">About</a>
        <a href="#" style="color:#fff;opacity:0.8;font-size:0.9rem;">Services</a>
        <a href="#" style="color:#fff;opacity:0.8;font-size:0.9rem;">Contact</a>
      </div>
    </div>
    <div>
      <h4 style="margin-bottom:12px;">Contact</h4>
      <p style="font-size:0.9rem;opacity:0.8;">hello@myapp.com</p>
      <p style="font-size:0.9rem;opacity:0.8;">+1 (555) 123-4567</p>
    </div>
  </div>
  <div style="text-align:center;margin-top:32px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.2);font-size:0.85rem;opacity:0.7;">
    &copy; 2026 MyApp. All rights reserved.
  </div>
</footer>`;
}
