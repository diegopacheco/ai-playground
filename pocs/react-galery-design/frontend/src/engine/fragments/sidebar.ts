export function generateSidebar(): string {
  return `<div class="sidebar-layout" style="display:flex;min-height:400px;">
  <aside style="width:250px;background:var(--bg);border-right:1px solid rgba(0,0,0,0.1);padding:24px;">
    <h3 style="margin-bottom:16px;color:var(--text);font-size:1rem;">Navigation</h3>
    <div style="display:flex;flex-direction:column;gap:8px;">
      <a href="#" style="padding:8px 12px;border-radius:var(--radius);color:var(--text);">Dashboard</a>
      <a href="#" style="padding:8px 12px;border-radius:var(--radius);background:var(--primary);color:#fff;">Projects</a>
      <a href="#" style="padding:8px 12px;border-radius:var(--radius);color:var(--text);">Settings</a>
      <a href="#" style="padding:8px 12px;border-radius:var(--radius);color:var(--text);">Reports</a>
      <a href="#" style="padding:8px 12px;border-radius:var(--radius);color:var(--text);">Help</a>
    </div>
  </aside>
  <main style="flex:1;padding:32px;">
    <h2 style="margin-bottom:16px;color:var(--text);">Projects</h2>
    <p style="color:var(--secondary);">Select a project from the sidebar to view its details, or create a new one to get started.</p>
  </main>
</div>`;
}
