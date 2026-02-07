export function generateContactForm(): string {
  return `<section style="padding:48px 32px;">
  <div style="max-width:600px;margin:0 auto;">
    <h2 style="text-align:center;margin-bottom:32px;color:var(--text);">Contact Us</h2>
    <form style="display:flex;flex-direction:column;gap:16px;">
      <input type="text" placeholder="Your Name" style="padding:12px 16px;border:1px solid rgba(0,0,0,0.15);border-radius:var(--radius);font-size:1rem;" />
      <input type="email" placeholder="Your Email" style="padding:12px 16px;border:1px solid rgba(0,0,0,0.15);border-radius:var(--radius);font-size:1rem;" />
      <textarea placeholder="Your Message" rows="5" style="padding:12px 16px;border:1px solid rgba(0,0,0,0.15);border-radius:var(--radius);font-size:1rem;resize:vertical;"></textarea>
      <button type="submit" style="padding:14px 24px;font-size:1rem;">Send Message</button>
    </form>
  </div>
</section>`;
}
