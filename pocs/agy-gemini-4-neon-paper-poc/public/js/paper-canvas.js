const PAGE_W = 1280;
const PAGE_H = 1760;

function seededRandom(seed) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function wrapLines(ctx, text, maxWidth) {
  const words = text.split(/\s+/);
  const lines = [];
  let line = '';
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function drawPaperBase(ctx, theme, pageNumber) {
  const rand = seededRandom(pageNumber * 937 + 1924);
  const grad = ctx.createRadialGradient(
    PAGE_W * 0.5,
    PAGE_H * 0.45,
    PAGE_W * 0.12,
    PAGE_W * 0.5,
    PAGE_H * 0.5,
    PAGE_H * 0.72
  );
  grad.addColorStop(0, theme.paper);
  grad.addColorStop(1, theme.paperDark);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, PAGE_W, PAGE_H);

  ctx.save();
  for (let i = 0; i < 2400; i++) {
    const x = rand() * PAGE_W;
    const y = rand() * PAGE_H;
    const w = 1 + rand() * 3.5;
    const h = 1 + rand() * 1.4;
    ctx.fillStyle = rand() > 0.5 ? 'rgba(65,45,22,0.035)' : 'rgba(255,250,238,0.04)';
    ctx.fillRect(x, y, w, h);
  }
  for (let i = 0; i < 110; i++) {
    const x = rand() * PAGE_W;
    const y = rand() * PAGE_H;
    const r = 1 + rand() * 3.2;
    ctx.fillStyle = `rgba(110,78,38,${0.03 + rand() * 0.045})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  const isOdd = pageNumber % 2 === 1;
  const spineGrad = isOdd
    ? ctx.createLinearGradient(0, 0, 110, 0)
    : ctx.createLinearGradient(PAGE_W, 0, PAGE_W - 110, 0);
  spineGrad.addColorStop(0, 'rgba(35,24,12,0.22)');
  spineGrad.addColorStop(0.45, 'rgba(35,24,12,0.07)');
  spineGrad.addColorStop(1, 'rgba(35,24,12,0)');
  ctx.fillStyle = spineGrad;
  ctx.fillRect(isOdd ? 0 : PAGE_W - 110, 0, 110, PAGE_H);

  const foldGrad = ctx.createLinearGradient(0, PAGE_H * 0.48, 0, PAGE_H * 0.52);
  foldGrad.addColorStop(0, 'rgba(40,28,14,0)');
  foldGrad.addColorStop(0.5, 'rgba(40,28,14,0.08)');
  foldGrad.addColorStop(1, 'rgba(40,28,14,0)');
  ctx.fillStyle = foldGrad;
  ctx.fillRect(0, PAGE_H * 0.48, PAGE_W, PAGE_H * 0.04);

  ctx.strokeStyle = 'rgba(40,28,14,0.16)';
  ctx.lineWidth = 2;
  ctx.strokeRect(24, 24, PAGE_W - 48, PAGE_H - 48);
}

function drawDoubleRule(ctx, x1, x2, y, color, thick = 4, thin = 1.5, gap = 6) {
  ctx.strokeStyle = color;
  ctx.lineWidth = thick;
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x2, y);
  ctx.stroke();

  ctx.lineWidth = thin;
  ctx.beginPath();
  ctx.moveTo(x1, y + gap);
  ctx.lineTo(x2, y + gap);
  ctx.stroke();
}

function drawEngravingHatch(ctx, x, y, w, h, step, color) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  for (let i = -h; i < w + h; i += step) {
    ctx.beginPath();
    ctx.moveTo(x + i, y);
    ctx.lineTo(x + i - h * 0.45, y + h);
    ctx.stroke();
  }
  ctx.restore();
}

function drawIllustration(ctx, illustration, theme) {
  if (!illustration) return;
  const b = illustration.box;
  const x = Math.round(b.x * PAGE_W);
  const y = Math.round(b.y * PAGE_H);
  const w = Math.round(b.w * PAGE_W);
  const h = Math.round(b.h * PAGE_H);
  const capH = 44;
  const imgH = h - capH;

  ctx.save();
  ctx.fillStyle = 'rgba(28,20,12,0.06)';
  ctx.fillRect(x, y, w, imgH);
  ctx.strokeStyle = theme.rule;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, imgH);
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 5, y + 5, w - 10, imgH - 10);

  ctx.save();
  ctx.beginPath();
  ctx.rect(x + 6, y + 6, w - 12, imgH - 12);
  ctx.clip();

  const ix = x + 6;
  const iy = y + 6;
  const iw = w - 12;
  const ih = imgH - 12;

  drawEngravingHatch(ctx, ix, iy, iw, ih * 0.56, 7, 'rgba(30,22,14,0.12)');

  if (illustration.kind === 'airship') {
    ctx.strokeStyle = theme.ink;
    ctx.fillStyle = theme.ink;
    const sunX = ix + iw * 0.78;
    const sunY = iy + ih * 0.25;
    for (let a = 0; a < Math.PI * 2; a += Math.PI / 14) {
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sunX + Math.cos(a) * 24, sunY + Math.sin(a) * 24);
      ctx.lineTo(sunX + Math.cos(a) * 165, sunY + Math.sin(a) * 165);
      ctx.stroke();
    }
    ctx.fillStyle = theme.paper;
    ctx.beginPath();
    ctx.arc(sunX, sunY, 22, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.stroke();

    const cx = ix + iw * 0.45;
    const cy = iy + ih * 0.38;
    ctx.fillStyle = theme.paper;
    ctx.beginPath();
    ctx.ellipse(cx, cy, iw * 0.32, ih * 0.16, -0.05, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.lineWidth = 1.2;
    for (let k = -2; k <= 2; k++) {
      ctx.beginPath();
      ctx.ellipse(cx, cy, iw * 0.32, ih * (0.045 * Math.abs(k) + 0.02), -0.05, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = theme.ink;
    ctx.fillRect(cx - iw * 0.07, cy + ih * 0.16, iw * 0.14, ih * 0.055);
    ctx.beginPath();
    ctx.moveTo(cx - iw * 0.28, cy - ih * 0.04);
    ctx.lineTo(cx - iw * 0.36, cy - ih * 0.14);
    ctx.lineTo(cx - iw * 0.22, cy - ih * 0.11);
    ctx.closePath();
    ctx.fill();

    const waterY = iy + ih * 0.72;
    ctx.fillStyle = theme.ink;
    for (let bx = ix; bx < ix + iw; bx += 42) {
      const bh = 28 + ((bx * 37) % 68);
      ctx.fillRect(bx, waterY - bh, 32, bh);
      ctx.fillStyle = theme.paper;
      for (let wy = waterY - bh + 8; wy < waterY - 8; wy += 12) {
        ctx.fillRect(bx + 6, wy, 6, 6);
        ctx.fillRect(bx + 18, wy, 6, 6);
      }
      ctx.fillStyle = theme.ink;
    }
    ctx.lineWidth = 2;
    for (let r = 0; r < 6; r++) {
      const yy = waterY + 10 + r * 16;
      ctx.beginPath();
      for (let wx = ix; wx <= ix + iw; wx += 16) {
        const wy = yy + Math.sin((wx + r * 20) * 0.06) * 3.5;
        if (wx === ix) ctx.moveTo(wx, wy);
        else ctx.lineTo(wx, wy);
      }
      ctx.stroke();
    }
  } else if (illustration.kind === 'locomotive') {
    const groundY = iy + ih * 0.78;
    ctx.strokeStyle = theme.ink;
    ctx.fillStyle = theme.ink;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(ix + iw * 0.28, groundY, iw * 0.24, Math.PI, 0);
    ctx.stroke();
    ctx.lineWidth = 1.5;
    for (let a = Math.PI; a <= Math.PI * 2; a += 0.16) {
      ctx.beginPath();
      ctx.moveTo(ix + iw * 0.28 + Math.cos(a) * iw * 0.24, groundY + Math.sin(a) * iw * 0.24);
      ctx.lineTo(ix + iw * 0.28 + Math.cos(a) * iw * 0.29, groundY + Math.sin(a) * iw * 0.29);
      ctx.stroke();
    }
    for (let i = 0; i < 8; i++) {
      const sx = ix + iw * (0.28 + i * 0.075);
      const sy = iy + ih * (0.34 - Math.sin(i * 0.6) * 0.14);
      ctx.fillStyle = theme.paper;
      ctx.beginPath();
      ctx.arc(sx, sy, 18 + i * 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.fillStyle = theme.ink;
    ctx.fillRect(ix + iw * 0.34, groundY - ih * 0.30, iw * 0.42, ih * 0.22);
    ctx.fillRect(ix + iw * 0.64, groundY - ih * 0.42, iw * 0.14, ih * 0.34);
    ctx.fillRect(ix + iw * 0.39, groundY - ih * 0.45, iw * 0.05, ih * 0.16);
    for (let wIdx = 0; wIdx < 4; wIdx++) {
      const wx = ix + iw * (0.41 + wIdx * 0.105);
      ctx.fillStyle = theme.paper;
      ctx.beginPath();
      ctx.arc(wx, groundY - 12, 20, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 4;
      ctx.stroke();
    }
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(ix, groundY + 12);
    ctx.lineTo(ix + iw, groundY + 12);
    ctx.stroke();
  } else if (illustration.kind === 'press') {
    ctx.strokeStyle = theme.ink;
    ctx.fillStyle = theme.ink;
    const c1x = ix + iw * 0.32;
    const c2x = ix + iw * 0.65;
    const cy = iy + ih * 0.48;
    const r = Math.min(iw, ih) * 0.23;
    [c1x, c2x].forEach((cx) => {
      ctx.fillStyle = theme.paper;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 4;
      ctx.stroke();
      for (let a = 0; a < Math.PI * 2; a += Math.PI / 8) {
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * (r * 0.28), cy + Math.sin(a) * (r * 0.28));
        ctx.lineTo(cx + Math.cos(a) * (r + 10), cy + Math.sin(a) * (r + 10));
        ctx.stroke();
      }
      ctx.fillStyle = theme.ink;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 0.22, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(ix + 20, iy + ih * 0.22);
    ctx.lineTo(c1x, cy - r);
    ctx.lineTo(c2x, cy + r);
    ctx.lineTo(ix + iw - 20, iy + ih * 0.78);
    ctx.stroke();
    ctx.strokeRect(ix + iw * 0.14, iy + ih * 0.78, iw * 0.72, ih * 0.12);
  } else if (illustration.kind === 'observatory') {
    ctx.fillStyle = theme.ink;
    ctx.fillRect(ix, iy, iw, ih * 0.64);
    ctx.fillStyle = theme.paper;
    const rand = seededRandom(77);
    for (let i = 0; i < 95; i++) {
      const sx = ix + rand() * iw;
      const sy = iy + rand() * ih * 0.58;
      ctx.fillRect(sx, sy, rand() > 0.8 ? 3 : 1.5, rand() > 0.8 ? 3 : 1.5);
    }
    const nx = ix + iw * 0.72;
    const ny = iy + ih * 0.28;
    ctx.strokeStyle = theme.paper;
    ctx.lineWidth = 1.6;
    for (let arm = 0; arm < 2; arm++) {
      ctx.beginPath();
      for (let t = 0; t < 4.5; t += 0.15) {
        const ang = t + arm * Math.PI;
        const rad = t * 14;
        const px = nx + Math.cos(ang) * rad * 1.4;
        const py = ny + Math.sin(ang) * rad * 0.65;
        if (t === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    const domeX = ix + iw * 0.34;
    const domeY = iy + ih * 0.72;
    ctx.fillStyle = theme.paper;
    ctx.strokeStyle = theme.ink;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(domeX, domeY, iw * 0.18, Math.PI, 0);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.fillRect(domeX - iw * 0.19, domeY, iw * 0.38, ih * 0.24);
    ctx.strokeRect(domeX - iw * 0.19, domeY, iw * 0.38, ih * 0.24);
    ctx.lineWidth = 8;
    ctx.strokeStyle = theme.ink;
    ctx.beginPath();
    ctx.moveTo(domeX + 10, domeY - 22);
    ctx.lineTo(domeX + iw * 0.22, domeY - ih * 0.36);
    ctx.stroke();
  } else if (illustration.kind === 'chart') {
    ctx.strokeStyle = 'rgba(35,24,14,0.25)';
    ctx.lineWidth = 1;
    for (let gx = ix + 40; gx < ix + iw; gx += 64) {
      ctx.beginPath();
      ctx.moveTo(gx, iy + 18);
      ctx.lineTo(gx, iy + ih - 24);
      ctx.stroke();
    }
    for (let gy = iy + 24; gy < iy + ih - 18; gy += 48) {
      ctx.beginPath();
      ctx.moveTo(ix + 24, gy);
      ctx.lineTo(ix + iw - 24, gy);
      ctx.stroke();
    }
    ctx.strokeStyle = theme.ink;
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    const pts = [0.74, 0.68, 0.71, 0.56, 0.60, 0.45, 0.49, 0.35, 0.39, 0.26, 0.22, 0.18];
    pts.forEach((p, idx) => {
      const px = ix + 34 + (idx / (pts.length - 1)) * (iw - 68);
      const py = iy + 22 + p * (ih - 54);
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    pts.forEach((p, idx) => {
      const px = ix + 34 + (idx / (pts.length - 1)) * (iw - 68);
      const py = iy + 22 + Math.min(0.9, p + 0.14) * (ih - 54);
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = theme.ink;
    ctx.font = 'bold 18px Georgia, serif';
    ctx.fillText('INDEX COMPOSITE  1920 — 1924', ix + 34, iy + 36);
  } else if (illustration.kind === 'crossword') {
    const n = 9;
    const size = Math.min(iw - 36, ih - 36);
    const cell = size / n;
    const ox = ix + (iw - size) / 2;
    const oy = iy + (ih - size) / 2;
    const blocks = new Set(['0,4', '1,4', '2,2', '2,6', '3,7', '4,0', '4,4', '4,8', '5,1', '6,2', '6,6', '7,4', '8,4']);
    let num = 1;
    ctx.font = 'bold 13px Georgia, serif';
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const cx = ox + c * cell;
        const cy = oy + r * cell;
        if (blocks.has(`${r},${c}`)) {
          ctx.fillStyle = theme.ink;
          ctx.fillRect(cx, cy, cell, cell);
        } else {
          ctx.fillStyle = theme.paper;
          ctx.fillRect(cx, cy, cell, cell);
          ctx.strokeStyle = theme.ink;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(cx, cy, cell, cell);
          const needsNum = r === 0 || c === 0 || blocks.has(`${r - 1},${c}`) || blocks.has(`${r},${c - 1}`);
          if (needsNum) {
            ctx.fillStyle = theme.ink;
            ctx.fillText(String(num++), cx + 4, cy + 14);
          }
        }
      }
    }
  } else if (illustration.kind === 'adbox') {
    ctx.strokeStyle = theme.ink;
    ctx.lineWidth = 5;
    ctx.strokeRect(ix + 14, iy + 14, iw - 28, ih - 28);
    ctx.lineWidth = 1.5;
    ctx.strokeRect(ix + 22, iy + 22, iw - 44, ih - 44);
    ctx.fillStyle = theme.ink;
    ctx.textAlign = 'center';
    ctx.font = 'bold 34px "Playfair Display", Georgia, serif';
    ctx.fillText('FINEST BRASS & INK', ix + iw / 2, iy + 78);
    ctx.font = 'italic 22px Georgia, serif';
    ctx.fillText('Warranted Solid Spun Metal & Emerald Glass', ix + iw / 2, iy + 114);
    const lx = ix + iw / 2;
    const ly = iy + ih * 0.57;
    ctx.fillStyle = theme.accent;
    ctx.beginPath();
    ctx.ellipse(lx, ly - 32, 74, 28, 0, Math.PI, 0);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = theme.ink;
    ctx.fillRect(lx - 6, ly - 32, 12, 82);
    ctx.beginPath();
    ctx.ellipse(lx, ly + 52, 54, 14, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = 'bold 24px Georgia, serif';
    ctx.fillText('EST. 1884 • CORTLANDT STREET', ix + iw / 2, iy + ih - 42);
    ctx.textAlign = 'left';
  } else if (illustration.kind === 'barometer') {
    const cx = ix + iw / 2;
    const cy = iy + ih / 2 + 6;
    const r = Math.min(iw, ih) * 0.39;
    ctx.fillStyle = theme.paper;
    ctx.strokeStyle = theme.ink;
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r - 12, 0, Math.PI * 2);
    ctx.stroke();
    for (let i = 0; i < 36; i++) {
      const a = (i / 36) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * (r - 24), cy + Math.sin(a) * (r - 24));
      ctx.lineTo(cx + Math.cos(a) * (r - 12), cy + Math.sin(a) * (r - 12));
      ctx.stroke();
    }
    ctx.fillStyle = theme.ink;
    ctx.textAlign = 'center';
    ctx.font = 'bold 18px Georgia, serif';
    ctx.fillText('CHANGE', cx, cy - r * 0.56);
    ctx.fillText('RAIN', cx - r * 0.52, cy - r * 0.14);
    ctx.fillText('FAIR', cx + r * 0.52, cy - r * 0.14);
    ctx.fillText('30.28 IN.', cx, cy + r * 0.54);
    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(cx - 18, cy + 16);
    ctx.lineTo(cx + r * 0.56, cy - r * 0.36);
    ctx.stroke();
    ctx.fillStyle = theme.ink;
    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.textAlign = 'left';
  }

  ctx.restore();

  ctx.fillStyle = theme.inkMuted;
  ctx.font = 'italic 18px "Libre Baskerville", Georgia, serif';
  const capLines = wrapLines(ctx, illustration.caption || '', w - 8);
  capLines.slice(0, 2).forEach((line, i) => {
    ctx.fillText(line, x + 2, y + imgH + 22 + i * 20);
  });
  ctx.restore();
}

function drawFrontMasthead(ctx, edition, page) {
  const { theme } = edition;
  const margin = 62;
  const topY = 58;

  ctx.strokeStyle = theme.rule;
  ctx.lineWidth = 2;
  ctx.strokeRect(margin, topY, 210, 96);
  ctx.strokeRect(PAGE_W - margin - 210, topY, 210, 96);

  ctx.fillStyle = theme.ink;
  ctx.textAlign = 'center';
  ctx.font = 'bold 15px "Libre Baskerville", Georgia, serif';
  ctx.fillText(edition.city, margin + 105, topY + 34);
  ctx.font = 'italic 14px Georgia, serif';
  ctx.fillText('WEATHER: CLEAR', margin + 105, topY + 60);
  ctx.fillText('BAROMETER 30.28', margin + 105, topY + 80);

  ctx.font = 'bold 16px "Libre Baskerville", Georgia, serif';
  ctx.fillText(edition.price, PAGE_W - margin - 105, topY + 36);
  ctx.font = 'italic 13px Georgia, serif';
  const mottoLines = wrapLines(ctx, edition.motto, 185);
  mottoLines.slice(0, 2).forEach((l, i) => {
    ctx.fillText(l, PAGE_W - margin - 105, topY + 60 + i * 18);
  });

  ctx.fillStyle = theme.ink;
  ctx.font = '74px "UnifrakturMaguntia", "Playfair Display", Georgia, serif';
  const maxMastW = PAGE_W - margin * 2 - 450;
  let mastSize = 74;
  while (ctx.measureText(edition.masthead).width > maxMastW && mastSize > 42) {
    mastSize -= 3;
    ctx.font = `${mastSize}px "UnifrakturMaguntia", "Playfair Display", Georgia, serif`;
  }
  ctx.fillText(edition.masthead, PAGE_W / 2, topY + 74);

  drawDoubleRule(ctx, margin, PAGE_W - margin, 174, theme.rule, 4, 1.5, 6);

  ctx.font = 'bold 17px "Libre Baskerville", Georgia, serif';
  ctx.textAlign = 'left';
  ctx.fillText(edition.issue, margin, 204);
  ctx.textAlign = 'center';
  ctx.fillText(edition.date, PAGE_W / 2, 204);
  ctx.textAlign = 'right';
  ctx.fillText(page.section, PAGE_W - margin, 204);

  drawDoubleRule(ctx, margin, PAGE_W - margin, 218, theme.rule, 2, 1.5, 5);

  ctx.textAlign = 'center';
  ctx.fillStyle = theme.ink;
  let bannerSize = 62;
  ctx.font = `900 ${bannerSize}px "Playfair Display", Georgia, serif`;
  while (ctx.measureText(page.banner).width > PAGE_W - margin * 2 - 20 && bannerSize > 34) {
    bannerSize -= 2;
    ctx.font = `900 ${bannerSize}px "Playfair Display", Georgia, serif`;
  }
  ctx.fillText(page.banner, PAGE_W / 2, 294);

  ctx.fillStyle = theme.inkMuted;
  ctx.font = 'italic 24px "Libre Baskerville", Georgia, serif';
  const deckLines = wrapLines(ctx, page.deck, PAGE_W - margin * 2 - 60);
  deckLines.slice(0, 2).forEach((line, idx) => {
    ctx.fillText(line, PAGE_W / 2, 336 + idx * 30);
  });

  drawDoubleRule(ctx, margin, PAGE_W - margin, 390, theme.rule, 2, 1, 5);

  if (page.editionStamp) {
    ctx.save();
    ctx.translate(PAGE_W - 186, 444);
    ctx.rotate(-0.08);
    ctx.fillStyle = 'rgba(239,228,204,0.88)';
    ctx.fillRect(-84, -20, 168, 40);
    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 3;
    ctx.strokeRect(-84, -20, 168, 40);
    ctx.lineWidth = 1;
    ctx.strokeRect(-79, -15, 158, 30);
    ctx.fillStyle = theme.accent;
    ctx.font = 'bold 16px "Libre Baskerville", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(page.editionStamp, 0, 6);
    ctx.restore();
  }

  ctx.textAlign = 'left';
}

function drawInnerHeader(ctx, edition, page) {
  const { theme } = edition;
  const margin = 62;

  ctx.fillStyle = theme.inkMuted;
  ctx.font = 'bold 17px "Libre Baskerville", Georgia, serif';
  ctx.textAlign = 'left';
  ctx.fillText(page.runningHead, margin, 84);
  ctx.textAlign = 'right';
  ctx.fillText(`PAGE ${page.number}`, PAGE_W - margin, 84);

  drawDoubleRule(ctx, margin, PAGE_W - margin, 102, theme.rule, 3, 1, 5);

  ctx.textAlign = 'center';
  ctx.fillStyle = theme.ink;
  let bannerSize = 48;
  ctx.font = `900 ${bannerSize}px "Playfair Display", Georgia, serif`;
  while (ctx.measureText(page.banner).width > PAGE_W - margin * 2 - 20 && bannerSize > 28) {
    bannerSize -= 2;
    ctx.font = `900 ${bannerSize}px "Playfair Display", Georgia, serif`;
  }
  ctx.fillText(page.banner, PAGE_W / 2, 172);

  ctx.fillStyle = theme.inkMuted;
  ctx.font = 'italic 22px "Libre Baskerville", Georgia, serif';
  const deckLines = wrapLines(ctx, page.deck, PAGE_W - margin * 2 - 40);
  deckLines.slice(0, 2).forEach((line, idx) => {
    ctx.fillText(line, PAGE_W / 2, 212 + idx * 28);
  });

  drawDoubleRule(ctx, margin, PAGE_W - margin, 268, theme.rule, 2, 1, 5);
  ctx.textAlign = 'left';
}

function drawArticle(ctx, article, theme) {
  const b = article.box;
  const x = Math.round(b.x * PAGE_W);
  const y = Math.round(b.y * PAGE_H);
  const w = Math.round(b.w * PAGE_W);
  const h = Math.round(b.h * PAGE_H);

  ctx.save();
  ctx.beginPath();
  ctx.rect(x - 4, y - 4, w + 8, h + 8);
  ctx.clip();

  ctx.strokeStyle = 'rgba(40,28,16,0.38)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + w, y);
  ctx.stroke();

  let cursorY = y + 26;

  ctx.fillStyle = theme.accent;
  ctx.font = 'bold 15px "Libre Baskerville", Georgia, serif';
  ctx.fillText(article.kicker, x, cursorY);
  cursorY += 32;

  ctx.fillStyle = theme.ink;
  const headSize = w > 520 ? 32 : 25;
  ctx.font = `bold ${headSize}px "Playfair Display", Georgia, serif`;
  const headLines = wrapLines(ctx, article.headline, w - 4);
  for (const line of headLines.slice(0, 3)) {
    ctx.fillText(line, x, cursorY);
    cursorY += headSize + 5;
  }

  ctx.fillStyle = theme.inkMuted;
  ctx.font = 'italic 16px Georgia, serif';
  ctx.fillText(article.byline, x, cursorY + 2);
  cursorY += 16;

  ctx.strokeStyle = 'rgba(40,28,16,0.45)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, cursorY);
  ctx.lineTo(x + Math.min(110, w * 0.45), cursorY);
  ctx.stroke();
  cursorY += 26;

  ctx.fillStyle = theme.ink;
  const bodySize = 19;
  const lineH = 26;
  ctx.font = `${bodySize}px "Libre Baskerville", Georgia, serif`;

  article.paragraphs.forEach((para, pIdx) => {
    if (cursorY > y + h - 18) return;
    if (pIdx === 0 && para.length > 2) {
      const firstChar = para[0];
      const rest = para.slice(1);
      ctx.save();
      ctx.font = 'bold 58px "Playfair Display", Georgia, serif';
      ctx.fillStyle = theme.ink;
      const dropW = ctx.measureText(firstChar).width + 10;
      ctx.fillText(firstChar, x, cursorY + 34);
      ctx.restore();

      const firstTwo = wrapLines(ctx, rest, Math.max(80, w - dropW - 6));
      const consumed = firstTwo.slice(0, 2);
      consumed.forEach((ln) => {
        if (cursorY <= y + h - 14) {
          ctx.fillText(ln, x + dropW, cursorY);
          cursorY += lineH;
        }
      });
      const remainingText = firstTwo.slice(2).join(' ');
      if (remainingText) {
        const restLines = wrapLines(ctx, remainingText, w - 6);
        for (const ln of restLines) {
          if (cursorY > y + h - 14) break;
          ctx.fillText(ln, x, cursorY);
          cursorY += lineH;
        }
      }
      cursorY += 10;
    } else {
      const lines = wrapLines(ctx, para, w - 6);
      for (const ln of lines) {
        if (cursorY > y + h - 14) break;
        ctx.fillText(ln, x, cursorY);
        cursorY += lineH;
      }
      cursorY += 10;
    }
  });

  ctx.restore();
}

export function renderPageCanvas(edition, pageIndex) {
  const page = edition.pages[pageIndex];
  const canvas = document.createElement('canvas');
  canvas.width = PAGE_W;
  canvas.height = PAGE_H;
  const ctx = canvas.getContext('2d');

  drawPaperBase(ctx, edition.theme, page.number);

  if (pageIndex === 0) {
    drawFrontMasthead(ctx, edition, page);
  } else {
    drawInnerHeader(ctx, edition, page);
  }

  drawIllustration(ctx, page.illustration, edition.theme);

  for (const article of page.articles) {
    drawArticle(ctx, article, edition.theme);
  }

  ctx.fillStyle = edition.theme.inkMuted;
  ctx.font = '14px Georgia, serif';
  ctx.textAlign = 'center';
  ctx.fillText(
    `${edition.masthead}  •  ${edition.date}  •  SHEET ${Math.ceil(page.number / 2)}  •  PAGE ${page.number}`,
    PAGE_W / 2,
    PAGE_H - 36
  );

  return canvas;
}

export function createDeskWoodTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d');
  const rand = seededRandom(42);

  const grad = ctx.createLinearGradient(0, 0, 1024, 1024);
  grad.addColorStop(0, '#2a1810');
  grad.addColorStop(0.5, '#3b2216');
  grad.addColorStop(1, '#22130c');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 1024, 1024);

  for (let y = 0; y < 1024; y += 2) {
    const wave = Math.sin(y * 0.018) * 14 + Math.cos(y * 0.007) * 24;
    const alpha = 0.04 + rand() * 0.09;
    ctx.strokeStyle = rand() > 0.35 ? `rgba(18,8,4,${alpha})` : `rgba(120,74,42,${alpha * 0.7})`;
    ctx.lineWidth = 1 + rand() * 2.2;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(340, y + wave, 680, y - wave, 1024, y + wave * 0.4);
    ctx.stroke();
  }

  ctx.strokeStyle = 'rgba(20,10,5,0.22)';
  ctx.lineWidth = 3;
  ctx.strokeRect(44, 44, 936, 936);

  return canvas;
}
