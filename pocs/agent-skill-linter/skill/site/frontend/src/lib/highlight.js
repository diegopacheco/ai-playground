const TOKEN = /(\/\/[^\n]*|#[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)|(\b\d[\d_.xXa-fA-F]*\b)|(@\w+)|(\b(?:abstract|assert|async|await|boolean|break|byte|case|catch|char|class|const|continue|default|do|double|else|enum|export|extends|final|finally|float|for|from|function|if|implements|import|in|instanceof|int|interface|let|long|new|null|of|package|private|protected|public|record|return|short|static|super|switch|synchronized|this|throw|throws|true|false|try|undefined|var|void|while|yield)\b)/g;

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function highlightLine(line) {
  let out = '';
  let last = 0;
  let match;
  TOKEN.lastIndex = 0;
  while ((match = TOKEN.exec(line))) {
    out += escapeHtml(line.slice(last, match.index));
    if (match[1]) out += '<span class="tok-com">' + escapeHtml(match[1]) + '</span>';
    else if (match[2]) out += '<span class="tok-str">' + escapeHtml(match[2]) + '</span>';
    else if (match[3]) out += '<span class="tok-num">' + escapeHtml(match[3]) + '</span>';
    else if (match[4]) out += '<span class="tok-ann">' + escapeHtml(match[4]) + '</span>';
    else if (match[5]) out += '<span class="tok-kw">' + escapeHtml(match[5]) + '</span>';
    last = TOKEN.lastIndex;
  }
  out += escapeHtml(line.slice(last));
  return out;
}
