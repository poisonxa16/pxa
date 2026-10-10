// agent-markdown.js: syntax highlighting and KaTeX math on rendered replies, shared with Classic
// through window.PXAMarkdown. KaTeX 0.16.22 is vendored at /vendor/katex (MIT, LICENSE beside the files).
// No CDN. A parse error falls back to the original delimiter text. $...$ and $$...$$ both render.
// Register with window.PXAMarkdown.add(fn(root, ctx)): it then runs in BOTH views.
(function () {
  "use strict";
  const M = window.PXAMarkdown;
  if (!M) return;

  const KW = /^(?:def|class|return|if|else|elif|for|while|import|from|as|with|try|except|finally|lambda|yield|pass|break|continue|True|False|None|and|or|not|in|is|print|function|const|let|var|async|await|new|this|typeof|return)$/;
  const TOK = /(`(?:\\.|[^`])*`|'''[\s\S]*?'''|"""[\s\S]*?"""|'(?:\\.|[^'\n])*'|"(?:\\.|[^"\n])*"|\/\/[^\n]*|#.*$|\/\*[\s\S]*?\*\/|\b[A-Za-z_][A-Za-z0-9_]*\b|\b\d+(?:\.\d+)?\b)/gm;

  function kind(bit) {
    const c = bit[0];
    if (c === "'" || c === '"' || c === "`") return "str";
    if (c === "#" || bit.slice(0, 2) === "//" || bit.slice(0, 2) === "/*") return "com";
    if (c >= "0" && c <= "9") return "num";
    if (KW.test(bit)) return "kw";
    return "";
  }
  function highlight(code) {
    const text = code.textContent || "";
    if (!text || text.length > 30000) return;
    const frag = document.createDocumentFragment();
    let last = 0, m;
    TOK.lastIndex = 0;
    while ((m = TOK.exec(text))) {
      const k = kind(m[0]);
      if (!k) continue;
      if (m.index > last) frag.append(document.createTextNode(text.slice(last, m.index)));
      const sp = document.createElement("span");
      sp.className = "ag-tok ag-tok-" + k;
      sp.textContent = m[0];
      frag.append(sp);
      last = m.index + m[0].length;
    }
    if (!frag.childNodes.length) return;
    if (last < text.length) frag.append(document.createTextNode(text.slice(last)));
    code.replaceChildren(frag);
  }

  function mathNode(src, display) {
    const sp = document.createElement("span");
    sp.className = "ag-math" + (display ? " ag-math-block" : "");
    const raw = (display ? "$$" : "$") + src + (display ? "$$" : "$");
    sp.setAttribute("data-tex", String(src).slice(0, 500));
    const K = window.katex;
    if (!K || typeof K.render !== "function") {
      sp.classList.add("ag-math-raw");
      sp.textContent = raw;
      return sp;
    }
    try {
      K.render(src, sp, {throwOnError: true, displayMode: !!display, output: "html", trust: false});
    } catch (e) {
      sp.classList.add("ag-math-raw");
      sp.textContent = raw;
    }
    return sp;
  }
  function splitMath(node) {
    const s = node.nodeValue;
    const re = /\$\$([^$]{1,800})\$\$|\$([^$\n]{1,400})\$/g;
    let m, last = 0, any = false;
    const frag = document.createDocumentFragment();
    while ((m = re.exec(s))) {
      const inner = m[1] != null ? m[1] : m[2];
      const display = m[1] != null;
      if (!inner.trim()) continue;
      if (!display && !/[\\^=_]/.test(inner)) continue;
      any = true;
      if (m.index > last) frag.append(document.createTextNode(s.slice(last, m.index)));
      frag.append(mathNode(inner, display));
      last = m.index + m[0].length;
    }
    if (!any) return;
    if (last < s.length) frag.append(document.createTextNode(s.slice(last)));
    node.parentNode.replaceChild(frag, node);
  }
  function mathify(root) {
    const nodes = [];
    (function walk(n) {
      if (!n) return;
      if (n.nodeType === 1) {
        const tag = n.tagName;
        if (tag === "PRE" || tag === "CODE" || tag === "SCRIPT" || tag === "STYLE" || tag === "TEXTAREA") return;
        if (n.classList.contains("ag-math")) return;
        Array.from(n.childNodes).forEach(walk);
        return;
      }
      if (n.nodeType === 3 && n.nodeValue && n.nodeValue.indexOf("$") >= 0) nodes.push(n);
    })(root);
    nodes.forEach(splitMath);
  }

  M.add(function (root) {
    if (!root || !root.querySelectorAll) return;
    root.querySelectorAll("pre code").forEach(highlight);
    mathify(root);
  });
})();
