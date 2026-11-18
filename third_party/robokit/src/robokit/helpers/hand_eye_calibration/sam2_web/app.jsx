/* global React, ReactDOM */
const { useState, useEffect, useRef, useCallback } = React;

const TWEAKS = /*EDITMODE-BEGIN*/{
  "checkerboard": true,
  "wsUrl": `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`
}/*EDITMODE-END*/;

// ─── atoms ───────────────────────────────────────────────────────────────────
const Ic = ({ children, size = 15, sw = 1.5 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
       stroke="currentColor" strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    {children}
  </svg>
);
const IPlus   = () => <Ic><path d="M12 5v14M5 12h14"/></Ic>;
const IMinus  = () => <Ic><path d="M5 12h14"/></Ic>;
const IBox    = () => <Ic><rect x="4" y="4" width="16" height="16"/></Ic>;
const IReset  = () => <Ic><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></Ic>;
const IUpload = () => <Ic><path d="M12 16V4M6 10l6-6 6 6"/><path d="M4 20h16"/></Ic>;

function Kbd({ children }) {
  return (
    <span className="mono" style={{
      display: "inline-block", minWidth: 16, padding: "1px 5px",
      border: "1px solid var(--hair)",
      borderRadius: 3, color: "var(--text-mute)",
      fontSize: 10.5, lineHeight: 1.3, textAlign: "center",
    }}>{children}</span>
  );
}

// ─── top bar ─────────────────────────────────────────────────────────────────
function TopBar({ status, wsUrl, onReconnect, imgMeta, caption }) {
  const color = { open: "var(--ok)", connecting: "var(--warn)", closed: "var(--err)" }[status];
  const label = { open: "connected", connecting: "reconnecting", closed: "disconnected" }[status];
  return (
    <header style={{
      height: 40, flex: "none",
      display: "flex", alignItems: "center", gap: 14,
      padding: "0 14px",
      borderBottom: "1px solid var(--hair)",
      background: "var(--bg-2)",
    }}>
      <div style={{ fontSize: 12, fontWeight: 500, color: "var(--text)" }}>
        SAM2 <span style={{ color: "var(--text-mute)", fontWeight: 400 }}>annotator</span>
      </div>

      {caption ? (
        <div style={{
          fontSize: 13, fontWeight: 600, color: "var(--text)",
          padding: "3px 10px", borderRadius: 4, background: "var(--bg-3, #2a2a2a)",
          border: "1px solid var(--hair)", letterSpacing: 0.2,
        }}>{caption}</div>
      ) : null}

      <div style={{ width: 1, height: 16, background: "var(--hair)" }}/>

      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{
          width: 7, height: 7, borderRadius: 99, background: color,
          display: "inline-block",
        }}/>
        <span style={{ fontSize: 12, color: "var(--text-dim)" }}>{label}</span>
        <span className="mono" style={{ fontSize: 11, color: "var(--text-mute)" }}>{wsUrl}</span>
        {status !== "open" && (
          <button onClick={onReconnect} style={{
            background: "transparent", border: "none",
            color: "var(--text-dim)", padding: "2px 6px",
            fontSize: 11, cursor: "pointer", textDecoration: "underline",
            textUnderlineOffset: 2, textDecorationColor: "var(--hair)",
          }}>retry</button>
        )}
      </div>

      <div style={{ flex: 1 }}/>

      {imgMeta && (
        <div className="mono" style={{ fontSize: 11, color: "var(--text-mute)" }}>
          {imgMeta.w}×{imgMeta.h}
        </div>
      )}
    </header>
  );
}

// ─── mode switcher (tabs, inline at top of canvas) ───────────────────────────
const MODES = [
  { id: "pos", label: "Positive", short: "1", icon: <IPlus />,  accent: "var(--pos)" },
  { id: "neg", label: "Negative", short: "2", icon: <IMinus />, accent: "var(--neg)" },
  { id: "box", label: "Box",      short: "3", icon: <IBox />,   accent: "var(--box)" },
];

function ModeTabs({ mode, setMode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {MODES.map(m => {
        const active = mode === m.id;
        return (
          <button key={m.id} onClick={() => setMode(m.id)}
            title={`${m.label} (${m.short})`}
            style={{
              display: "flex", alignItems: "center", gap: 10,
              padding: active ? "8px 10px 8px 7px" : "8px 10px",
              background: active ? "var(--panel-2)" : "var(--bg)",
              borderTop: "1px solid " + (active ? "color-mix(in oklab, white 18%, transparent)" : "var(--hair)"),
              borderRight: "1px solid " + (active ? "color-mix(in oklab, white 18%, transparent)" : "var(--hair)"),
              borderBottom: "1px solid " + (active ? "color-mix(in oklab, white 18%, transparent)" : "var(--hair)"),
              borderLeft: active ? `3px solid ${m.accent}` : "1px solid var(--hair)",
              borderRadius: 4,
              color: active ? m.accent : "var(--text-dim)",
              fontWeight: active ? 500 : 400,
              fontSize: 12.5, cursor: "pointer",
              textAlign: "left",
              transition: "background .12s, border-color .12s, color .12s",
            }}
            onMouseEnter={e => {
              if (active) return;
              e.currentTarget.style.background = "var(--panel-2)";
              e.currentTarget.style.color = "var(--text)";
              e.currentTarget.style.borderColor = "color-mix(in oklab, white 12%, transparent)";
            }}
            onMouseLeave={e => {
              if (active) return;
              e.currentTarget.style.background = "var(--bg)";
              e.currentTarget.style.color = "var(--text-dim)";
              e.currentTarget.style.borderColor = "var(--hair)";
            }}>
            <span style={{ display: "grid", placeItems: "center", width: 15, opacity: active ? 1 : 0.8 }}>{m.icon}</span>
            <span>{m.label}</span>
            <span className="mono" style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-mute)" }}>{m.short}</span>
          </button>
        );
      })}
    </div>
  );
}

// ─── canvas ──────────────────────────────────────────────────────────────────
function CanvasArea({ mode, image, mask, points, box, dragBox,
                      onClick, onBoxStart, onBoxMove, onBoxEnd,
                      onDeletePoint, onDeleteBox }) {
  const wrapRef = useRef(null);
  const [rect, setRect] = useState({ w: 0, h: 0, x: 0, y: 0, scale: 1 });

  const layout = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap || !image) return;
    const cw = wrap.clientWidth, ch = wrap.clientHeight;
    const pad = 28;
    const s = Math.min((cw - pad*2) / image.w, (ch - pad*2) / image.h);
    const w = image.w * s, h = image.h * s;
    setRect({ w, h, x: (cw - w) / 2, y: (ch - h) / 2, scale: s });
  }, [image]);
  useEffect(() => { layout(); }, [layout]);
  useEffect(() => {
    const ro = new ResizeObserver(layout);
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [layout]);

  const toImg = (cx, cy) => ({
    x: Math.round((cx - rect.x) / rect.scale),
    y: Math.round((cy - rect.y) / rect.scale),
  });
  const inside = (cx, cy) => cx >= rect.x && cx <= rect.x + rect.w && cy >= rect.y && cy <= rect.y + rect.h;

  const onDown = e => {
    if (!image) return;
    const r = wrapRef.current.getBoundingClientRect();
    const cx = e.clientX - r.left, cy = e.clientY - r.top;
    if (!inside(cx, cy)) return;
    const p = toImg(cx, cy);
    if (e.button === 2) { onClick(p, false); return; }
    if (e.button === 0 && mode === "box") { onBoxStart(p); return; }
    if (e.button === 0) onClick(p, mode === "pos");
  };
  const onMove = e => {
    if (!image || !dragBox) return;
    const r = wrapRef.current.getBoundingClientRect();
    onBoxMove(toImg(e.clientX - r.left, e.clientY - r.top));
  };
  const onUp = e => {
    if (!image || !dragBox) return;
    const r = wrapRef.current.getBoundingClientRect();
    onBoxEnd(toImg(e.clientX - r.left, e.clientY - r.top));
  };

  return (
    <div ref={wrapRef}
      onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}
      onContextMenu={e => e.preventDefault()}
      style={{
        position: "relative", flex: 1, minWidth: 0, overflow: "hidden",
        cursor: !image ? "default" : mode === "box" ? "crosshair" : "crosshair",
        background: "var(--bg)",
      }}>
      {!image && (
        <div style={{
          position: "absolute", inset: 0, display: "grid", placeItems: "center",
          color: "var(--text-mute)", fontSize: 12, textAlign: "center",
        }}>
          <div>
            No image loaded.
            <div style={{ marginTop: 6, color: "var(--text-mute)" }}>Press <Kbd>E</Kbd> to load an example.</div>
          </div>
        </div>
      )}

      {image && (
        <div style={{
          position: "absolute", left: rect.x, top: rect.y, width: rect.w, height: rect.h,
          boxShadow: "0 0 0 1px var(--hair)",
        }}>
          {TWEAKS.checkerboard && <Checker />}
          <img src={`data:image/png;base64,${image.b64}`} alt="" draggable={false}
               style={{ position: "absolute", inset: 0, width: "100%", height: "100%", userSelect: "none" }}/>
          {mask && (
            <img src={`data:image/png;base64,${mask}`} alt=""
                 style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}/>
          )}
          <svg viewBox={`0 0 ${image.w} ${image.h}`} preserveAspectRatio="none"
               style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", overflow: "visible" }}>
            {box && <BoxShape b={box} scale={rect.scale} committed onDelete={onDeleteBox} />}
            {dragBox && <BoxShape b={dragBox} scale={rect.scale} />}
            {points.map((p, i) => (
              <Marker key={i} p={p} scale={rect.scale} onDelete={() => onDeletePoint(i)} />
            ))}
          </svg>
        </div>
      )}
    </div>
  );
}

function Checker() {
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'>
    <rect width='32' height='32' fill='#26292e'/>
    <rect width='16' height='16' fill='#202327'/>
    <rect x='16' y='16' width='16' height='16' fill='#202327'/>
  </svg>`;
  return <div style={{
    position: "absolute", inset: 0,
    backgroundImage: `url("data:image/svg+xml;utf8,${encodeURIComponent(svg)}")`,
    backgroundSize: "20px 20px",
  }}/>;
}

function Marker({ p, scale, onDelete }) {
  const color = p.positive ? "var(--pos)" : "var(--neg)";
  const r = 7 / scale;
  const sw = 1.4 / scale;
  const handler = onDelete && ((e) => { e.stopPropagation(); onDelete(); });
  return (
    <g style={onDelete ? { pointerEvents: "auto", cursor: "pointer" } : undefined}
       onPointerDown={handler}>
      {onDelete && <circle cx={p.x} cy={p.y} r={r * 1.4} fill="transparent" />}
      <circle cx={p.x} cy={p.y} r={r} fill="none" stroke={color} strokeWidth={sw} />
      <circle cx={p.x} cy={p.y} r={r*0.2} fill={color} />
      {p.positive ? (
        <g stroke={color} strokeWidth={sw} strokeLinecap="round">
          <line x1={p.x - r*0.5} y1={p.y} x2={p.x + r*0.5} y2={p.y} />
          <line x1={p.x} y1={p.y - r*0.5} x2={p.x} y2={p.y + r*0.5} />
        </g>
      ) : (
        <line x1={p.x - r*0.5} y1={p.y} x2={p.x + r*0.5} y2={p.y}
              stroke={color} strokeWidth={sw} strokeLinecap="round"/>
      )}
    </g>
  );
}

function BoxShape({ b, scale, committed, onDelete }) {
  const x = Math.min(b.x1, b.x2), y = Math.min(b.y1, b.y2);
  const w = Math.abs(b.x2 - b.x1), h = Math.abs(b.y2 - b.y1);
  const sw = 1.3 / scale;
  const hitSw = 10 / scale;
  const color = "var(--box)";
  const dash = committed ? "none" : `${5/scale} ${4/scale}`;
  const handler = onDelete && ((e) => { e.stopPropagation(); onDelete(); });
  return (
    <g style={onDelete ? { pointerEvents: "auto", cursor: "pointer" } : undefined}
       onPointerDown={handler}>
      {onDelete && <rect x={x} y={y} width={w} height={h}
        fill="none" stroke="transparent" strokeWidth={hitSw}/>}
      <rect x={x} y={y} width={w} height={h}
        fill="none" stroke={color} strokeWidth={sw} strokeDasharray={dash}/>
    </g>
  );
}

// ─── side panel ──────────────────────────────────────────────────────────────
function SidePanel({ mode, setMode, points, box, onReset, onSubmit, onLoad, connected }) {
  const pos = points.filter(p => p.positive).length;
  const neg = points.length - pos;
  const ready = points.length > 0 || !!box;

  return (
    <aside style={{
      width: 260, flex: "none",
      display: "flex", flexDirection: "column",
      borderLeft: "1px solid var(--hair)",
      background: "var(--bg-2)",
    }}>
      <div style={{ padding: "10px 10px", borderBottom: "1px solid var(--hair)" }}>
        <ModeTabs mode={mode} setMode={setMode} />
      </div>

      <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--hair)" }}>
        <Label>Prompts</Label>
        <div style={{ display: "flex", gap: 18 }}>
          <Count label="positive" value={pos} />
          <Count label="negative" value={neg} />
          <Count label="box" value={box ? 1 : 0} />
        </div>
      </div>

      <div style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: 6 }}>
        <Btn onClick={onSubmit} disabled={!ready || !connected} primary shortcut="⏎">
          Submit
        </Btn>
        <Btn onClick={onReset} disabled={!ready} shortcut="R">
          <IReset /> Reset
        </Btn>
        <Btn onClick={onLoad} disabled={!connected} shortcut="E">
          <IUpload /> Load example
        </Btn>
      </div>

      <div style={{ flex: 1 }}/>

      <div style={{ padding: "12px 16px 14px", borderTop: "1px solid var(--hair)" }}>
        <Label>Shortcuts</Label>
        <ShortRow k="L-click" label="add point (current mode)" />
        <ShortRow k="R-click" label="negative point" />
        <ShortRow k="click point" label="delete that point" />
        <ShortRow k="click box edge" label="delete box" />
        <ShortRow k="3 / drag" label="box" />
        <ShortRow k="R" label="reset" />
        <ShortRow k="E" label="load example" />
        <ShortRow k="⏎" label="submit" />
        <ShortRow k="⌫" label="undo" />
      </div>
    </aside>
  );
}

function Label({ children }) {
  return <div style={{
    fontSize: 10.5, letterSpacing: "0.1em", textTransform: "uppercase",
    color: "var(--text-mute)", marginBottom: 8,
  }}>{children}</div>;
}

function Count({ label, value }) {
  return (
    <div>
      <div className="mono" style={{ fontSize: 18, color: "var(--text)", lineHeight: 1, fontWeight: 400 }}>
        {String(value).padStart(2, "0")}
      </div>
      <div style={{ fontSize: 11, color: "var(--text-mute)", marginTop: 4 }}>{label}</div>
    </div>
  );
}

function Btn({ children, onClick, disabled, primary, shortcut }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      display: "flex", alignItems: "center", gap: 8,
      padding: "7px 10px",
      border: "1px solid var(--hair)",
      background: primary && !disabled ? "var(--panel-2)" : "transparent",
      color: disabled ? "var(--text-mute)" : "var(--text-dim)",
      borderRadius: 4, cursor: disabled ? "not-allowed" : "pointer",
      fontSize: 12.5,
    }}
    onMouseEnter={e => !disabled && (e.currentTarget.style.background = "var(--panel-2)")}
    onMouseLeave={e => !disabled && (e.currentTarget.style.background = primary ? "var(--panel-2)" : "transparent")}>
      {children}
      {shortcut && <span style={{ marginLeft: "auto" }}><Kbd>{shortcut}</Kbd></span>}
    </button>
  );
}

function ShortRow({ k, label }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "3px 0" }}>
      <span style={{ fontSize: 11.5, color: "var(--text-dim)" }}>{label}</span>
      <span className="mono" style={{ fontSize: 11, color: "var(--text-mute)" }}>{k}</span>
    </div>
  );
}

// ─── socket with mock fallback ───────────────────────────────────────────────
function useAnnotatorSocket(url, setPrompts) {
  const [status, setStatus] = useState("connecting");
  const [image, setImage] = useState(null);
  const [mask, setMask] = useState(null);
  const [label, setLabel] = useState("");
  const wsRef = useRef(null);
  const retryRef = useRef(0);
  const mockRef = useRef(null);
  const usingMockRef = useRef(false);

  const openMock = useCallback(() => {
    usingMockRef.current = true;
    setStatus("open");
    mockRef.current = makeMockBackend({
      onImage:   (b64, w, h) => {
        setImage({ b64, w, h });
        setPrompts([], null);
      },
      onPreview: (b64) => setMask(b64),
    });
  }, [setPrompts]);

  const connect = useCallback(() => {
    mockRef.current?.reset();
    setStatus("connecting");
    usingMockRef.current = false;
    let ws;
    try { ws = new WebSocket(url); } catch { setStatus("closed"); openMock(); return; }
    wsRef.current = ws;
    const t = setTimeout(() => { if (ws.readyState !== 1) { try { ws.close(); } catch {} openMock(); } }, 900);
    ws.onopen = () => { clearTimeout(t); retryRef.current = 0; setStatus("open"); };
    ws.onclose = () => {
      clearTimeout(t);
      setStatus("closed");
      if (!usingMockRef.current && retryRef.current < 3) {
        retryRef.current += 1; setStatus("connecting");
        setTimeout(connect, 600 * retryRef.current);
      } else if (retryRef.current >= 3) openMock();
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "image")   {
          setImage({ b64: msg.image_b64, w: msg.w, h: msg.h });
          setMask(null);
          setLabel(msg.label || "");
          setPrompts(msg.points || [], msg.box || null);
        }
        if (msg.type === "preview") setMask(msg.mask_b64);
      } catch {}
    };
  }, [url, openMock, setPrompts]);

  useEffect(() => { connect(); return () => { try { wsRef.current?.close(); } catch {} }; }, [connect]);

  const send = useCallback((msg) => {
    if (mockRef.current) return mockRef.current.send(msg);
    const ws = wsRef.current;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg));
  }, []);

  return { status, image, mask, label, send, reconnect: connect };
}

function makeMockBackend({ onImage, onPreview }) {
  let points = [], box = null, current = null;
  function loadExample() {
    const w = 900, h = 600;
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, "#3a4f6b"); g.addColorStop(0.55, "#6d8aa8"); g.addColorStop(0.56, "#8aa07b"); g.addColorStop(1, "#2f3b2a");
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "rgba(255,230,180,.9)"; ctx.beginPath(); ctx.arc(700, 180, 60, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = "#2a3445";
    ctx.beginPath(); ctx.moveTo(0,340); ctx.lineTo(220,200); ctx.lineTo(360,310); ctx.lineTo(540,180); ctx.lineTo(760,330); ctx.lineTo(900,230); ctx.lineTo(900,340); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#c8342a";
    ctx.beginPath();
    const rr = (x,y,w,h,r) => { ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); };
    rr(340,380,240,90,22); ctx.fill();
    ctx.fillStyle = "#1a1a1a"; ctx.beginPath(); ctx.arc(400,480,22,0,Math.PI*2); ctx.arc(530,480,22,0,Math.PI*2); ctx.fill();
    current = { w, h, subject: { x: 340, y: 380, w: 240, h: 90 } };
    onImage(c.toDataURL("image/png").split(",")[1], w, h);
    points = []; box = null; updatePreview();
  }
  function updatePreview() {
    if (!current) return;
    const { w, h, subject } = current;
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    if (points.length === 0 && !box) { onPreview(c.toDataURL("image/png").split(",")[1]); return; }
    let cx = subject.x + subject.w/2, cy = subject.y + subject.h/2;
    let rx = subject.w/2 + 20, ry = subject.h/2 + 24;
    if (box) {
      cx = (box.x1 + box.x2)/2; cy = (box.y1 + box.y2)/2;
      rx = Math.abs(box.x2 - box.x1)/2; ry = Math.abs(box.y2 - box.y1)/2;
    }
    ctx.save(); ctx.translate(cx, cy);
    ctx.beginPath(); ctx.ellipse(0,0,rx,ry,0,0,Math.PI*2);
    ctx.fillStyle = "rgba(110, 180, 210, 0.5)"; ctx.fill();
    ctx.restore();
    ctx.globalCompositeOperation = "destination-out";
    points.filter(p => !p.positive).forEach(p => {
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 55);
      g.addColorStop(0, "rgba(0,0,0,1)"); g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x,p.y,55,0,Math.PI*2); ctx.fill();
    });
    ctx.globalCompositeOperation = "source-over";
    points.filter(p => p.positive).forEach(p => {
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 50);
      g.addColorStop(0, "rgba(110,180,210,0.55)"); g.addColorStop(1, "rgba(110,180,210,0)");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x,p.y,50,0,Math.PI*2); ctx.fill();
    });
    onPreview(c.toDataURL("image/png").split(",")[1]);
  }
  return {
    send(msg) {
      if (msg.type === "set_state") {
        points = msg.points.map(p => ({ x: p.x, y: p.y, positive: p.positive }));
        box = msg.box ? { x1: msg.box.x1, y1: msg.box.y1, x2: msg.box.x2, y2: msg.box.y2 } : null;
        updatePreview();
      }
      else if (msg.type === "submit")       { /* noop */ }
      else if (msg.type === "load_example") { loadExample(); }
    },
    reset() { points = []; box = null; },
  };
}

// ─── tweaks ──────────────────────────────────────────────────────────────────
function TweaksPanel({ visible, values, onChange }) {
  if (!visible) return null;
  return (
    <div style={{
      position: "fixed", right: 12, bottom: 12, width: 240,
      background: "var(--bg-2)", border: "1px solid var(--hair)",
      borderRadius: 5, padding: "10px 12px",
      zIndex: 50,
    }}>
      <div style={{ fontSize: 10.5, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-mute)", marginBottom: 8 }}>Tweaks</div>
      <Row label="Checkerboard">
        <input type="checkbox" checked={values.checkerboard}
          onChange={e => onChange({ checkerboard: e.target.checked })}/>
      </Row>
      <Row label="WS URL">
        <input type="text" value={values.wsUrl}
          onChange={e => onChange({ wsUrl: e.target.value })}
          style={{
            width: 150, background: "var(--bg)", border: "1px solid var(--hair)",
            color: "var(--text-dim)", padding: "2px 6px", borderRadius: 3,
            fontSize: 11, fontFamily: "JetBrains Mono, monospace",
          }}/>
      </Row>
    </div>
  );
}
function Row({ label, children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, padding: "5px 0" }}>
      <div style={{ fontSize: 12, color: "var(--text-dim)" }}>{label}</div>
      {children}
    </div>
  );
}

// ─── app ─────────────────────────────────────────────────────────────────────
function App() {
  const [tweaks, setTweaks] = useState(TWEAKS);
  useEffect(() => { Object.assign(TWEAKS, tweaks); }, [tweaks]);

  const [mode, setMode] = useState("pos");
  const [points, setPoints] = useState([]);
  const [box, setBox] = useState(null);
  const [dragBox, setDragBox] = useState(null);

  const setPrompts = useCallback((seedPoints, seedBox) => {
    setPoints(seedPoints.map(p => ({ x: p.x, y: p.y, positive: p.positive })));
    setBox(seedBox ? { x1: seedBox[0], y1: seedBox[1], x2: seedBox[2], y2: seedBox[3] } : null);
    setDragBox(null);
  }, []);
  const { status, image, mask, label, send, reconnect } = useAnnotatorSocket(tweaks.wsUrl, setPrompts);
  const connected = status === "open";

  const addPoint = (p, positive) => setPoints(ps => [...ps, { x: p.x, y: p.y, positive }]);
  const deletePoint = (i) => setPoints(ps => ps.filter((_, j) => j !== i));
  const deleteBox = () => setBox(null);
  const beginBox = (p) => setDragBox({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
  const moveBox  = (p) => setDragBox(d => d && ({ ...d, x2: p.x, y2: p.y }));
  const endBox   = (p) => {
    if (!dragBox) return;
    const b = { x1: dragBox.x1, y1: dragBox.y1, x2: p.x, y2: p.y };
    if (Math.abs(b.x2 - b.x1) < 3 || Math.abs(b.y2 - b.y1) < 3) { setDragBox(null); return; }
    setBox(b); setDragBox(null);
  };
  const reset = () => { setPoints([]); setBox(null); setDragBox(null); };
  const submit = () => { if (points.length || box) send({ type: "submit" }); };
  const loadExample = () => send({ type: "load_example" });
  const undo = () => {
    if (dragBox) setDragBox(null);
    else if (points.length) setPoints(ps => ps.slice(0, -1));
    else if (box) setBox(null);
  };

  useEffect(() => {
    if (!image) return;
    send({ type: "set_state", points, box });
  }, [points, box, send]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.target && /input|textarea/i.test(e.target.tagName)) return;
      if (e.key === "1") setMode("pos");
      else if (e.key === "2") setMode("neg");
      else if (e.key === "3") setMode("box");
      else if (e.key === "r" || e.key === "R") reset();
      else if (e.key === "e" || e.key === "E") loadExample();
      else if (e.key === "Enter") submit();
      else if (e.key === "Backspace") undo();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const [tweaksOpen, setTweaksOpen] = useState(false);
  useEffect(() => {
    const onMsg = (e) => {
      if (e.data?.type === "__activate_edit_mode") setTweaksOpen(true);
      else if (e.data?.type === "__deactivate_edit_mode") setTweaksOpen(false);
    };
    window.addEventListener("message", onMsg);
    try { window.parent.postMessage({ type: "__edit_mode_available" }, "*"); } catch {}
    return () => window.removeEventListener("message", onMsg);
  }, []);
  const onTweak = (patch) => {
    setTweaks(v => ({ ...v, ...patch }));
    try { window.parent.postMessage({ type: "__edit_mode_set_keys", edits: patch }, "*"); } catch {}
  };

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <TopBar status={status} wsUrl={tweaks.wsUrl} onReconnect={reconnect} imgMeta={image} caption={label}/>
      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        <CanvasArea mode={mode} image={image} mask={mask}
          points={points} box={box} dragBox={dragBox}
          onClick={addPoint} onBoxStart={beginBox} onBoxMove={moveBox} onBoxEnd={endBox}
          onDeletePoint={deletePoint} onDeleteBox={deleteBox}/>
        <SidePanel mode={mode} setMode={setMode}
          points={points} box={box}
          onReset={reset} onSubmit={submit} onLoad={loadExample}
          connected={connected}/>
      </div>
      <TweaksPanel visible={tweaksOpen} values={tweaks} onChange={onTweak}/>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
