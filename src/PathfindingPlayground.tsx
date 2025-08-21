import { useEffect, useMemo, useRef, useState } from "react";

type Coord = { r: number; c: number };
type Mode = "start" | "end" | "block" | "required" | "erase";
type VisitOrder = "as-clicked" | "optimize";
type CellKey = string;

const key = (r: number, c: number): CellKey => `${r},${c}`;
const fromKey = (k: CellKey): Coord => {
  const [r, c] = k.split(",").map(Number);
  return { r, c };
};

function neighbors(n: number, { r, c }: Coord): Coord[] {
  const dirs = [
    { r: -1, c: 0 },
    { r: 1, c: 0 },
    { r: 0, c: -1 },
    { r: 0, c: 1 },
  ];
  const out: Coord[] = [];
  for (const d of dirs) {
    const nr = r + d.r;
    const nc = c + d.c;
    if (nr >= 0 && nr < n && nc >= 0 && nc < n) out.push({ r: nr, c: nc });
  }
  return out;
}

function reconstructPath(parent: Map<CellKey, CellKey>, start: Coord, goal: Coord): Coord[] {
  const path: Coord[] = [];
  let cur: CellKey | undefined = key(goal.r, goal.c);
  const sk = key(start.r, start.c);
  while (cur && cur !== sk) {
    path.push(fromKey(cur));
    cur = parent.get(cur);
  }
  if (cur === sk) path.push(start);
  return path.reverse();
}

function bfs(
  n: number,
  blocked: Set<CellKey>,
  start: Coord,
  goal: Coord,
  forbidden?: Set<CellKey>
): { path: Coord[]; visited: CellKey[] } | { path: null; visited: CellKey[] } {
  if (key(start.r, start.c) === key(goal.r, goal.c)) return { path: [start], visited: [] };
  const q: Coord[] = [start];
  const visited = new Set<CellKey>([key(start.r, start.c)]);
  const order: CellKey[] = [key(start.r, start.c)];
  const parent = new Map<CellKey, CellKey>();
  const goalKey = key(goal.r, goal.c);

  while (q.length) {
    const cur = q.shift()!;
    for (const nb of neighbors(n, cur)) {
      const nbKey = key(nb.r, nb.c);
      if (blocked.has(nbKey)) continue;
      if (forbidden && forbidden.has(nbKey)) continue;
      if (visited.has(nbKey)) continue;
      visited.add(nbKey);
      order.push(nbKey);
      parent.set(nbKey, key(cur.r, cur.c));
      if (nbKey === goalKey) {
        return { path: reconstructPath(parent, start, goal), visited: order };
      }
      q.push(nb);
    }
  }
  return { path: null, visited: order };
}

function optimizeOrder(
  n: number,
  blocked: Set<CellKey>,
  start: Coord,
  required: Coord[],
  end: Coord
): { order: Coord[]; pairPaths: Map<string, Coord[]> } | { order: null; reason: string } {
  const allPoints = [start, ...required, end];
  const K = allPoints.length;
  const dist: number[][] = Array.from({ length: K }, () => Array(K).fill(Infinity));
  const pairPaths = new Map<string, Coord[]>();

  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      if (i === j) {
        dist[i][j] = 0;
        pairPaths.set(`${i}-${j}`, [allPoints[i]]);
        continue;
      }
      const res = bfs(n, blocked, allPoints[i], allPoints[j]);
      if (res.path) {
        dist[i][j] = res.path.length - 1;
        pairPaths.set(`${i}-${j}`, res.path);
      } else {
        dist[i][j] = Infinity;
      }
    }
  }

  for (let j = 1; j < K; j++) if (!isFinite(dist[0][j])) return { order: null, reason: "Unreachable waypoint/end." };

  const R = required.length;
  if (R === 0) return { order: [start, end], pairPaths };

  const dp = new Map<string, { cost: number; prev: number | null }>();

  for (let i = 1; i <= R; i++) {
    const m = 1 << (i - 1);
    dp.set(`${m},${i}`, { cost: dist[0][i], prev: 0 });
  }

  for (let mask = 1; mask < 1 << R; mask++) {
    for (let i = 1; i <= R; i++) {
      if (!(mask & (1 << (i - 1)))) continue;
      const keyDP = `${mask},${i}`;
      const state = dp.get(keyDP);
      if (!state) continue;
      for (let j = 1; j <= R; j++) {
        if (mask & (1 << (j - 1))) continue;
        const nextMask = mask | (1 << (j - 1));
        const newCost = state.cost + dist[i][j];
        const k2 = `${nextMask},${j}`;
        const cur = dp.get(k2);
        if (isFinite(newCost) && (!cur || newCost < cur.cost)) {
          dp.set(k2, { cost: newCost, prev: i });
        }
      }
    }
  }

  const full = (1 << R) - 1;
  let bestCost = Infinity;
  let bestLast = -1;
  for (let i = 1; i <= R; i++) {
    const st = dp.get(`${full},${i}`);
    if (!st) continue;
    const total = st.cost + dist[i][K - 1];
    if (total < bestCost) {
      bestCost = total;
      bestLast = i;
    }
  }
  if (!isFinite(bestCost) || bestLast === -1) return { order: null, reason: "No complete route to end via all waypoints." };

  const seq: number[] = [K - 1, bestLast];
  let mask = full;
  let cur = bestLast;
  while (mask) {
    const st = dp.get(`${mask},${cur}`)!;
    if (st.prev === 0) {
      seq.push(0);
      break;
    }
    seq.push(st.prev!);
    mask = mask & ~(1 << (cur - 1));
    cur = st.prev!;
  }
  seq.reverse();
  const order: Coord[] = seq.map((idx) => allPoints[idx]);
  return { order, pairPaths };
}

function concatPaths(pairPaths: Map<string, Coord[]>, order: Coord[]): Coord[] {
  const path: Coord[] = [];
  for (let i = 0; i < order.length - 1; i++) {
    const a = order[i];
    const b = order[i + 1];
    let seg: Coord[] | undefined = undefined;
    for (const [, v] of pairPaths.entries()) {
      const ia = v.length ? v[0] : null;
      const ib = v.length ? v[v.length - 1] : null;
      if (ia && ib && ia.r === a.r && ia.c === a.c && ib.r === b.r && ib.c === b.c) {
        seg = v;
        break;
      }
    }
    if (!seg) throw new Error("Missing segment in pairPaths");
    if (i === 0) path.push(...seg);
    else path.push(...seg.slice(1));
  }
  return path;
}

export default function PathfindingPlayground() {
  const [n, setN] = useState<number>(20);
  const [mode, setMode] = useState<Mode>("block");
  const [visitOrder, setVisitOrder] = useState<VisitOrder>("as-clicked");
  const [animate, setAnimate] = useState<boolean>(true);
  const [allowRevisit, setAllowRevisit] = useState<boolean>(true);
  const [start, setStart] = useState<Coord | null>(null);
  const [end, setEnd] = useState<Coord | null>(null);
  const [blocked, setBlocked] = useState<Set<CellKey>>(new Set());
  const [required, setRequired] = useState<Coord[]>([]);
  const [isMouseDown, setIsMouseDown] = useState(false);
  const [path, setPath] = useState<Coord[]>([]);
  const [animIndex, setAnimIndex] = useState<number>(0);
  const animRef = useRef<number | null>(null);

  useEffect(() => {
    if (!animate || path.length === 0) return;
    setAnimIndex(0);
    if (animRef.current) cancelAnimationFrame(animRef.current);
    let last = performance.now();
    const step = (t: number) => {
      if (t - last > 18) {
        setAnimIndex((i) => (i + 1 < path.length ? i + 1 : path.length));
        last = t;
      }
      if (animIndex + 1 < path.length) animRef.current = requestAnimationFrame(step);
    };
    animRef.current = requestAnimationFrame(step);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      animRef.current = null;
    };
  }, [path, animate]);

  const visiblePath = useMemo(
    () => path.slice(0, animate ? animIndex : path.length),
    [path, animIndex, animate]
  );

  const pathSet = useMemo(
    () => new Set(visiblePath.map((p) => key(p.r, p.c))),
    [visiblePath]
  );

  const arrowMap = useMemo(() => {
    const m = new Map<CellKey, string>();
    for (let i = 0; i < visiblePath.length - 1; i++) {
      const a = visiblePath[i];
      const b = visiblePath[i + 1];
      const dr = b.r - a.r;
      const dc = b.c - a.c;
      let arrow = "";
      if (dr === -1 && dc === 0) arrow = "↑";
      else if (dr === 1 && dc === 0) arrow = "↓";
      else if (dr === 0 && dc === -1) arrow = "←";
      else if (dr === 0 && dc === 1) arrow = "→";
      if (arrow) m.set(key(a.r, a.c), arrow);
    }
    return m;
  }, [visiblePath]);

  const startKey = start ? key(start.r, start.c) : null;
  const endKey = end ? key(end.r, end.c) : null;
  const requiredKeys = useMemo(() => new Set(required.map((r) => key(r.r, r.c))), [required]);

  const grid = useMemo(() => {
    return Array.from({ length: n }, (_, r) =>
      Array.from({ length: n }, (_, c) => {
        const k = key(r, c);
        let t: "empty" | "blocked" | "start" | "end" | "required" | "path" = "empty";
        if (blocked.has(k)) t = "blocked";
        if (startKey === k) t = "start";
        if (endKey === k) t = "end";
        if (requiredKeys.has(k)) t = "required";
        if (pathSet.has(k) && !(requiredKeys.has(k) || startKey === k || endKey === k)) t = "path";
        return { r, c, k, t };
      })
    );
  }, [n, blocked, startKey, endKey, requiredKeys, pathSet]);

  const applyCell = (r: number, c: number) => {
    const k = key(r, c);
    if (mode === "erase") {
      setBlocked((s) => {
        const ns = new Set(s);
        ns.delete(k);
        return ns;
      });
      setRequired((arr) => arr.filter((p) => !(p.r === r && p.c === c)));
      if (startKey === k) setStart(null);
      if (endKey === k) setEnd(null);
      setPath([]);
      return;
    }
    if (mode === "block") {
      if (startKey === k || endKey === k || requiredKeys.has(k)) return;
      setBlocked((s) => new Set(s).add(k));
      setPath([]);
      return;
    }
    if (mode === "required") {
      if (blocked.has(k) || startKey === k || endKey === k) return;
      setRequired((arr) => (arr.some((p) => p.r === r && p.c === c) ? arr : [...arr, { r, c }]));
      setPath([]);
      return;
    }
    if (mode === "start") {
      if (blocked.has(k) || requiredKeys.has(k) || endKey === k) return;
      setStart({ r, c });
      setPath([]);
      return;
    }
    if (mode === "end") {
      if (blocked.has(k) || requiredKeys.has(k) || startKey === k) return;
      setEnd({ r, c });
      setPath([]);
      return;
    }
  };

  const onMouseDown = (r: number, c: number) => {
    setIsMouseDown(true);
    applyCell(r, c);
  };
  const onMouseEnter = (r: number, c: number) => {
    if (isMouseDown && (mode === "block" || mode === "erase")) applyCell(r, c);
  };
  const onMouseUp = () => setIsMouseDown(false);

  const clearPath = () => {
    setPath([]);
    setAnimIndex(0);
  };

  const resetGrid = () => {
    setStart(null);
    setEnd(null);
    setBlocked(new Set());
    setRequired([]);
    setPath([]);
    setAnimIndex(0);
  };

  const buildForbid = (visitedGlobal: Set<CellKey>, sequence: Coord[], i: number) => {
    const s = sequence[i], g = sequence[i + 1];
    const forbid = new Set<CellKey>(visitedGlobal);
    for (let j = i + 1; j < sequence.length; j++) {
      forbid.add(key(sequence[j].r, sequence[j].c));
    }
    forbid.delete(key(s.r, s.c));
    forbid.delete(key(g.r, g.c));
    return forbid;
  };

  const run = () => {
    clearPath();
    if (!start || !end) {
      alert("Please set both Start and End.");
      return;
    }

    if (visitOrder === "as-clicked") {
      const sequence: Coord[] = [start, ...required, end];
      let accum: Coord[] = [];
      const visitedGlobal = new Set<CellKey>([key(start.r, start.c)]);

      for (let i = 0; i < sequence.length - 1; i++) {
        const s = sequence[i], g = sequence[i + 1];
        const forbid = allowRevisit ? undefined : buildForbid(visitedGlobal, sequence, i);

        const res = bfs(n, blocked, s, g, forbid);
        if (!res.path) {
          alert(`No path for segment ${i + 1} under current constraints.`);
          return;
        }
        if (i === 0) accum = res.path;
        else accum = [...accum, ...res.path.slice(1)];

        if (!allowRevisit) for (const p of res.path) visitedGlobal.add(key(p.r, p.c));
      }
      setPath(accum);
      return;
    }

    const opt = optimizeOrder(n, blocked, start, required, end);
    if (!("order" in opt) || !opt.order) {
      alert(opt.reason || "Optimization failed.");
      return;
    }

    if (allowRevisit) {
      try {
        const p = concatPaths(opt.pairPaths, opt.order);
        setPath(p);
      } catch {
        let accum: Coord[] = [];
        for (let i = 0; i < opt.order.length - 1; i++) {
          const res = bfs(n, blocked, opt.order[i], opt.order[i + 1]);
          if (!res.path) {
            alert("A segment in optimized order is unreachable.");
            return;
          }
          if (i === 0) accum = res.path;
          else accum = [...accum, ...res.path.slice(1)];
        }
        setPath(accum);
      }
    } else {
      let accum: Coord[] = [];
      const visitedGlobal = new Set<CellKey>([key(start.r, start.c)]);

      for (let i = 0; i < opt.order.length - 1; i++) {
        const s = opt.order[i], g = opt.order[i + 1];
        const forbid = buildForbid(visitedGlobal, opt.order, i);

        const res = bfs(n, blocked, s, g, forbid);
        if (!res.path) {
          alert("No path under no-revisit constraint for the optimized order.");
          return;
        }
        if (i === 0) accum = res.path;
        else accum = [...accum, ...res.path.slice(1)];
        for (const p of res.path) visitedGlobal.add(key(p.r, p.c));
      }
      setPath(accum);
    }
  };

  const exportState = () => {
    const state = { n, start, end, blocked: Array.from(blocked), required };
    navigator.clipboard.writeText(JSON.stringify(state)).then(() => {
      alert("State copied to clipboard as JSON.");
    });
  };
  const importState = () => {
    const raw = prompt("Paste JSON exported state:");
    if (!raw) return;
    try {
      const st = JSON.parse(raw);
      if (typeof st.n === "number") setN(st.n);
      setStart(st.start ?? null);
      setEnd(st.end ?? null);
      setBlocked(new Set(st.blocked ?? []));
      setRequired((st.required ?? []).map((p: any) => ({ r: p.r, c: p.c })));
      clearPath();
    } catch {
      alert("Invalid JSON.");
    }
  };

  return (
    <div className="min-h-screen w-full bg-neutral-50 text-neutral-900 p-4 flex flex-col gap-4">
      <header className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Pathfinding Playground</h1>
        <div className="ml-auto flex flex-wrap gap-2">
          <button className={`px-3 py-1 rounded-2xl border ${mode === "start" ? "bg-black text-white" : "bg-white"}`} onClick={() => setMode("start")}>Start</button>
          <button className={`px-3 py-1 rounded-2xl border ${mode === "end" ? "bg-black text-white" : "bg-white"}`} onClick={() => setMode("end")}>End</button>
          <button className={`px-3 py-1 rounded-2xl border ${mode === "block" ? "bg-black text-white" : "bg-white"}`} onClick={() => setMode("block")}>Block</button>
          <button className={`px-3 py-1 rounded-2xl border ${mode === "required" ? "bg-black text-white" : "bg-white"}`} onClick={() => setMode("required")}>Required</button>
          <button className={`px-3 py-1 rounded-2xl border ${mode === "erase" ? "bg-black text-white" : "bg-white"}`} onClick={() => setMode("erase")}>Erase</button>
        </div>
      </header>

      <section className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-sm">Grid</label>
          <input type="range" min={8} max={48} value={n} onChange={(e) => setN(parseInt(e.target.value))} />
          <span className="text-sm w-10 text-center">{n}×{n}</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm">Order</label>
          <select className="px-2 py-1 rounded-xl border bg-white" value={visitOrder} onChange={(e) => setVisitOrder(e.target.value as VisitOrder)}>
            <option value="as-clicked">As-clicked (sequence)</option>
            <option value="optimize">Optimize waypoints</option>
          </select>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={animate} onChange={(e) => setAnimate(e.target.checked)} />
          Animate path
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={allowRevisit} onChange={(e) => setAllowRevisit(e.target.checked)} />
          Allow revisiting cells
        </label>
        <div className="ml-auto flex gap-2">
          <button className="px-3 py-1 rounded-2xl border bg-white" onClick={exportState}>Export</button>
          <button className="px-3 py-1 rounded-2xl border bg-white" onClick={importState}>Import</button>
          <button className="px-3 py-1 rounded-2xl border bg-white" onClick={clearPath}>Clear Path</button>
          <button className="px-3 py-1 rounded-2xl border bg-white" onClick={resetGrid}>Reset Grid</button>
          <button className="px-3 py-1 rounded-2xl border bg-black text-white" onClick={run}>Run ▶</button>
        </div>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-[auto_260px] gap-4">
        <div className="bg-white rounded-2xl p-3 shadow-sm select-none" onMouseLeave={onMouseUp} onMouseUp={onMouseUp}>
          <div className="grid" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))`, gap: 1 }}>
            {grid.flat().map((cell) => {
              const arrow = arrowMap.get(cell.k);
              return (
                <div
                  key={cell.k}
                  onMouseDown={() => onMouseDown(cell.r, cell.c)}
                  onMouseEnter={() => onMouseEnter(cell.r, cell.c)}
                  className={
                    "aspect-square cursor-crosshair rounded-sm border flex items-center justify-center " +
                    (cell.t === "blocked"
                      ? "bg-neutral-800 border-neutral-800"
                      : cell.t === "start"
                      ? "bg-green-500 border-green-600"
                      : cell.t === "end"
                      ? "bg-red-500 border-red-600"
                      : cell.t === "required"
                      ? "bg-amber-400 border-amber-500"
                      : cell.t === "path"
                      ? "bg-sky-400 border-sky-500"
                      : "bg-white border-neutral-200")
                  }
                  title={`${cell.r},${cell.c}`}
                >
                  {arrow && <span className="text-black font-bold select-none text-[12px] sm:text-sm leading-none">{arrow}</span>}
                </div>
              );
            })}
          </div>
        </div>

        <aside className="bg-white rounded-2xl p-3 shadow-sm text-sm leading-6">
          <h2 className="font-semibold mb-2">How to use</h2>
          <ol className="list-decimal list-inside space-y-1">
            <li>Select <span className="font-medium">Start</span> and click a cell.</li>
            <li>Select <span className="font-medium">End</span> and click a cell.</li>
            <li>Draw <span className="font-medium">Block</span> walls (drag to paint).</li>
            <li>Mark <span className="font-medium">Required</span> waypoints (must-visit).</li>
            <li>Choose visit order and press <span className="font-medium">Run</span>.</li>
          </ol>
          <h3 className="font-semibold mt-3 mb-1">Legend</h3>
          <div className="flex flex-wrap gap-2 items-center">
            <span className="inline-flex items-center gap-2"><span className="w-4 h-4 bg-green-500 inline-block rounded-sm border" />Start</span>
            <span className="inline-flex items-center gap-2"><span className="w-4 h-4 bg-red-500 inline-block rounded-sm border" />End</span>
            <span className="inline-flex items-center gap-2"><span className="w-4 h-4 bg-amber-400 inline-block rounded-sm border" />Required</span>
            <span className="inline-flex items-center gap-2"><span className="w-4 h-4 bg-neutral-800 inline-block rounded-sm border" />Blocked</span>
            <span className="inline-flex items-center gap-2"><span className="w-4 h-4 bg-sky-400 inline-block rounded-sm border" />Path</span>
          </div>
          <h3 className="font-semibold mt-3 mb-1">Notes</h3>
          <ul className="list-disc list-inside space-y-1">
            <li>Pathfinding uses BFS with unit weights.</li>
            <li>Optimize waypoints uses Held–Karp DP for exact order on small sets.</li>
            <li>Export copies a JSON snapshot for reproducible demos.</li>
          </ul>
        </aside>
      </section>

      <footer className="text-xs text-neutral-500 pt-1">Built for CV demos • Extend with weighted cells, diagonal moves, or RL training.</footer>
    </div>
  );
}

