import { useEffect, useMemo, useRef, useState } from "react";

type Coord = { r: number; c: number };
type Mode = "start" | "end" | "block" | "required" | "erase";
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

/* -------------------- A* over expanded state space --------------------
   State = (r, c, mask, visitedCells?) where:
   - (r,c): position
   - mask: bitmask of required waypoints visited so far
   - visitedCells: only when no-revisit is enforced (tracks cells on the path)
   Heuristic h(n) = 0 (requested) -> optimal but slower = Dijkstra on this state space
------------------------------------------------------------------------ */
type AStarNode = {
  r: number;
  c: number;
  mask: number;
  g: number;
  f: number;
  parent?: AStarNode;
  visitedCells?: Set<CellKey>; // only used when allowRevisit=false
};

function astarMulti(
  n: number,
  blocked: Set<CellKey>,
  start: Coord,
  required: Coord[],
  end: Coord,
  allowRevisit: boolean
): Coord[] | null {
  const reqIndexByKey = new Map<CellKey, number>();
  required.forEach((p, i) => reqIndexByKey.set(key(p.r, p.c), i));
  const FULL = (1 << required.length) - 1;

  const startMask =
    reqIndexByKey.has(key(start.r, start.c)) ? (1 << (reqIndexByKey.get(key(start.r, start.c))!)) : 0;

  // If start is already at end and all required done
  if (start.r === end.r && start.c === end.c && startMask === FULL) return [start];

  const startNode: AStarNode = {
    r: start.r,
    c: start.c,
    mask: startMask,
    g: 0,
    f: 0, // h = 0
    parent: undefined,
    visitedCells: allowRevisit ? undefined : new Set<CellKey>([key(start.r, start.c)]),
  };

  // open set (min by f); simple array is fine for our grids
  const open: AStarNode[] = [startNode];

  // best g seen for (r,c,mask) -> prune dominated states
  const bestG = new Map<string, number>();
  const stateKey = (r: number, c: number, mask: number) => `${r},${c}|${mask}`;
  bestG.set(stateKey(start.r, start.c, startMask), 0);

  while (open.length) {
    // pop node with smallest f (linear scan; OK for demo sizes)
    let bestIdx = 0;
    for (let i = 1; i < open.length; i++) if (open[i].f < open[bestIdx].f) bestIdx = i;
    const cur = open.splice(bestIdx, 1)[0];

    // goal check: at end with all required visited
    if (cur.r === end.r && cur.c === end.c && cur.mask === FULL) {
      const out: Coord[] = [];
      let p: AStarNode | undefined = cur;
      while (p) {
        out.push({ r: p.r, c: p.c });
        p = p.parent;
      }
      return out.reverse();
    }

    // expand neighbors
    for (const nb of neighbors(n, { r: cur.r, c: cur.c })) {
      const nbKey = key(nb.r, nb.c);
      if (blocked.has(nbKey)) continue;

      // don't step on END until all required are done, to avoid early finishing or blocking
      if (nb.r === end.r && nb.c === end.c && cur.mask !== FULL) continue;

      // no-revisit constraint: path must be simple -> skip if already on our current path
      if (!allowRevisit && cur.visitedCells!.has(nbKey)) continue;

      // update mask when we visit a required waypoint
      let nextMask = cur.mask;
      const reqIdx = reqIndexByKey.get(nbKey);
      if (reqIdx !== undefined) nextMask = nextMask | (1 << reqIdx);

      const nextG = cur.g + 1;
      const sk = stateKey(nb.r, nb.c, nextMask);
      const prevBest = bestG.get(sk);
      if (prevBest !== undefined && nextG >= prevBest) continue;

      // heuristic h = 0 (as requested)
      const nextH = 0;
      const nextNode: AStarNode = {
        r: nb.r,
        c: nb.c,
        mask: nextMask,
        g: nextG,
        f: nextG + nextH,
        parent: cur,
        visitedCells: allowRevisit
          ? undefined
          : (() => {
              const s = new Set(cur.visitedCells);
              s.add(nbKey);
              return s;
            })(),
      };

      bestG.set(sk, nextG);
      open.push(nextNode);
    }
  }

  return null; // no route
}

/* -------------------- Component -------------------- */
export default function PathfindingPlayground() {
  const [n, setN] = useState<number>(20);
  const [mode, setMode] = useState<Mode>("block");
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
  const pathSet = useMemo(() => new Set(visiblePath.map((p) => key(p.r, p.c))), [visiblePath]);

  // build arrow rotation (deg) for each "from" step in the visible path
  const arrowDeg = useMemo(() => {
    const m = new Map<CellKey, number>();
    for (let i = 0; i < visiblePath.length - 1; i++) {
      const a = visiblePath[i];
      const b = visiblePath[i + 1];
      const dr = b.r - a.r;
      const dc = b.c - a.c;
      if (dr === 0 && dc === 1) m.set(key(a.r, a.c), 0);       // →
      else if (dr === 1 && dc === 0) m.set(key(a.r, a.c), 90); // ↓
      else if (dr === 0 && dc === -1) m.set(key(a.r, a.c), 180); // ←
      else if (dr === -1 && dc === 0) m.set(key(a.r, a.c), -90); // ↑
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

  const run = () => {
    clearPath();
    if (!start || !end) {
      alert("Please set both Start and End.");
      return;
    }
    const p = astarMulti(n, blocked, start, required, end, allowRevisit);
    if (!p) {
      alert("No route under current constraints.");
      return;
    }
    setPath(p);
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
              const deg = arrowDeg.get(cell.k);
              return (
                <div
                  key={cell.k}
                  onMouseDown={() => onMouseDown(cell.r, cell.c)}
                  onMouseEnter={() => onMouseEnter(cell.r, cell.c)}
                  className={
                    "relative aspect-square cursor-crosshair rounded-sm border " +
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
                  {deg !== undefined && (
                    <svg
                      viewBox="0 0 100 100"
                      className="pointer-events-none absolute inset-0 z-10"
                      style={{ transform: `rotate(${deg}deg)`, transformOrigin: "50% 50%" }}
                      aria-hidden
                    >
                      <line x1="20" y1="50" x2="72" y2="50" stroke="black" strokeWidth="10" strokeLinecap="round" />
                      <polygon points="70,35 95,50 70,65" fill="black" />
                    </svg>
                  )}
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
            <li>Press <span className="font-medium">Run</span> to compute the optimal path (A* with h=0).</li>
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
            <li>A* runs on (row, col, required-mask) with h(n)=0 (optimal).</li>
            <li>“Allow revisiting cells” off enforces a simple path (no cell used twice).</li>
            <li>Arrows show the final optimal path direction.</li>
          </ul>
        </aside>
      </section>

      <footer className="text-xs text-neutral-500 pt-1">Built for CV demos • Try adding Manhattan heuristic or MST lower bound for speed.</footer>
    </div>
  );
}
