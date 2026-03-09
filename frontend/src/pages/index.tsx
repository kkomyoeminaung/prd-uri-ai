import { useState, useRef, useEffect } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const PACCAYA_COLORS: Record<string, string> = {
  hetu: "#ef4444", nissaya: "#3b82f6", indriya: "#8b5cf6",
  avigata: "#10b981", anantara: "#f59e0b", sahajata: "#06b6d4",
  annamanna: "#ec4899", vigata: "#6b7280",
};
function getColor(n: string) {
  for (const [k, v] of Object.entries(PACCAYA_COLORS))
    if (n.startsWith(k)) return v;
  return "#94a3b8";
}

interface Msg {
  role: "user" | "assistant";
  content: string;
  causal?: { dominant_paccaya:[string,number][]; upanissaya_score:number; asevana_score:number; alpha_correction:number };
}

export default function Home() {
  const [msgs, setMsgs]           = useState<Msg[]>([]);
  const [input, setInput]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [sessionId, setSessionId] = useState<string|null>(null);
  const [panel, setPanel]         = useState(false);  // hidden by default on mobile
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:"smooth" }); }, [msgs]);

  async function send() {
    if (!input.trim() || loading) return;
    const txt = input.trim();
    setInput("");
    setMsgs(p => [...p, { role:"user", content:txt }]);
    setLoading(true);

    // Optimistic assistant placeholder
    setMsgs(p => [...p, { role:"assistant", content:"" }]);

    try {
      // Use streaming endpoint
      const res = await fetch(`${API}/api/chat/stream`, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ message:txt, session_id:sessionId, stream:true }),
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let causal: Msg["causal"] | null = null;
      let full = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));
          if (data.type === "meta") {
            if (!sessionId) setSessionId(data.session_id);
            causal = {
              dominant_paccaya: data.dominant_paccaya,
              upanissaya_score: data.upanissaya_score,
              asevana_score:    data.asevana_score,
              alpha_correction: data.alpha_correction,
            };
          } else if (data.type === "token") {
            full += data.text;
            setMsgs(p => {
              const n = [...p];
              n[n.length-1] = { role:"assistant", content:full, causal: causal||undefined };
              return n;
            });
          }
        }
      }
    } catch {
      // Fallback to non-streaming
      try {
        const res  = await fetch(`${API}/api/chat/`, {
          method:"POST", headers:{"Content-Type":"application/json"},
          body: JSON.stringify({ message:txt, session_id:sessionId }),
        });
        const data = await res.json();
        if (!sessionId) setSessionId(data.session_id);
        setMsgs(p => {
          const n = [...p];
          n[n.length-1] = {
            role:"assistant", content:data.response,
            causal:{ dominant_paccaya:data.dominant_paccaya, upanissaya_score:data.upanissaya_score,
                     asevana_score:data.asevana_score, alpha_correction:data.alpha_correction },
          };
          return n;
        });
      } catch {
        setMsgs(p => { const n=[...p]; n[n.length-1]={role:"assistant",content:"⚠️ Cannot reach backend."}; return n; });
      }
    } finally { setLoading(false); }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-100" style={{fontFamily:"system-ui,sans-serif"}}>

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-900 border-b border-gray-800 shrink-0">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-cyan-400 text-lg">⚛</span>
            <span className="font-bold text-white">PRD-URI AI</span>
            <span className="text-xs text-gray-500 hidden sm:inline">v3 · SU(5) Causal AGI</span>
          </div>
          <p className="text-xs text-gray-600 mt-0.5">α=1.274 · 24 Paccaya · Claude API</p>
        </div>
        <button onClick={()=>setPanel(p=>!p)}
          className="text-xs px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-cyan-400 transition">
          {panel ? "⊠ Hide" : "⊞ Causal"}
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden">

        {/* Chat area */}
        <div className="flex flex-col flex-1 overflow-hidden">
          <div className="flex-1 overflow-y-auto px-3 py-4 space-y-3">
            {msgs.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center pb-20">
                <div className="text-5xl mb-4">⚛</div>
                <h2 className="text-gray-300 font-semibold text-lg mb-1">PRD-URI AI</h2>
                <p className="text-gray-500 text-sm max-w-xs">
                  Causal intelligence based on Pattana-Relational Dynamics and SU(5) algebra.
                  Ask anything.
                </p>
                <div className="mt-6 grid grid-cols-1 gap-2 max-w-xs w-full">
                  {["I feel stuck in old patterns","Explain black hole thermodynamics","What is Hetu in PRD theory?"].map(q=>(
                    <button key={q} onClick={()=>{setInput(q);}}
                      className="text-left text-xs px-3 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 transition">
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {msgs.map((m,i)=>(
              <div key={i} className={`flex ${m.role==="user"?"justify-end":"justify-start"}`}>
                <div className={`max-w-xl rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap leading-relaxed ${
                  m.role==="user"
                    ? "bg-cyan-700 text-white rounded-tr-sm"
                    : "bg-gray-800 text-gray-100 border border-gray-700 rounded-tl-sm"
                }`}>
                  {m.content || <span className="animate-pulse text-gray-500">▍</span>}

                  {/* Causal analysis panel under assistant msg */}
                  {m.causal && panel && m.role==="assistant" && (
                    <div className="mt-3 pt-2 border-t border-gray-700 space-y-2">
                      <div className="flex flex-wrap gap-1">
                        {m.causal.dominant_paccaya.map(([n,s])=>(
                          <span key={n} className="text-xs px-2 py-0.5 rounded-full text-white font-medium"
                            style={{backgroundColor:getColor(n)}}>
                            {n} {(s*100).toFixed(1)}%
                          </span>
                        ))}
                      </div>
                      <div className="flex flex-wrap gap-3 text-xs text-gray-500">
                        <span>Upanissaya <b className="text-red-400">{(m.causal.upanissaya_score*100).toFixed(0)}%</b></span>
                        <span>Asevana <b className="text-green-400">{(m.causal.asevana_score*100).toFixed(0)}%</b></span>
                        <span>α·corr <b className="text-cyan-400">{m.causal.alpha_correction.toFixed(4)}</b></span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef}/>
          </div>

          {/* Input bar */}
          <div className="shrink-0 px-3 py-3 bg-gray-900 border-t border-gray-800">
            <div className="flex gap-2 items-end">
              <textarea
                value={input}
                onChange={e=>setInput(e.target.value)}
                onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}}}
                placeholder="Ask anything…"
                rows={1}
                className="flex-1 bg-gray-800 text-gray-100 rounded-xl px-4 py-2.5 text-sm outline-none border border-gray-700 focus:border-cyan-500 resize-none transition"
                style={{minHeight:"42px",maxHeight:"120px"}}
              />
              <button onClick={send} disabled={loading||!input.trim()}
                className="px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 rounded-xl text-sm font-semibold transition shrink-0">
                {loading ? "…" : "Send"}
              </button>
            </div>
            <p className="text-xs text-gray-700 mt-1 text-center">PRD-URI AI · Myo Min Aung 2026</p>
          </div>
        </div>

        {/* Causal sidebar */}
        {panel && (
          <div className="w-56 shrink-0 bg-gray-900 border-l border-gray-800 p-4 overflow-y-auto text-xs hidden sm:block">
            <h2 className="text-cyan-400 font-bold mb-3 text-sm">⚛ URI Theory</h2>
            <div className="space-y-3">
              <div>
                <p className="text-gray-500 text-xs mb-1">Relational constant</p>
                <p className="text-cyan-300 text-2xl font-bold">α=1.274</p>
              </div>
              <div>
                <p className="text-gray-500 text-xs">Hawking correction</p>
                <p className="text-gray-400 text-xs mt-0.5">T_PRD = T_H·(1+α·l²P/A)</p>
              </div>
              <div>
                <p className="text-gray-500 text-xs">Causal transform</p>
                <p className="text-gray-400 text-xs mt-0.5">Ψ_out = Σ ω_i Ĝ_i Ψ_in</p>
              </div>
              <div>
                <p className="text-gray-500 text-xs mb-2">24 Paccaya</p>
                {Object.entries(PACCAYA_COLORS).map(([n,c])=>(
                  <div key={n} className="flex items-center gap-2 mb-1.5">
                    <div className="w-2 h-2 rounded-full shrink-0" style={{backgroundColor:c}}/>
                    <span className="text-gray-400 capitalize text-xs">{n}</span>
                  </div>
                ))}
              </div>
              <div>
                <p className="text-gray-500 text-xs">Counseling flow</p>
                <p className="text-gray-400 text-xs mt-0.5">Upanissaya → Nissaya → Asevana</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
