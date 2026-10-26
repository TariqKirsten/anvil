// The pipeline assistant beside the canvas: a conversation, free form. It talks a new pipeline through and builds it when
// asked; what it does shows as lines between the messages. While it works the canvas is locked, and Undo puts back what
// its newest changes changed.

import {
  IconArrowBackUp, IconEye, IconHammer, IconLoader2, IconPlayerStop, IconSearch, IconSend, IconSparkles, IconX,
} from "@tabler/icons-react";
import { Fragment, useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { api } from "../api";
import { TURN_WORKING, type AssistantMessage, type AssistantTurn, type Conversation } from "./graph";

const EXAMPLES = [
  "Short messages bank customers send, labelled with whether they're about fraud",
  "Label the messages in a dataset I already have",
  "Questions and short, friendly answers for a support chat model",
];
const CHANGES = new Set(["set_pipeline", "change_step", "add_step", "remove_step"]);

function message(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

// What scrolls the conversation: the side panel on a wide screen. On a phone the page scrolls, and the conversation
// stays where the person put it.
function panelOf(el: HTMLElement | null): HTMLElement | null {
  for (let p = el?.parentElement ?? null; p; p = p.parentElement) {
    const overflow = getComputedStyle(p).overflowY;
    if (overflow === "auto" || overflow === "scroll") return p;
  }
  return null;
}

export function AssistantView({ pipelineId, conversation, onChange, beforeSend, onPipelineChanged }: {
  pipelineId: string;
  conversation: Conversation | null;
  onChange: (c: Conversation) => void;
  beforeSend: () => Promise<void>; // the canvas's waiting changes are saved first, so the assistant sees them
  onPipelineChanged: () => void; // Undo changed the pipeline: the canvas reads it again
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  const log = useRef<HTMLDivElement>(null);
  const following = useRef(true); // at the newest message, so new ones scroll into view
  const messages = conversation?.messages ?? [];
  const turn = conversation?.turn ?? null;
  const working = !!turn && TURN_WORKING.includes(turn.state);

  // The newest message stays in view as they come, unless the person has scrolled up to read.
  useEffect(() => {
    const panel = panelOf(log.current);
    if (!panel) return;
    const onScroll = () => {
      following.current = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 120;
    };
    panel.addEventListener("scroll", onScroll, { passive: true });
    return () => panel.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const panel = panelOf(log.current);
    if (panel && following.current) panel.scrollTop = panel.scrollHeight;
  }, [messages.length, turn?.doing, turn?.state]);

  async function send() {
    const said = text.trim();
    if (!said || busy || working) return;
    setBusy(true);
    setError(null);
    try {
      await beforeSend();
      following.current = true; // the answer is what they'll want to see
      onChange(await api<Conversation>(`/api/pipelines/${pipelineId}/assistant`, { json: { message: said } }));
      setText("");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  async function act(path: string, method = "POST") {
    setError(null);
    try {
      await api(`/api/pipelines/${pipelineId}/assistant${path}`, { method });
      onChange(await api<Conversation>(`/api/pipelines/${pipelineId}/assistant`));
    } catch (e) {
      setError(message(e));
    }
  }

  async function undo() {
    setError(null);
    try {
      onChange(await api<Conversation>(`/api/pipelines/${pipelineId}/assistant/undo`, { method: "POST" }));
      onPipelineChanged();
    } catch (e) {
      setError(message(e));
    }
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <div className="assistant">
      {messages.length > 0 && (
        <div className="assistant-tools">
          <span className="hint" title="Settings chooses the assistant's model">
            <IconSparkles size={14} aria-hidden /> {conversation?.model || "No model chosen"}
          </span>
          {conversation?.undo && !working && (
            <button type="button" className="link" onClick={undo} title="Put back what the assistant's newest changes changed">
              <IconArrowBackUp size={15} aria-hidden /> Undo its changes
            </button>
          )}
          {!working && (
            clearing ? (
              <span className="assistant-confirm">
                Clear the conversation?
                <button type="button" className="link danger-link" onClick={() => {
                  setClearing(false);
                  void act("", "DELETE");
                }}>Clear</button>
                <button type="button" className="link" onClick={() => setClearing(false)}>Keep</button>
              </span>
            ) : (
              <button type="button" className="link" onClick={() => setClearing(true)}
                title="The conversation goes; the pipeline stays as it is">Start over</button>
            )
          )}
        </div>
      )}

      <div className="assistant-log" aria-live="polite" ref={log}>
        {messages.length === 0 ? (
          <Intro onExample={(example) => setText(example)} />
        ) : (
          messages.map((m, i) => <Message key={i} m={m} />)
        )}
        {turn && <TurnLine turn={turn} pipelineId={pipelineId} onAct={act} />}
      </div>

      <div className="ask-box">
        {error && <p className="error-box">{error}</p>}
        {conversation && !conversation.ready ? (
          <p className="hint">Choose the assistant's model in <a href="#/settings">Settings</a> first.</p>
        ) : (
          <>
            <textarea aria-label="Message to the assistant" value={text} rows={3} disabled={working}
              placeholder="Describe the data you want, or ask about the pipeline"
              onChange={(e) => setText(e.target.value)} onKeyDown={onKey} />
            <div className="ask-actions">
              <span className="hint">Enter sends, Shift+Enter starts a new line.</span>
              {working ? (
                <button type="button" onClick={() => act("/stop")} disabled={turn?.state === "starting"}>
                  <IconPlayerStop size={16} aria-hidden /> Stop
                </button>
              ) : (
                <button type="button" className="primary" onClick={send} disabled={busy || !text.trim()}>
                  {busy ? <IconLoader2 size={16} className="spin" aria-hidden /> : <IconSend size={16} aria-hidden />}
                  Send
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Intro({ onExample }: { onExample: (text: string) => void }) {
  return (
    <div className="assistant-intro">
      <h2><IconSparkles size={16} aria-hidden /> Describe the data you want</h2>
      <p className="side-text">Tell the assistant what you need, and talk it through. It can look at your models,
        datasets and templates, asks what it needs to, and proposes a pipeline; say so and it builds it and previews a
        few rows. Undo puts back anything it changed.</p>
      <div className="assistant-examples">
        {EXAMPLES.map((example) => (
          <button key={example} type="button" className="learn-chip" onClick={() => onExample(example)}>{example}</button>
        ))}
      </div>
    </div>
  );
}

function Message({ m }: { m: AssistantMessage }) {
  if (m.role === "user") return <div className="said said-you"><p>{m.text}</p></div>;
  if (m.role === "assistant") return <div className="said said-assistant"><Markdown text={m.text} /></div>;
  if (m.role === "note") return <p className="said-note">{m.text}</p>;
  const icon = !m.ok ? <IconX size={14} aria-hidden /> : m.tool === "preview" ? <IconEye size={14} aria-hidden />
    : CHANGES.has(m.tool ?? "") ? <IconHammer size={14} aria-hidden /> : <IconSearch size={14} aria-hidden />;
  return <p className={`said-action${m.ok ? "" : " failed"}`}>{icon}<span>{m.text}</span></p>;
}

// What the turn is doing now, what it waits for, or why it failed.
function TurnLine({ turn, pipelineId, onAct }: { turn: AssistantTurn; pipelineId: string;
  onAct: (path: string) => void }) {
  const [stopping, setStopping] = useState(false);
  if (turn.state === "failed" && turn.error) return <p className="error-box">{turn.error}</p>;
  if (turn.state === "waiting" && turn.waiting_for) {
    const w = turn.waiting_for;
    async function stopRun() {
      setStopping(true);
      await api(`/api/pipelines/${w.pipeline}/runs/${w.run}/stop`, { method: "POST" }).catch(() => null);
    }
    return (
      <div className="assistant-wait">
        <p>
          <IconLoader2 size={15} className="spin" aria-hidden />{" "}
          {w.training ? <>Waiting: {w.training} is training on the GPUs the assistant's model loads onto. It answers
            when training ends.</>
            : <>Waiting: {w.pipeline === pipelineId ? "this pipeline's" : `${w.name ?? w.pipeline}'s`}{" "}
              {w.run === "preview" ? "preview" : "run"} is using its server with {w.model}, and a server that holds one
              model at a time can't load the assistant's until that ends.</>}
        </p>
        <div className="confirm-actions">
          {!w.training && w.pipeline && w.run && (
            <button type="button" onClick={stopRun} disabled={stopping}>
              <IconPlayerStop size={16} aria-hidden /> {stopping ? "Stopping…" : `Stop that ${w.run === "preview" ? "preview" : "run"}`}
            </button>
          )}
          <button type="button" onClick={() => onAct("/go")}>Ask anyway</button>
        </div>
      </div>
    );
  }
  if (!TURN_WORKING.includes(turn.state)) return null;
  return (
    <p className="assistant-doing">
      <IconLoader2 size={15} className="spin" aria-hidden /> {turn.doing ?? "Working"}…
    </p>
  );
}

// The little markdown a model writes: paragraphs, lists, **bold**, *italics* and `code`. Built as elements, so nothing
// it writes becomes HTML.
export function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r/g, "").split("\n");
  const bullet = /^\s*[-*•]\s+/;
  const numbered = /^\s*\d+[.)]\s+/;
  const heading = /^#{1,6}\s+/;
  const blocks: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    const list = bullet.test(line) ? bullet : numbered.test(line) ? numbered : null;
    if (list) {
      const items: string[] = [];
      while (i < lines.length && list.test(lines[i])) items.push(lines[i++].replace(list, ""));
      const children = items.map((item, k) => <li key={k}>{inline(item)}</li>);
      blocks.push(list === bullet ? <ul key={i}>{children}</ul> : <ol key={i}>{children}</ol>);
    } else if (heading.test(line)) {
      blocks.push(<p key={i} className="md-heading">{inline(line.replace(heading, ""))}</p>);
      i++;
    } else {
      const para: string[] = [];
      while (i < lines.length && lines[i].trim() && !bullet.test(lines[i]) && !numbered.test(lines[i])
        && !heading.test(lines[i])) para.push(lines[i++]);
      blocks.push(<p key={i}>{para.map((l, k) => <Fragment key={k}>{k > 0 && <br />}{inline(l)}</Fragment>)}</p>);
    }
  }
  return <div className="md">{blocks}</div>;
}

function inline(text: string): ReactNode[] {
  const found: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(/`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*/g)) {
    const at = m.index ?? 0;
    if (at > last) found.push(text.slice(last, at));
    const t = m[0];
    found.push(t.startsWith("`") ? <code key={at}>{t.slice(1, -1)}</code>
      : t.startsWith("**") ? <strong key={at}>{t.slice(2, -2)}</strong> : <em key={at}>{t.slice(1, -1)}</em>);
    last = at + t.length;
  }
  if (last < text.length) found.push(text.slice(last));
  return found;
}
