// Small building blocks shared by every section of the screen.

import { IconAlertTriangle, IconCheck, IconInfoCircle, IconX } from "@tabler/icons-react";
import type { ReactNode } from "react";
import type { Report, Status } from "./types";

export function Section({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="section">
      <div className="section-head">
        <h2>{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Segmented<T extends string>(props: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={props.label}>
      {props.options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={props.value === o.value}
          className={props.value === o.value ? "on" : ""}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// Under the input: the error if there is one, else a warning (it can go ahead, but should know), else the hint.
export function Field(props: {
  id: string;
  label: string;
  error?: string;
  warn?: string;
  hint?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={props.className ? `field ${props.className}` : "field"}>
      <label htmlFor={props.id}>{props.label}</label>
      {props.children}
      {props.error ? (
        <p className="field-error" role="alert">
          {props.error}
        </p>
      ) : props.warn ? (
        <p className="warn-text">{props.warn}</p>
      ) : props.hint ? (
        <p className="hint">{props.hint}</p>
      ) : null}
    </div>
  );
}

// A number input that keeps what you type (even while it's briefly invalid) and reports numbers upward.
export function NumberInput(props: {
  id: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  invalid?: boolean;
}) {
  return (
    <input
      id={props.id}
      type="number"
      inputMode="decimal"
      value={Number.isNaN(props.value) ? "" : props.value}
      step={props.step ?? 1}
      min={props.min}
      aria-invalid={props.invalid || undefined}
      onChange={(e) => props.onChange(e.target.value === "" ? Number.NaN : Number(e.target.value))}
    />
  );
}

export function Badge({ tone = "neutral", className, title, children }: {
  tone?: "ok" | "warn" | "bad" | "neutral" | "accent";
  className?: string;
  title?: string;
  children: ReactNode;
}) {
  return <span className={`badge badge-${tone}${className ? ` ${className}` : ""}`} title={title}>{children}</span>;
}

const ICONS: Record<Status, ReactNode> = {
  ok: <IconCheck size={16} aria-hidden />,
  info: <IconInfoCircle size={16} aria-hidden />,
  warn: <IconAlertTriangle size={16} aria-hidden />,
  bad: <IconX size={16} aria-hidden />,
};

export function Checklist({ report }: { report: Report }) {
  return (
    <div className="checklist">
      {report.items.map((item) => (
        <div key={item.label} className={`check check-${item.status}`}>
          {ICONS[item.status]}
          <span className="check-label">{item.label}</span>
          <span className="check-detail">{item.detail}</span>
        </div>
      ))}
      <p className={`verdict verdict-${report.verdict}`}>{report.message}</p>
    </div>
  );
}
