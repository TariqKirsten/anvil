// A language model's sample replies: at every check the trainer answers the same three held-out prompts, so what the
// model has learned can be read next to the reply the data gives, check by check, along with whether it stops by itself.

import { useState } from "react";
import type { Sample } from "./types";
import { Badge, Field, Section } from "./ui";

export interface SampledCheck {
  key: string;
  label: string;
  samples: Sample[];
}

export function Samples({ checks, preferred }: { checks: SampledCheck[]; preferred: string }) {
  const [chosen, setChosen] = useState<string | null>(null); // null: follow the preferred check as the run goes on
  const shown = checks.find((c) => c.key === chosen) ?? checks.find((c) => c.key === preferred) ?? checks[0];
  return (
    <Section title="Sample replies">
      <div className="question-pick">
        <Field id="sample-check" label="Replies from">
          <select id="sample-check" value={shown.key} onChange={(e) => setChosen(e.target.value)}>
            {checks.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
        </Field>
      </div>
      <div className="samples">
        {shown.samples.map((s, i) => (
          <div key={i} className="sample">
            <div className="sample-part">
              <span className="sample-label">Prompt</span>
              <pre className="sample-prompt">{s.prompt}</pre>
            </div>
            <div className="sample-pair">
              <div className="sample-part">
                <span className="sample-label">
                  The model
                  {s.ended ? <Badge tone="ok">ended</Badge> : <Badge tone="warn">cut off at {s.tokens} tokens</Badge>}
                </span>
                <pre>{s.reply || "(nothing)"}</pre>
              </div>
              <div className="sample-part">
                <span className="sample-label">The data</span>
                <pre>{s.expected || "(no reply in the data)"}</pre>
              </div>
            </div>
          </div>
        ))}
      </div>
      <p className="hint">
        The same three validation prompts at every check, answered greedily in up to 96 tokens; the final test answers
        three test prompts. A long prompt shows its end. Ended: the model stopped by itself, as a finished model must;
        cut off: it ran out of room, so it may not have learned to stop yet.
      </p>
    </Section>
  );
}
