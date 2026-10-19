// Each question's scores at one check, or at the final test: a table per question type, one row per question, with
// how many rows asked it and how often each answer came up, so a score that rests on a handful of cases shows.

import { useState } from "react";
import type { QuestionScore, QuestionType, Scores } from "./types";
import { Field, Section } from "./ui";

const FEW = 30; // an answer that came up fewer times than this makes the scores beside it rough

export interface ScoredCheck {
  key: string;
  label: string;
  scores: Scores;
}

// Runs from before each question's scores were kept have only their yes/no questions, as auroc_<name>,
// catch_<name> and false_alarms_<name>.
export function questionsOf(scores: Scores): Record<string, QuestionScore> | null {
  if (scores.questions) return scores.questions;
  const names = Object.keys(scores).filter((k) => k.startsWith("auroc_")).map((k) => k.slice(6));
  if (!names.length) return null;
  const value = (key: string) => (typeof scores[key] === "number" ? (scores[key] as number) : undefined);
  return Object.fromEntries(names.map((name) => [name, {
    type: "noul" as const,
    auroc: value(`auroc_${name}`),
    catch_rate: value(`catch_${name}`),
    false_alarms: value(`false_alarms_${name}`),
  }]));
}

function num(v: number | undefined, digits = 3) {
  return typeof v === "number" ? v.toFixed(digits) : "–";
}

function whole(v: number | undefined) {
  return typeof v === "number" ? v.toLocaleString() : "–";
}

function few(n: number | undefined) {
  return typeof n === "number" && n < FEW ? "few" : undefined;
}

// "fine: 6 · spam: 3"
function spread(answers: Record<string, number> | undefined) {
  return answers ? Object.entries(answers).map(([answer, n]) => `${answer}: ${n.toLocaleString()}`).join(" · ") : "–";
}

function fewest(answers: Record<string, number> | undefined) {
  const counts = Object.values(answers ?? {});
  return counts.length ? Math.min(...counts) : undefined;
}

const GROUPS: { type: QuestionType; title: string }[] = [
  { type: "noul", title: "Yes/no questions" },
  { type: "choice", title: "Choice questions" },
  { type: "score", title: "Score questions" },
];

export function QuestionScores({ checks, preferred, falseAlarmPercent }: {
  checks: ScoredCheck[];
  preferred: string;
  falseAlarmPercent: number;
}) {
  const [chosen, setChosen] = useState<string | null>(null); // null: follow the preferred check as the run goes on
  const shown = checks.find((c) => c.key === chosen) ?? checks.find((c) => c.key === preferred) ?? checks[0];
  const questions = questionsOf(shown.scores) ?? {};
  const names = Object.keys(questions).sort();
  const of = (type: QuestionType) => names.filter((n) => questions[n].type === type);
  const types = GROUPS.filter((g) => of(g.type).length > 0).map((g) => g.type);

  return (
    <Section title="By question">
      <div className="question-pick">
        <Field id="question-check" label="Scores from">
          <select id="question-check" value={shown.key} onChange={(e) => setChosen(e.target.value)}>
            {checks.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
        </Field>
      </div>

      {GROUPS.filter((g) => types.includes(g.type)).map((g) => (
        <div key={g.type} className="question-group">
          <h3>
            {g.title} <span className="hint">{of(g.type).length}</span>
          </h3>
          <div className="preview-scroll">
            <table className="table">
              <thead>
                {g.type === "noul" ? (
                  <tr>
                    <th>Question</th>
                    <th title="Rows that asked it">Rows</th>
                    <th title="Rows whose answer is yes">Yes</th>
                    <th title="Rows whose answer is no">No</th>
                    <th title="How well yes cases rank above no cases: 0.5 is chance, 1 is perfect">AUROC</th>
                    <th title={`Yes cases caught when at most ${falseAlarmPercent}% of no cases may be flagged`}>
                      Catch rate
                    </th>
                    <th title="Share of no cases flagged at 50%">False alarms</th>
                  </tr>
                ) : (
                  <tr>
                    <th>Question</th>
                    <th title="Rows that asked it">Rows</th>
                    <th className="question-answers" title="How often each answer is the right one">
                      {g.type === "choice" ? "Answers" : "Levels"}
                    </th>
                    <th title="The rarest answer's count">Fewest</th>
                    {g.type === "choice" ? (
                      <th title="F1 per answer, averaged, so a rare answer counts as much as a common one">Macro-F1</th>
                    ) : (
                      <th title="How many levels off, on average (lower is better)">MAE</th>
                    )}
                    <th title={g.type === "choice" ? "Share answered right" : "Share at exactly the right level"}>
                      {g.type === "choice" ? "Accuracy" : "Exact"}
                    </th>
                  </tr>
                )}
              </thead>
              <tbody>
                {of(g.type).map((name) => {
                  const q = questions[name];
                  if (g.type === "noul")
                    return (
                      <tr key={name}>
                        <td>{name}</td>
                        <td>{whole(q.rows)}</td>
                        <td className={few(q.answers?.yes)}>{whole(q.answers?.yes)}</td>
                        <td className={few(q.answers?.no)}>{whole(q.answers?.no)}</td>
                        <td>{num(q.auroc)}</td>
                        <td>{num(q.catch_rate)}</td>
                        <td>{num(q.false_alarms)}</td>
                      </tr>
                    );
                  const rarest = fewest(q.answers);
                  return (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{whole(q.rows)}</td>
                      <td className="question-answers" title={spread(q.answers)}>{spread(q.answers)}</td>
                      <td className={few(rarest)}>{whole(rarest)}</td>
                      <td>{num(g.type === "choice" ? q.macro_f1 : q.mae)}</td>
                      <td>{num(q.accuracy)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      <p className="hint">
        Numbers in amber: an answer that came up fewer than {FEW} times, so the scores beside it can swing a lot from
        one check to the next.
        {types.includes("noul") && ` AUROC: how well yes cases rank above no cases (0.5 is chance, 1 perfect). Catch rate: the share of yes cases caught when at most ${falseAlarmPercent}% of no cases may be flagged. False alarms: the share of no cases flagged at 50%.`}
        {types.includes("choice") && " Macro-F1: the F1 score of each answer, averaged, so a rare answer counts as much as a common one."}
        {types.includes("score") && " MAE: how many levels off the answer is, on average (lower is better); Exact: the share at exactly the right level."}
      </p>
    </Section>
  );
}
