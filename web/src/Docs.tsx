// The docs: what the settings and numbers mean, for people who use anvil from the screen and never see its code. For
// now, picking the best checkpoint and every score a check gives. #/docs/<id> scrolls to a part, such as one score.

import { IconInfoCircle } from "@tabler/icons-react";
import { useEffect, type ReactNode } from "react";
import { Topbar } from "./Topbar";
import { Badge, Section } from "./ui";

interface Score {
  id: string; // its address: #/docs/<id>
  name: string; // as Pick best by lists it
  tag?: string;
  needs: string; // the questions it needs
  better: "Higher" | "Lower";
  what: ReactNode;
  training: ReactNode;
  when: ReactNode;
}

const SCORES: Score[] = [
  {
    id: "overall",
    name: "Overall",
    tag: "The default",
    needs: "Mixed question types",
    better: "Higher",
    what: (
      <>
        Every question gets its own 0-to-1 score: AUROC for yes/no, macro-F1 for choice, and 1 minus the absolute
        error over the levels' range for score questions. Overall is their average, so each question counts the same,
        however many rows ask it.
      </>
    ),
    training: (
      <>
        The scores under it give rare answers full weight, so a model can't look good by learning only the common ones.
        It's an average, so the questions with the most room left move it most: yes/no AUROC often gets close to 1
        early, which leaves the choice and score questions to drive it late in a run.
      </>
    ),
    when: <>Every question matters. That's the usual case, and why it's the default.</>,
  },
  {
    id: "auroc",
    name: "Mean AUROC",
    needs: "Yes/no questions",
    better: "Higher",
    what: (
      <>
        For each yes/no question, the chance that a random yes case gets a higher yes score than a random no case: 0.5
        is guessing, 1 is perfect. Averaged over the yes/no questions.
      </>
    ),
    training: (
      <>
        It doesn't depend on a cut-off, and a rare yes doesn't fool it. But it says nothing about where the 50% cut-off
        lands: a model can rank perfectly and still flag too many no cases at 50% (the run page's False alarms column
        shows that). It often climbs fast and then creeps, so late in a run it can barely tell checkpoints apart.
      </>
    ),
    when: <>You'll set your own cut-off after training, or you use the ranking itself, such as sorting cases by risk.</>,
  },
  {
    id: "catch-rate",
    name: "Catch rate at N% false alarms",
    needs: "Yes/no questions",
    better: "Higher",
    what: (
      <>
        The share of real yes cases caught when the cut-off is set so that at most N% of no cases are flagged. N is{" "}
        <strong>False alarms allowed</strong>, next to Pick best by (1% unless you change it). Averaged over the yes/no
        questions.
      </>
    ),
    training: (
      <>
        It's the number a guard lives by, but it's jumpy on a small validation set. With 200 no cases at 1%, the cut-off
        is the third-highest no case's score, so one confidently wrong no moves it; with 15 yes cases, each one caught
        is worth almost 7 points.
      </>
    ),
    when: (
      <>
        The model is a guard or a filter, and flagging an innocent case costs something. Set the allowance to the false
        alarm rate you can live with.
      </>
    ),
  },
  {
    id: "macro-f1",
    name: "Mean macro-F1",
    needs: "Choice questions",
    better: "Higher",
    what: (
      <>
        For each answer, F1 weighs how many of the model's picks of it were right against how many of the real ones it
        found. Macro-F1 averages that over the answers, so a rare answer counts as much as a common one. Averaged over
        the choice questions (Macro-F1 on the run page).
      </>
    ),
    training: (
      <>
        Early on, a model often leans on the common answers: accuracy looks fine while macro-F1 stays low. It rises as
        the model learns the rare ones, sometimes while accuracy dips; in one run, a three-answer question went from
        0.40 to 0.48 while its accuracy fell from 61% to 53%. It's built on the model's top answer, so it jumps
        between checks when an answer has few validation rows. The By question table marks answers seen fewer than 30
        times in amber.
      </>
    ),
    when: <>Your questions are choices and the rare answers matter, which is most of the time.</>,
  },
  {
    id: "mae",
    name: "Mean absolute error",
    needs: "Score questions",
    better: "Lower",
    what: (
      <>
        How many levels off the model's answer is, on average: answering 3 when the truth is 1 is 2 off. Averaged over
        the score questions (MAE on the run page).
      </>
    ),
    training: (
      <>
        It rewards near misses, which exact accuracy doesn't: a model that's usually one level off beats one that's
        sometimes exact and sometimes far off. It isn't scaled, so a question with more levels weighs more in the
        mean; Overall divides each question's error by its range first.
      </>
    ),
    when: <>The answers are ratings, such as severity or quality, and close counts.</>,
  },
  {
    id: "calibration",
    name: "Calibration",
    needs: "Any decision data",
    better: "Lower",
    what: (
      <>
        How far the model's confidence is from how often it's right: answers given at 80% should be right about 80% of
        the time. The gaps are averaged over 10 confidence bands, weighted by how many answers fall in each (Calib.
        error on the run page).
      </>
    ),
    training: (
      <>
        Fine-tuning tends to make a model surer of itself, so calibration can slip late in a run even while the
        answers improve.
      </>
    ),
    when: (
      <>
        Not on its own: a model that always says "no, 90% sure" on data that's 90% no is perfectly calibrated and
        useless. When you need honest probabilities, pick by validation loss and check calibration on the checkpoint
        it picks.
      </>
    ),
  },
  {
    id: "accuracy",
    name: "Accuracy",
    needs: "Any decision data",
    better: "Higher",
    what: (
      <>
        The share of questions answered right, over every question and row. For score questions only the exact level
        counts.
      </>
    ),
    training: (
      <>
        Easy to read, but the common answers dominate it: with 19 yes cases in 250, always answering no scores 92%. A
        model that learns only the common answers looks good on it.
      </>
    ),
    when: <>The answers are about equally common, and every mistake costs the same.</>,
  },
  {
    id: "val-loss",
    name: "Validation loss",
    needs: "Any data",
    better: "Lower",
    what: (
      <>
        How much probability the model gave the right answer, averaged as −log p: 0 when it's sure and right, large
        when it's sure and wrong. For a language model it's measured on the tokens it learns (the assistant's replies,
        the completion, or all of the text), and the run page's Perplexity is e to the loss: roughly how many tokens
        the model is choosing between at each step.
      </>
    ),
    training: (
      <>
        The standard sign of overfitting: when the training loss keeps falling but validation loss turns up, the model
        has started memorising its training rows. For decision models it punishes confident mistakes hard, so it can
        stall while the answers keep improving. In one Clef run it called steps 2,000 and 3,000 a tie (0.465 and
        0.466) and kept the earlier one, while Overall picked step 3,000, which caught 11 of a rare question's 15 yes
        cases instead of 9 and raised mean macro-F1 from 0.46 to 0.51.
      </>
    ),
    when: (
      <>
        Always, for a language model: it's the only choice. For a decision model, when you'll use its probabilities as
        they are, such as with your own cut-off or as a risk score: it rewards being right and being honestly confident
        at once.
      </>
    ),
  },
];

// What matters most -> the score to pick best by (the id of its entry above).
const GUIDE: [string, string, string][] = [
  ["Every question, rare answers included (the usual case)", "overall", "Overall"],
  ["Ranking yes/no cases well, with your own cut-off later", "auroc", "Mean AUROC"],
  ["A guard: catching yes cases while rarely flagging a no", "catch-rate", "Catch rate"],
  ["Getting rare answers to choice questions right", "macro-f1", "Mean macro-F1"],
  ["Ratings that land close, even when not exact", "mae", "Mean absolute error"],
  ["Answers about equally common, mistakes that cost the same", "accuracy", "Accuracy"],
  ["Probabilities you'll use as they are (check calibration too)", "val-loss", "Validation loss"],
  ["A language model, where it's the only choice", "val-loss", "Validation loss"],
];

export function Docs({ part, onLogout }: { part: string | null; onLogout: () => void }) {
  useEffect(() => {
    if (part) document.getElementById(part)?.scrollIntoView();
  }, [part]);

  return (
    <div className="page">
      <Topbar onLogout={onLogout} />
      <div className="page-head">
        <h1>Docs</h1>
      </div>
      <p className="doc-lead">How anvil picks the best checkpoint, and what each of its scores says about a fine-tune.</p>

      <Section title="Picking the best checkpoint">
        <div className="doc-text">
          <p>
            Each checkpoint is checked on the validation set and gets several scores. <strong>Pick best by</strong>,
            under Checkpoints on New run, says which one ranks them, and the best checkpoint becomes the model. The same
            score decides which checkpoints <strong>Keep best</strong> leaves on disk, and when{" "}
            <strong>Stop after no gain for</strong> ends a run early, so it steers the whole run, not just the final
            pick.
          </p>
          <p>
            This is the recipe the field uses (Hugging Face's metric_for_best_model and load_best_model_at_end, Keras's
            and Lightning's EarlyStopping): one monitored score with a direction, the best checkpoint kept and loaded
            at the end, patience counted in checks without a gain, and a least gain that counts (their min_delta). A
            check beats the best only by more than <strong>Least gain that counts</strong>; a smaller rise neither
            moves the pick nor starts the no-gain count again. 0 counts any rise, which is their default too. Every
            score comes with ± one standard error on the run page, worked out by resampling the validation rows: set
            the least gain to about that, and a run no longer stops late because the score drifted up by noise, or
            picks a checkpoint for a gain that wasn't one.
          </p>
          <p className="doc-callout">
            <IconInfoCircle size={16} aria-hidden />
            <span>
              <strong>Not sure? Use the default.</strong> It's Overall when the validation set mixes question types.
              With only one type, the form switches to that type's own score (Mean AUROC, Mean macro-F1 or Mean absolute
              error), since Overall would say much the same. Language models are always picked by validation loss.
            </span>
          </p>
        </div>
        <table className="doc-table">
          <thead>
            <tr>
              <th>What matters most</th>
              <th>Pick best by</th>
            </tr>
          </thead>
          <tbody>
            {GUIDE.map(([matters, id, pick]) => (
              <tr key={matters}>
                <td>{matters}</td>
                <td>
                  <a href={`#/docs/${id}`}>{pick}</a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="What each score means">
        {SCORES.map((s) => (
          <article key={s.id} id={s.id} className="doc-score">
            <div className="doc-score-head">
              <h3>{s.name}</h3>
              {s.tag && <Badge tone="accent">{s.tag}</Badge>}
            </div>
            <p className="hint">
              {s.needs} · {s.better} is better
            </p>
            <dl className="doc-facts">
              <dt>What it is</dt>
              <dd>{s.what}</dd>
              <dt>While training</dt>
              <dd>{s.training}</dd>
              <dt>Pick it when</dt>
              <dd>{s.when}</dd>
            </dl>
          </article>
        ))}
      </Section>

      <Section title="Reading the scores">
        <ul className="doc-list">
          <li>
            <strong>Step 0</strong> is the model before training, shown for comparison and never picked. The first
            checks can score below it while the model adjusts; a run whose checks never beat it hasn't learned the task
            yet.
          </li>
          <li>
            <strong>Small validation sets make every score jumpy</strong>, most of all the ones built on the top answer:
            accuracy, macro-F1 and the catch rate. Validation loss is the steadiest: it's a proper scoring rule, which
            rewards honest probabilities, while AUROC only ranks and the top-answer scores depend on a cut-off. Checks
            use up to 2,000 validation rows; below that, a bigger Validation&nbsp;% steadies them, at the cost of
            training rows. The ± beside the picked score on the run page is one standard error: with 19 yes cases a
            yes/no question's AUROC carries about ±0.08, and two checks within about one ± of each other are a
            toss-up, so the pick between them is luck, and the picked score flatters the model by about that much.
          </li>
          <li>
            <strong>The test score is fair whatever you pick.</strong> The test set is used once, at the end, on the
            picked checkpoint, so it plays no part in picking.
          </li>
          <li>
            <strong>You can change your mind, within limits.</strong> The run page shows every score at every check,
            and Export as GGUF offers any kept checkpoint. Keep best keeps the top checkpoints by the score you picked
            (plus the newest), so a higher Keep best leaves more to choose from.
          </li>
        </ul>
      </Section>
    </div>
  );
}
