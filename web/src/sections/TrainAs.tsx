import { pickByOptions, withMethod, type FormState, type SectionProps } from "../form";
import { Field, Section, Segmented } from "../ui";

const METHOD_HINTS = {
  clef: "A small new network, the head, reads the model and scores every option: all of a row's questions in one pass. Cloudflare's Clef design.",
  letters: "The model picks a letter for each option, one question per pass.",
  chat: "Conversations, trained through the model's chat template.",
  text: "Raw text, for continued pre-training or plain completion.",
};

export function TrainAs({ form, update }: SectionProps) {
  const decision = form.trainAs === "decision";
  return (
    <Section title="Train as">
      <Segmented
        label="Train as"
        value={form.trainAs}
        options={[
          { value: "lm", label: "Language model" },
          { value: "decision", label: "Decision model" },
        ]}
        onChange={(trainAs) =>
          update((f) => ({
            ...withMethod(f, { trainAs }),
            // the first that suits; Checkpoints moves it on if the prepared data can't score it
            checkpoints: { ...f.checkpoints, pickBy: pickByOptions(trainAs, null, f.checkpoints.falseAlarmPercent)[0].value },
          }))
        }
      />
      <div className="row">
        <div className="field">
          <span className="label">{decision ? "Method" : "Data format"}</span>
          {decision ? (
            <Segmented
              label="Method"
              value={form.decisionMethod}
              options={[
                { value: "clef", label: "Clef head" },
                { value: "letters", label: "Letters" },
              ]}
              onChange={(decisionMethod) => update((f) => withMethod(f, { decisionMethod }))}
            />
          ) : (
            <Segmented
              label="Data format"
              value={form.lmFormat}
              options={[
                { value: "chat", label: "Chat" },
                { value: "text", label: "Plain text" },
              ]}
              onChange={(lmFormat) => update((f) => ({ ...f, lmFormat }))}
            />
          )}
          <p className="hint">{METHOD_HINTS[decision ? form.decisionMethod : form.lmFormat]}</p>
        </div>
        <Field id="training-type" label="Training type"
          hint={form.trainingType === "full"
            ? "Every weight trains, in bf16: about 8 bytes a parameter plus the activations with AdamW, so a 2B model fits a 32 GB card. Small updates aren't lost: the optimizer rounds them stochastically. fp32 weights, at twice the memory, are under Advanced."
            : undefined}>
          <select
            id="training-type"
            value={form.trainingType}
            onChange={(e) => update((f) => withMethod(f, { trainingType: e.target.value as FormState["trainingType"] }))}
          >
            <option value="lora">LoRA</option>
            <option value="qlora">QLoRA (4-bit)</option>
            <option value="full">Full</option>
          </select>
        </Field>
      </div>
    </Section>
  );
}
