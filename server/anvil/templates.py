"""Pipelines to start from: each one a working example to learn from and change. New pipeline offers them.

Each template is a graph like any pipeline's (anvil.pipelines), with an about text for its About tab and a note on
every step saying why it's there and what to try. Their Models point at radiance's Gemma on this machine, so a
template runs once the Model has radiance's key. A Seed data step whose source is "sample" reads a file from
samples/, copied into the uploads folder (where datasets can be picked from) the first time it's used.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from . import config

SAMPLES = Path(__file__).resolve().parent / "samples"
GEMMA = {"base_url": "http://host.docker.internal:8000/v1", "model": "gemma4-31b", "thinking": "off", "parallel": 32,
         "max_tokens": 2048, "timeout": 300}
CONTEXT = "These are messages customers send to their bank's chat assistant."
ROW = 136  # one step's height on the canvas, with the gap below it
KEY = ("To run it, pick the Model step, paste radiance's key (./radiance.sh key prints it) and press Connect. Then "
       "Preview, and Run.")

WRITER_NOTE = ("The writer: radiance's Gemma at temperature 1, so it doesn't write the same thing twice. Every line "
               "starts at a Model.")
JUDGE_NOTE = ("The second opinion: Gemma at temperature 0, reading each message on its own, without knowing what it "
              "was asked to be. It's plugged into the Question's 2nd opinion handle with the dashed line.")
DEDUPE_NOTE = ("Models repeat themselves. Exact copies and near-copies (sharing 70% of their three-word runs, the rule "
               "Prepare data uses) are dropped, so no message is in the data twice.")

TEAMS = {  # option name: what it covers, the way the writer is told and the Question's option means it
    "cards": "lost, stolen or blocked cards, card limits and PINs",
    "payments": "transfers, EFTs, debit orders and payments that haven't arrived",
    "loans": "personal loans, overdrafts, credit cards and home loans",
    "fraud": "scams, phishing, and transactions the customer didn't make",
    "digital": "signing in, OTPs, and the app or website not working",
    "accounts": "opening, closing or changing an account and its details",
}
TEAM_VALUES = [f"{name}: {meaning}" for name, meaning in TEAMS.items()]
TEAM_QUESTION = {
    "name": "team", "type": "choice", "instructions": "Which team should handle the customer's message?",
    "options": [{"name": name, "meaning": meaning} for name, meaning in TEAMS.items()],
    "about": "message", "context": CONTEXT, "answer": "column", "from_column": "topic",
    "mapping": {value: value.split(":")[0] for value in TEAM_VALUES},
}
URGENCY = ["a question with no hurry", "a problem they'd like fixed soon", "something blocking them right now",
           "an emergency: money is going or gone"]
HARMFUL = ("Is the user's message a harmful request: asking for help, information or content that could cause real-world "
           "harm, such as fraud, theft, violence, weapons, drugs, self-harm, hacking, invading someone's privacy, hate or "
           "sexual content?")


def step(node_id: str, kind: str, row: float, note: str, x: float = 0, **cfg) -> dict:
    return {"id": node_id, "type": kind, "x": x, "y": row * ROW, "note": note, "config": cfg}


def line(*ids: str) -> list[dict]:
    return [{"id": f"{a}-rows-{b}", "from": a, "to": b, "kind": "rows"} for a, b in zip(ids, ids[1:])]


def second(model: str, question: str) -> dict:
    return {"id": f"{model}-check-{question}", "from": model, "to": question, "kind": "check"}


TEMPLATES: list[dict] = [
    {
        "id": "label-messages",
        "name": "Label messages you already have",
        "summary": "Start from a file of messages and let Gemma answer a yes/no question about each one.",
        "learn": ["Seed data: rows from a file", "A Question the model answers", "Keeping only sure answers",
                  "Saving a decision dataset"],
        "makes": "Decision · yes/no",
        "about": ("The quickest way to a decision dataset: messages you already have, and a model answering a question "
                  "about each one. This pipeline labels 24 sample bank messages that come with anvil. Point Seed data at "
                  "your own file to label yours.\n\n" + KEY + "\n\nTry next: change the question, add a second Question "
                  "step (every row then asks both), or untick \"Only keep answers the model is sure of\" and compare."),
        "nodes": [
            step("model", "model", 0, "Every line starts at a Model. This one is radiance's Gemma, with thinking off "
                 "(faster, and plenty for labelling) and temperature 0, so a message always gets the same answer.",
                 name="Gemma", temperature=0, **GEMMA),
            step("seed", "seed", 1, "Seed data starts the rows from a file: each of its rows becomes a row here, with "
                 "the columns ticked below. When they run out, so does the run.",
                 source="sample", columns=["text", "channel"], order="in_order"),
            step("ask", "question", 2, "A Question labels each row. Here the model answers it, reading the text column; "
                 "the context tells it where the messages come from. Only sure answers count: one the model isn't sure "
                 "of is left out, and the row asks the question without scoring it.",
                 name="frustrated", type="noul", instructions="Is the customer frustrated?",
                 yes="annoyed, angry, impatient or upset about how things are going", no="calm, neutral or happy",
                 about="text", context=CONTEXT, answer="model", sure_only=True),
            step("save", "save", 3, "Save writes the dataset. The state is what the decision model reads, "
                 "{\"message\": the text}, and each Question above becomes one of its questions. Rows is more than the "
                 "file holds, so the run labels all 24 and stops.",
                 name="labelled-messages", rows=100, format="decision", state_key="message", state_column="text"),
        ],
        "edges": line("model", "seed", "ask", "save"),
        "sample": "bank-messages.jsonl",
    },
    {
        "id": "yes-no-from-scratch",
        "name": "Write a yes/no dataset from scratch",
        "summary": "Gemma writes frustrated and calm bank messages; the mood it was told to write is the label, and a "
                   "second opinion checks it.",
        "learn": ["Lists that decide what gets written", "Several rows from one request", "Labels from a column",
                  "A second opinion", "Dedupe", "Balancing the answers"],
        "makes": "Decision · yes/no",
        "about": ("No data? A model can write it. The Lists decide what gets written, a topic and a mood for each "
                  "request, so you choose the mix rather than the model. The label is the mood the model was told to "
                  "write, and a second model checks it: a message that came out calm when it was meant to be frustrated "
                  "loses its label, and Save drops it.\n\nThis is how decision-0.8B's frustration flag was made. Run it, "
                  "then press Train on it in the Runs tab.\n\n" + KEY + "\n\nTry next: add a mood (\"polite but firm\"), "
                  "add topics, or raise Rows per reply to 8 and watch how many Dedupe drops."),
        "nodes": [
            step("model", "model", 0, WRITER_NOTE, name="Gemma", temperature=1.0, **GEMMA),
            step("topic", "list", 1, "A List gives each row a value. Topics at random spread the messages over what a "
                 "bank hears about. More topics means fewer near-copies for Dedupe to drop.",
                 column="topic", pick="random", values=[
                     "a card that was blocked, lost or stolen", "a transfer or EFT that hasn't arrived",
                     "fees or charges they don't understand", "resetting their PIN or app password",
                     "a debit order they want to stop", "the app being down during load-shedding",
                     "a loan or overdraft application", "a scam message they want to check",
                     "a SASSA grant or salary that hasn't come in", "updating their FICA documents",
                     "a cash deposit that isn't showing", "sending money abroad"]),
            step("mood", "list", 2, "Taking turns, so frustrated and calm come up as often as each other. The mood "
                 "steers the writing, and it becomes the label.",
                 column="mood", pick="turns", values=["frustrated", "calm"]),
            step("write", "generate", 3, "The model writes 4 messages a request, as a JSON list, and each becomes a "
                 "row: faster than one at a time, and more varied, because it sees the others. The prompt uses "
                 "{{ mood }} and {{ topic }} from the Lists above.",
                 column="message", per_reply=4, prompt=(
                     "Write short messages (one or two sentences) that a {{ mood }} customer sends to their bank's chat "
                     "assistant about {{ topic }}. Make each one different: vary the wording, the details and how "
                     "formal it is. Write only the messages.")),
            step("ask", "question", 4, "The label comes from the mood column: frustrated means yes, calm means no. The "
                 "judge plugged into the 2nd opinion handle reads each message too; where it's sure the label is "
                 "wrong, the answer is left out.",
                 name="frustrated", type="noul", instructions="Is the customer frustrated?", about="message",
                 context=CONTEXT, answer="column", from_column="mood", mapping={"frustrated": "yes", "calm": "no"}),
            step("dedupe", "dedupe", 5, DEDUPE_NOTE, column="message"),
            step("save", "save", 6, "1,000 rows, balanced by mood: 500 frustrated and 500 calm. The mood List then only "
                 "makes the mood still short, so nothing is written for nothing. Rows that lost their label have no "
                 "answer left, and are dropped here.",
                 name="bank-frustration", rows=1000, format="decision", state_column="message", balance="mood"),
            step("judge", "model", 4, JUDGE_NOTE, x=330, name="Gemma judge", temperature=0, **GEMMA),
        ],
        "edges": line("model", "topic", "mood", "write", "ask", "dedupe", "save") + [second("judge", "ask")],
    },
    {
        "id": "route-to-team",
        "name": "Route messages to a team",
        "summary": "A choice question for a new domain: which bank team should handle each customer's message.",
        "learn": ["A choice question and its options", "Mapping a List's values to answers",
                  "Varying how messages are written", "Balancing six answers"],
        "makes": "Decision · choice",
        "about": ("Decision models aren't only for safety flags. This one learns which team should take a bank "
                  "customer's message: a choice question, whose options each say what they cover. The topic List "
                  "tells the writer what the message is about, the Question's mapping turns that into the team's name, "
                  "and a judge checks each answer.\n\n" + KEY + "\n\nTry next: add a team (\"business banking\"), make the options' "
                  "meanings sharper, or add a tone that's hard to route, such as \"rambling\"."),
        "nodes": [
            step("model", "model", 0, WRITER_NOTE, name="Gemma", temperature=1.0, **GEMMA),
            step("topic", "list", 1, "One value per team, taking turns, each saying what the team covers, so the "
                 "writer knows what to write about. The Question's mapping turns each into the team's name.",
                 column="topic", pick="turns", values=TEAM_VALUES),
            step("tone", "list", 2, "How the customer writes, at random. Varying it teaches the decision model to route "
                 "by what a message is about, not how it sounds.",
                 column="tone", pick="random",
                 values=["polite", "in a hurry", "annoyed", "confused and unsure what's wrong", "very brief"]),
            step("write", "generate", 3, "Each request asks for 4 messages for one team, in one tone.",
                 column="message", per_reply=4, prompt=(
                     "Write short messages (one or two sentences) that a bank customer sends to their bank's chat "
                     "assistant. The customer is {{ tone }}, and the message is about {{ topic }}. Make each one "
                     "different: vary the details and the wording. Write only the messages.")),
            step("route", "question", 4, "A choice question: each option is an answer, and its meaning tells the model "
                 "(and later the decision model) when to pick it. The answer comes from the topic List, through the "
                 "mapping; the judge checks it.", **TEAM_QUESTION),
            step("dedupe", "dedupe", 5, DEDUPE_NOTE, column="message"),
            step("save", "save", 6, "Balanced by topic, so each team gets 200 of the 1,200 rows.",
                 name="bank-routing", rows=1200, format="decision", state_column="message", balance="topic"),
            step("judge", "model", 4, JUDGE_NOTE, x=330, name="Gemma judge", temperature=0, **GEMMA),
        ],
        "edges": line("model", "topic", "tone", "write", "route", "dedupe", "save") + [second("judge", "route")],
    },
    {
        "id": "several-questions",
        "name": "Several questions per message",
        "summary": "Every row asks three questions at once: a team (choice), an urgency (score) and a yes/no the "
                   "model answers.",
        "learn": ["A score question and its levels", "Several questions in one row",
                  "Answers from Lists and from the model, mixed"],
        "makes": "Decision · choice, score, yes/no",
        "about": ("A Clef decision model answers several questions about one message in a single pass, and it learns "
                  "best when its training rows ask them together, as decision-0.8B v7 does. Every row here asks three: "
                  "which team (a choice), how urgent (a score from 0 to 3), and whether the message holds private "
                  "details (yes or no).\n\nThe first two answers come from Lists that told the writer what to write. "
                  "The third is asked of the model, because only reading the message can tell.\n\n" + KEY +
                  "\n\nTry next: add a fourth question, or change urgency's levels and their mapping."),
        "nodes": [
            step("model", "model", 0, WRITER_NOTE + " It also answers the private-details question.",
                 name="Gemma", temperature=1.0, **GEMMA),
            step("topic", "list", 1, "What the message is about, one value per team, taking turns, as in Route "
                 "messages to a team.", column="topic", pick="turns", values=TEAM_VALUES),
            step("situation", "list", 2, "How urgent the customer's situation is, at random. The urgency Question maps "
                 "each value to a level.", column="situation", pick="random", values=URGENCY),
            step("write", "generate", 3, "One in four messages gets a made-up account, card or ID number, so the "
                 "private-details question has both answers to learn from.",
                 column="message", per_reply=3, prompt=(
                     "Write short messages (one or two sentences) that a bank customer sends to their bank's chat "
                     "assistant about {{ topic }}. Each is {{ situation }}. In about one message in four, include a "
                     "made-up account, card or ID number. Make each one different. Write only the messages.")),
            step("route", "question", 4, "The team, from the topic List through the mapping.", **TEAM_QUESTION),
            step("urgent", "question", 5, "A score question: levels from lowest to highest, and the answer is the "
                 "level's number. The mapping gives each situation its level.",
                 name="urgency", type="score", instructions="How urgent is the customer's message?", levels=[
                     "no hurry: a question, or something for later", "soon: a problem they want fixed in a day or two",
                     "now: something is blocking them right now", "emergency: money is being lost, or already has been"],
                 about="message", context=CONTEXT, answer="column", from_column="situation",
                 mapping={value: str(level) for level, value in enumerate(URGENCY)}),
            step("private", "question", 6, "Asked of the model: only reading the message can tell. Sure answers "
                 "only, so a guess doesn't become a label.",
                 name="private_details", type="noul", about="message", context=CONTEXT, answer="model",
                 sure_only=True, instructions="Does the message include private details, such as an account number, "
                 "a card number, an ID number or a password?"),
            step("save", "save", 7, "Every Question above asks in every row: one decision row holds all three answers, "
                 "the way a Clef model learns them best.",
                 name="bank-three-questions", rows=900, format="decision", state_column="message", balance="topic"),
        ],
        "edges": line("model", "topic", "situation", "write", "route", "urgent", "private", "save"),
    },
    {
        "id": "chat-replies",
        "name": "Chat replies for a language model",
        "summary": "Gemma writes customer questions and answers them, saved as chat conversations to fine-tune a "
                   "language model on.",
        "learn": ["Two Generate steps in a line", "A system message", "A Filter with two rules", "The chat format"],
        "makes": "Chat",
        "about": ("Language models learn from conversations. The first Generate step writes what customers ask; the "
                  "second answers each question as the assistant you'd like to train would. A Filter drops answers "
                  "that run long or slip out of character, and Save writes the rest in the chat format that New run "
                  "trains language models on.\n\n" + KEY + "\n\nTry next: change the assistant's system message, add a "
                  "tone List for the questions, or save as Prompt and completion instead."),
        "nodes": [
            step("model", "model", 0, WRITER_NOTE + " It writes both the questions and the answers.",
                 name="Gemma", temperature=1.0, **GEMMA),
            step("topic", "list", 1, "What the customers ask about, at random.", column="topic", pick="random", values=[
                "card fees", "a payment that bounced", "opening a savings account", "travel and using the card abroad",
                "home loan repayments", "setting up the banking app", "a disputed transaction",
                "budgeting and saving tips", "buying prepaid electricity", "an increase on a credit card limit"]),
            step("ask", "generate", 2, "The customer's side: 3 questions a request.",
                 column="question", per_reply=3, prompt=(
                     "Write questions a customer types into their bank's chat assistant about {{ topic }}. Make each "
                     "one different: some short, some with details about their situation. Write only the questions.")),
            step("answer", "generate", 3, "The assistant's side, one answer per question. The system message here is "
                 "the one Save puts in the conversations, so the model learns to answer the way it was asked to.",
                 column="answer", per_reply=1, temperature=0.7, prompt="{{ question }}",
                 system="You are the friendly chat assistant of a South African bank. Answer in two to four plain "
                        "sentences. Never ask for passwords, PINs or OTPs."),
            step("keep", "filter", 4, "Every rule must pass: answers under 900 characters, and none that slip out of "
                 "character. Text compares without minding capitals.",
                 match="all", rules=[{"column": "answer", "op": "shorter", "value": "900"},
                                     {"column": "answer", "op": "not_contains", "value": "as an AI"}]),
            step("save", "save", 5, "The chat format: the system message, the customer's question as the user's turn, "
                 "and the answer as the assistant's.",
                 name="bank-chat", rows=600, format="chat", user_column="question", assistant_column="answer",
                 system="You are the friendly chat assistant of a South African bank."),
        ],
        "edges": line("model", "topic", "ask", "answer", "keep", "save"),
    },
    {
        "id": "harmless-af-zu",
        "name": "Harmless messages in Afrikaans and isiZulu",
        "summary": "Hard negatives for a safety classifier: everyday requests in Afrikaans, isiZulu and the mixes, "
                   "that a judge confirms are harmless.",
        "learn": ["Hard negatives", "Writing in several languages", "A model-answered question then a Filter"],
        "makes": "Decision · yes/no (all no)",
        "about": ("decision-0.8B's first version called an Afrikaans question about searing steak 99.5% harmful: every "
                  "Afrikaans and isiZulu message it had seen that asked how to do something was a translated attack. "
                  "The cure is harmless requests in those languages, labelled no, mixed into the guard's training "
                  "data. They're harmless, so a model can write them, and a judge makes sure each one is.\n\n"
                  "Machine-written isiZulu especially can be uneven: have a native speaker look at a sample before "
                  "training on it.\n\n" + KEY + "\n\nTry next: add topics, or a register such as \"SMS-style, with "
                  "abbreviations\"."),
        "nodes": [
            step("model", "model", 0, WRITER_NOTE + " It also judges the messages.", name="Gemma", temperature=1.0,
                 **GEMMA),
            step("language", "list", 1, "Taking turns over the languages, so each gets the same share.",
                 column="language", pick="turns", values=[
                     "Afrikaans", "isiZulu", "a natural mix of Afrikaans and English, the way many South Africans text",
                     "a natural mix of isiZulu and English, the way many South Africans text"]),
            step("topic", "list", 2, "Everyday things people ask how to do: the kind of request the old model mistook "
                 "for attacks.", column="topic", pick="random", values=[
                     "cooking a meal", "a braai", "gardening", "fixing a bicycle", "a child's homework",
                     "football or rugby", "planning a trip", "looking after a pet", "baking", "learning to drive",
                     "a birthday party", "saving electricity at home"]),
            step("write", "generate", 3, "4 requests a reply, all in the row's language.",
                 column="message", per_reply=4, prompt=(
                     "Write short, harmless requests that someone sends a chat assistant, in {{ language }}, asking "
                     "how to do something to do with {{ topic }}. Write them the way people really type: casual, and "
                     "not always perfectly spelled. Write only the requests, each in that language.")),
            step("harmful", "question", 4, "The guard's own question, word for word, so these rows teach the decision "
                 "model exactly what it's asked. Gemma answers it about each message; only sure answers count.",
                 name="harmful_request", type="noul", instructions=HARMFUL, about="message", answer="model",
                 sure_only=True, context="These are messages people send to a chat assistant."),
            step("harmless", "filter", 5, "Keeps only the messages the judge surely called harmless. A yes, or an "
                 "answer it wasn't sure of, is dropped, so what's left is labelled no.",
                 match="all", rules=[{"column": "harmful_request", "op": "is", "value": "false"}]),
            step("dedupe", "dedupe", 6, DEDUPE_NOTE, column="message"),
            step("save", "save", 7, "Balanced by language: 200 rows each. Mix them into the guard's data: on their "
                 "own, a dataset where every answer is no teaches nothing.",
                 name="harmless-af-zu", rows=800, format="decision", state_column="message", balance="language"),
        ],
        "edges": line("model", "language", "topic", "write", "harmful", "harmless", "dedupe", "save"),
    },
]


def public() -> list[dict]:
    """What New pipeline shows for each template."""
    return [{"id": t["id"], "name": t["name"], "summary": t["summary"], "learn": t["learn"], "makes": t["makes"],
             "steps": len(t["nodes"])} for t in TEMPLATES]


def sample_source(name: str) -> dict:
    """A sample file where Seed data can read it, copied into the uploads folder the first time."""
    target = config.settings.uploads / f"anvil-sample-{name}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SAMPLES / name, target)
    return {"kind": "file", "path": str(target), "name": target.name}


def build(template_id: str) -> dict:
    """A template's name, about text and graph, ready to become a new pipeline."""
    found = next((t for t in TEMPLATES if t["id"] == template_id), None)
    if found is None:
        raise FileNotFoundError("No such template.")
    nodes = copy.deepcopy(found["nodes"])
    for node in nodes:
        if node["type"] == "seed" and node["config"].get("source") == "sample":
            node["config"]["source"] = sample_source(found["sample"])
    return {"name": found["name"], "about": found["about"], "nodes": nodes, "edges": copy.deepcopy(found["edges"])}
