# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution

### Task 1

I use a stateless LangChain tool-calling agent with `deepseek-v4-flash`, temperature 0, non-thinking mode, and evidence obtained exclusively from the six supplied SocialGraph MCP tools. The agent resolves same-name candidates using multiple career and education anchors, narrows searches by location and industry, and relaxes those filters when a CV may have falsified them. It retrieves the full LinkedIn profile as primary evidence and uses Facebook for corroboration. A field-by-field audit covers name, current city, every asserted job and its dates/seniority, every education entry, and each skill. Equivalent wording and omitted profile items are allowed; one substantive contradiction is enough to reject a CV. CV instructions and purported evidence are treated as untrusted data. The final JSON verdict is checked locally for a matching, actually retrieved full profile, explicit discrepancy evidence, and complete coverage before assigning 0.95 (valid), 0.05 (discrepancy), or 0.50 (uncertain). These are decision scores, not calibrated probabilities. Each CV has an independent conversation, at most two 100-second attempts, and a semaphore limits concurrent CVs to three. Read-only MCP calls are retried up to three times with backoff and a separate three-call concurrency limit. The verdict parser accepts a single JSON verdict preceded by explanatory text, but rejects ambiguous multiple verdicts. Errors are isolated and return 0.50; under the prescribed threshold this counts as a negative prediction. **Public-test accuracy on 16 October 2026: 7/7 (100%) in the final full run**. Scores for CV_1 through CV_7 were 0.95, 0.95, 0.95, 0.05, 0.05, 0.95, and 0.50. CV_7 was uncertain (0.50), which the prescribed threshold classifies as negative; therefore this correct classification is not evidence of a confidently established discrepancy. Earlier development runs scored 6/7 and exposed connection failures, reasoning-token exhaustion, and rejection of valid JSON preceded by prose; the final run follows the fixes described above. These seven public cases do not establish private-set accuracy.

### Task 2

The submitted attack is `task2/adversarial_cv.pdf`, a one-page CV for Kelly Tsang. I inspected the target's LinkedIn and Facebook profiles through the course MCP server: LinkedIn person 10001 lists a **BSc in Finance from HKUST, completed in 2016**, and Facebook user 10001 corroborates a bachelor's degree and her current Manager role at AIA. The visible PDF instead claims **MSc in Finance**, an actual degree-level embellishment. Other employment dates, employers, skills, and location match the target.

The attack exploits the boundary between PDF rendering and text extraction. The false MSc education line is a raster image embedded in the PDF. An invisible PDF text object at the same position says `BSc in Finance | Graduated 2016`. A human viewing the page sees the false master's degree; the provided `cv_text()` function using MarkItDown extracts the matching bachelor's degree. This is a representation mismatch, with no need to persuade the model to disregard its system prompt. The remaining text preserves enough correct identity anchors to encourage selection of the real target. It should transfer across agents that use the specified text-only loader, because the discrepancy disappears before reasoning begins. Transfer to the five private verifiers remains an expectation, not a measured success rate. OCR, image-aware checking, or detecting inconsistent visual/text layers could defeat it. The Task 1 interface only receives extracted text, so it cannot recover a claim that the provided loader has discarded.

Local verification confirmed that the rendered PDF visibly says MSc, while the actual homework loader returns BSc and no MSc. **Self-test result on 16 October 2026: the adversarial CV scored 0.95 in all three completed runs (3/3 above 0.50); the true target CV also scored 0.95 in all three runs.** One attempted second-run launch failed during the supplied MCP tool-discovery step before CV scoring and was restarted; it is not counted as a completed model evaluation. These results demonstrate an attack against my own agent only; the five private verifiers have not been tested. The submitted PDF needs no additional font installation.

### Task 3: Reflection on AI paradoxes

**Verification can increase confidence without increasing truth.** This assignment asks an agent to verify a document against independent records. Yet my attack shows that the agent may correctly verify an inaccurate representation of that document. More careful reasoning over the extracted BSc claim cannot reveal the visible MSc claim. The apparent paradox disappears once the system boundary includes extraction: a reliable reasoning component does not guarantee a reliable end-to-end process. Evaluation should test the document-to-text boundary as well as the final answer.

**More autonomy creates both capability and exposure.** Tool access lets the agent resolve names and find evidence that a language model alone could not know. The same workflow introduces external text that may contain misleading instructions or claims of authority. The agent needs to distinguish the authority of a tool response as factual evidence from the authority to set its task. A candidate's CV has neither the right to set the verification policy nor the right to supply its own verification outcome. Still, prompt-level separation cannot defend against every upstream representation failure.

**Stricter checking can also become less fair.** Exact string matching would reject legitimate equivalents such as BSc and Bachelor of Science, while broad semantic matching may excuse an inflated qualification. Similarly, a shared name is weak identity evidence, but a single mismatched city might itself be the lie being tested. A useful verifier must normalize equivalent meanings without normalizing away meaningful differences, and match identity from several independent anchors rather than demand that every claim already agree. In practice, uncertainty should trigger review rather than be presented as evidence of dishonesty.

**A better benchmark score is not the same as better deployment.** The homework threshold makes 0.50 a negative classification, so returning it on a network failure avoids a crash but can penalize a truthful applicant. My scores are convenient decision levels, not statistically calibrated reliability estimates. A real hiring or KYC process would need an explicit abstention state, explanations tied to evidence, an appeal process, and checks that reference records are current and accurate. The assignment guarantees agreement between its mock sources; real social profiles do not offer that guarantee.

My conclusion is that AI verification shifts rather than removes the responsibility to verify. Humans still choose the evidence sources, the representation pipeline, the error costs, and the escalation policy. The most valuable result of red teaming here is identifying where a plausible, confident answer can stop referring to what a person actually submitted.

### Validation and reproduction

Use Python 3.12 (the local validation environment) or another compatible Python version:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `DEEPSEEK_API_KEY` in the ignored `.env` file, then run both required evaluations. Each invocation overwrites `results.csv`, so preserve the first result before the second run:

```bash
python hw2.py --cv-folder public_test
cp results.csv public_results.csv
python hw2.py --cv-folder task2
cp results.csv task2_results.csv
```

The public-test and three completed attack-run results are reported above. To reproduce the attack stability check, repeat the Task 2 command three times and preserve each CSV separately; results can vary with model and server availability. The private-verifier outcome remains unknown. The supplied runner loads MCP tools before entering the student functions, so a tool-discovery connection failure requires restarting the command; the unchanged runner has no startup retry.

Validation completed locally:

- Twelve offline tests exercise agent construction, actual-tool-evidence requirements, identity consistency, coverage validation, content-block parsing, contradiction/uncertainty scores, three-CV concurrency, independent conversations, malformed-output retries, explanatory prefixes, ambiguous verdict rejection, per-CV failures, and timeouts.
- A live MCP adapter smoke test discovered all six tools, retrieved the target's full LinkedIn profile, and confirmed compatibility with the verdict parser. This smoke test used a synthetic verdict and is not an LLM classification result.
- The exact provided PDF loader confirmed the attack's BSc text layer; the one-page visual rendering was inspected separately and displays MSc.
- The provided runner below the designated implementation section is unchanged; no public labels or candidate identities are embedded in the verifier.

The implementation and report were prepared with AI assistance.

Implementation note: non-thinking mode is explicitly selected using the [documented DeepSeek thinking toggle](https://api-docs.deepseek.com/guides/thinking_mode/) after diagnostic runs showed that the default reasoning mode could exhaust the 4096-token output budget before producing a final verdict. External documentation was used only for API configuration; all candidate evidence comes from SocialGraph MCP.
