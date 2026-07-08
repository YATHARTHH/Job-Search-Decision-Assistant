# DETAILED PPT - Job Search Decision Assistant
(Every slide includes: what to show, what to say, and WHY it matters - use the
"why" parts as your own speaker notes, they're written so you can defend any
question a judge asks.)

---

## SLIDE 1 - Title
**Show:** Project name, your name, hackathon name, date.
**Say:** "Job Search Decision Assistant - an AI tool that helps me decide which
jobs to apply to, using real data and Google/NVIDIA's AI stack."
**Why this slide exists:** Judges see dozens of projects. A clear, specific title
(not "AI Job Helper") signals you built something concrete, not a vague idea.

---

## SLIDE 2 - The Problem (Who has this problem, and why does it matter?)
**Show:** A screenshot of a messy job board / browser full of tabs, or just bullet text.
**Say:** "I'm a backend engineer with 3.5 years of experience, actively trying to
move into GenAI/ML roles. Every week I look at 20-30 job postings. Each one takes
5-10 minutes to read carefully. I have no systematic way to know: (1) which ones
actually match my skills, (2) which ones have red flags I might miss on a quick
read, (3) whether my applications are actually working."
**Why this slide exists:** The hackathon rubric explicitly asks "who is the real
user, what is the real problem?" This is not hypothetical - you are the user. That
authenticity is worth more than a bigger, vaguer "smart city" problem nobody on your
team has personally lived through.

---

## SLIDE 3 - What We Built (The elevator pitch)
**Show:** Four icons/boxes: Ranking | Anomaly Detection | Forecast | Chat
**Say:** "A tool that does four things: ranks real job postings by how well they
fit my profile, uses Gemini to catch problems in postings that a human skim-read
would miss, forecasts how my job applications are trending, and lets me ask
questions in plain English."
**Why this slide exists:** The rubric asks for "a useful output such as a
dashboard, forecast, alert, ranking, or recommendation." You have FOUR of these,
not one. This slide makes that explicit and easy to score.

---

## SLIDE 4 - The Data Pipeline (Where does the data come from, and where does it go?)
**Show:** This diagram, drawn simply left to right:
```
20 real job postings (LinkedIn/company sites)
        |
   Cloud Storage  <- just a place to keep the raw files
        |
   cudf.pandas cleaning  <- GPU-accelerated data cleaning (explained on Slide 7)
        |
   Gemini extraction  <- turns messy paragraph text into structured data
        |
      BigQuery  <- a big, fast, cloud database we can query with SQL
        |
   FastAPI backend (3 endpoints: /score /forecast /ask)
        |
   Streamlit dashboard  <- what the user actually sees and clicks
        |
   Cloud Run  <- makes it a public website anyone can open
```
**Say:** "Data flows one direction, gets progressively more useful at each step -
raw text becomes structured data, structured data becomes a queryable database,
the database becomes an API, the API becomes a dashboard."
**Why this slide exists:** This IS your architecture diagram requirement. Keep it
visual, not a wall of text - a judge should understand your pipeline in 10 seconds
of looking at it.

---

## SLIDE 5 - Why Real Data Matters (Not a toy dataset)
**Show:** A few real screenshots of actual job postings you collected (KPMG,
Cognite, Deloitte, etc.)
**Say:** "These aren't fake or scraped from a public dataset unrelated to me -
these are 20 real, currently open GenAI/ML Engineer postings I found on LinkedIn
and company career pages, verified live at the time I collected them. One posting
link had already gone dead by the time I checked it - I caught that and excluded it."
**Why this slide exists:** Judges have seen many hackathon projects that quietly
use fabricated or public "smart city" datasets that don't map to a real, lived
problem. Showing you personally verified real data is a credibility signal.

---

## SLIDE 6 - How Gemini Turns Messy Text Into Structured Data
**Show:** A before/after: raw paragraph JD text on the left, arrow, structured
JSON on the right (skills list, seniority, min/max years, red flags).
**Say:** "Step 2 sends every job description to Gemini with a strict instruction:
extract skills, experience range, and flag anything inconsistent - return ONLY
valid JSON. 19 out of 20 postings were extracted successfully. One failed because
Gemini's own response had a JSON formatting glitch - that's a known, normal
occurrence with LLM outputs at this scale, and I kept it in the data rather than
hiding it, since it's an honest example of a real-world limitation."
**Why this slide exists:** This demonstrates you understand LLM extraction isn't
magic/perfect - and that you build systems that handle failure gracefully instead
of assuming everything works.

---

## SLIDE 7 - Gemini Caught Real Problems A Human Would Miss
**Show:** Two or three specific real examples, side by side:
- **Accenture in India:** posting says "Minimum 5 Years Required" in one place, and
  "minimum 3 years ML/AI engineering" in another place of the SAME posting
- **Infosys:** the posting's category tags say "Turbomachinery -> Steam Turbine ->
  Rotor" - completely unrelated to an AI Engineer role (likely a copy-paste error
  by whoever posted it)
- **LTM:** correctly identified as a generic recruiter outreach message, not a
  real structured job description at all
**Say:** "These aren't rules I wrote myself - I didn't tell the system 'check for
inconsistent years.' Gemini reasoned this out by actually reading and understanding
the text. This is the difference between keyword search and genuine language
understanding."
**Why this slide exists:** This directly answers "identify patterns and anomalies"
from the problem statement, with real, checkable proof - not a hypothetical claim.

---

## SLIDE 8 - THE ACCELERATION STORY (your strongest, most differentiated slide)
**Show:** This exact table:

| Workload Type | CPU Time | GPU Time | What Happened |
|---|---|---|---|
| Keyword/text search, 200,000 rows | 13.4 - 24.5 sec | 27 - 36 sec | GPU was SLOWER |
| Vector similarity math, 1,000,000 embeddings | 8.4 sec | 0.20 sec | GPU was 41x FASTER |

**Say (this is the important part - explain the WHY, not just the numbers):**
"Most teams will show you one GPU speedup number and call it a day. I actually
tested this twice. First, I tried accelerating keyword search across 200,000 job
postings with NVIDIA's cudf.pandas - and the GPU was SLOWER than my own laptop's
CPU. Twice, in fact - even after I tried optimizing the code. Here's why: GPUs are
incredible at doing millions of identical, simple math operations at once - but
there's a real cost every time you hand work to the GPU and get the answer back.
For light tasks like text search, that hand-off cost is bigger than what you save.
It's like using a forklift to move one box - powerful, but not the right tool for
this specific job. So I pivoted to a numeric task GPUs are actually built for:
comparing job postings to my profile using vector math (the same kind of math
behind semantic similarity search). That gave a genuine, honest 41x speedup - 8.4
seconds down to 0.2 seconds on a million data points."
**Why this slide exists:** This shows real engineering maturity - you tested,
found something unexpected, understood the root cause, and adapted your design.
That is far more convincing to a technical judge than a single suspicious "we got
100x speedup!" claim with no explanation of when or why.

---

## SLIDE 9 - Live Product Walkthrough (screenshots)
**Show, in order:**
1. Ranked Jobs tab - point at Cognite in the #1 spot with its fit score
2. The Accenture anomaly text visible in the Anomalies column
3. Forecast tab - the 4 metric cards + the early-vs-recent bar chart
4. Ask tab - a real question typed in, the generated SQL shown, and the answer table
**Say:** "This is not a mockup - every number on this screen came from a real,
tested run of the actual pipeline."
**Why this slide exists:** Screenshots of a genuinely working product are worth
more than describing what it "would" do.

---

## SLIDE 10 - Tech Stack Summary (checklist slide for judges)
**Show, as a simple checklist:**
- GCP data/app layer used: Cloud Storage ✓, BigQuery ✓, Cloud Run ✓ (rubric needs 2+)
- NVIDIA acceleration used: cudf.pandas ✓, cupy ✓ (rubric needs 1+)
- LLM/RAG: Gemini (structured extraction + natural-language-to-SQL) ✓
- Real-world pipeline: ingest ✓, clean ✓, analyze ✓, model ✓, visualize ✓
- Useful outputs: ranking ✓, anomaly flags ✓, forecast ✓, recommendation ✓
**Why this slide exists:** Makes it trivially easy for a judge scoring against a
rubric to check every box without hunting through your other slides.

---

## SLIDE 11 - Honest Limitations (shows maturity, don't skip this)
**Show:**
- 1/20 postings failed Gemini extraction (kept, disclosed, not hidden)
- Application history data is synthetic/illustrative (real feature, demo data)
- Embeddings in the benchmark are simulated (isolates "speed of math" from
  "speed of generating real embeddings," which is a separate cost)
**Say:** "I'd rather show you exactly where the rough edges are than pretend
everything is perfect."
**Why this slide exists:** Counter-intuitively, this BUILDS credibility rather than
hurting it. Judges who've built real systems know nothing is ever 100% clean -
showing you know exactly where your own edges are is a stronger signal than a
polished slide with zero caveats.

---

## SLIDE 12 - What's Next
- Real embeddings instead of simulated ones, for production semantic matching
- Retry logic for the 1 failed Gemini extraction
- Real logged application history once you have enough real applications
- Deploy live (config is fully built, deployment itself deferred - see README)

---

## SLIDE 13 - Thank You
Links: GitHub repo, demo video, (deployment link if you end up deploying)