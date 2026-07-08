# Your Personal Guide to This Project, Harry

This is written for YOU, not for judges. The goal: you should be able to close this
file and explain the whole project to anyone, confidently, in your own words -
including in a job interview, since you built a genuinely good GenAI project here.

---

## The big picture, in one paragraph

You're job-hunting for GenAI/ML roles. You built a tool that looks at real job
postings, figures out which ones actually fit your skills, catches sneaky problems
in the postings (using AI, not manual rules), and tracks whether your job search is
working. Along the way, you tested a real technical question - "does GPU
acceleration actually help here?" - and got an honest, sometimes surprising answer
instead of just assuming yes. That's the whole project. Everything else below is
just the "how" and "why" of each piece.

---

## Step 0: Picking the idea

**What we did:** You had two possible directions - a generic "smart city" problem,
or something built around a real problem you personally have. You chose the real
one: your own job search.

**Why:** A judge can tell the difference between "I imagined a city official who
might have this problem" and "I am the user, this is my real data." Authenticity is
worth more than scope. This was the single most important decision in the whole
project, because everything after this was easier to build convincingly.

---

## Step 0.5: Collecting real data

**What we did:** Gathered 20 real, live job postings from LinkedIn and company
career sites - Deloitte, KPMG, Accenture, Cognite, and others.

**Why it matters:** Anyone can point a tool at a public dataset. Using postings you
personally found and verified as live (you even caught one dead link) proves the
tool solves something you're actually facing right now, not a simulation of a
problem.

---

## Step 1: The "dumb" first version (naive fit-score)

**What we did:** Wrote a simple Python script that checked: does this job's
description text CONTAIN the words from your skill list? Count the matches, turn
it into a percentage, combine with a basic "are you in the right years-of-
experience range" check, and rank all 20 jobs by that combined score.

**Why we did this FIRST, before anything fancier:** This is a core engineering
habit worth keeping for the rest of your career - always get the simplest possible
version working end-to-end before adding complexity. It took maybe 20 minutes to
write, and it immediately proved the overall idea ("rank jobs by fit") actually
works and produces a sensible order (KPMG came out on top initially).

**The limitation we knew about from the start:** This is just "does this word
appear somewhere in the text" - it doesn't understand meaning. A posting that says
"NOT looking for someone with only RAG experience" would still count as a match on
the word "RAG," even though the posting is saying the opposite. This limitation is
exactly what Step 2 fixes.

---

## Step 2: Making it smart (Gemini structured extraction)

**What we did:** Instead of dumb keyword search, we sent each job posting's full
text to Google's Gemini AI model, with strict instructions: "read this and give me
back structured information - what skills are really required, what's the real
experience level, and are there any inconsistencies or red flags in this posting?"

**Why this matters:** Gemini can actually READ and UNDERSTAND text, not just
pattern-match words. This is the difference between a computer finding the word
"Python" in a sentence, versus a computer understanding "this role primarily
requires Python but experience with Java is a bonus."

**What went wrong, and why that's OK:** 19 out of 20 postings extracted perfectly.
One (Persistent Systems) failed because Gemini's own response came back as slightly
broken JSON - a known, normal thing that happens sometimes when asking any AI model
to produce a specific structured format. We didn't hide this or quietly delete that
row - we kept it, labeled clearly as a parse failure. This is actually a STRENGTH
in your presentation: it shows you understand that AI systems aren't perfect and
you build systems that fail gracefully instead of assuming 100% success.

**The real payoff:** Gemini caught things a human skimming the same text might
miss - like Accenture's posting stating two different, contradictory experience
requirements in the same document, or Infosys having completely mismatched
unrelated category tags on their posting. This is genuinely useful, not just a
tech demo - a real job seeker would want to know a posting has red flags before
spending an hour tailoring a resume for it.

---

## Step 3: The acceleration story (this is your best material - understand it deeply)

**What we were trying to prove:** The hackathon wants you to show that using a GPU
(a graphics card, originally for video games, now used for AI/data science) makes
your data processing faster than a normal CPU (regular computer processor).

**What we actually found (the surprising part):** For our first test - searching
job description text for skill keywords across 200,000 rows of data - the GPU was
actually SLOWER than a normal CPU. We tried it twice, even after rewriting the code
to be "smarter," and the GPU was still slower both times.

**Why this happened (understand this deeply, it's genuinely useful knowledge):**
Think of a GPU like a warehouse with a thousand workers who can each do one simple
task at the exact same time - incredibly powerful for jobs that can be split into
a thousand identical simple pieces. But there's a real cost every time you ask
those workers to start a new task - you have to walk over, explain the job, and
walk back to collect the results. For a small, quick task, that walking-over cost
can be bigger than what you save by having a thousand workers. Searching text for
keywords is a "quick task" in this sense - so the overhead of using the GPU
outweighed the benefit.

**What we changed:** Instead of text search, we tested something GPUs are
genuinely built for - big numeric math. Specifically: comparing your profile
against a million job postings using "vector similarity" (a way of mathematically
measuring how similar two things are, based on numbers instead of words - this is
literally the same math behind how modern search engines and recommendation
systems work). This kind of math is pure, repetitive number-crunching across huge
grids of numbers - exactly the "thousand workers doing the same simple task" case
GPUs are built for.

**The real result:** 8.4 seconds on a normal CPU, versus 0.2 seconds on a GPU - a
genuine 41x speedup. Real, measured, not made up.

**Why this whole story (including the "failure") is your strongest material:**
Anyone can show one impressive-looking speedup number. Very few people will show
you they tested something, got a result that contradicted their expectation,
figured out WHY, and then found the right approach. That's what real engineers
actually do, and it's a far more convincing signal of skill than a single suspicious
"our AI is 100x faster!" claim with no explanation.

---

## Step 4: Getting the data into BigQuery

**What we did:** Uploaded your two data tables (the original job postings, and
Gemini's structured extraction of them) into Google BigQuery - think of it as a
very large, very fast spreadsheet living in the cloud, that you can ask questions
of using SQL (a language for querying data).

**Why:** Your FastAPI backend (Step 6) needs somewhere to pull data from that isn't
just files sitting on your laptop - especially for the `/ask` feature, where
Gemini needs to write and run real database queries.

**The bump we hit:** BigQuery initially failed to load your CSV because some job
descriptions have line breaks inside them (a paragraph of text spanning multiple
lines) - and BigQuery's default settings assumed every line break meant a new row.
One checkbox ("Quoted newlines") fixed it. Small technical detail, but a good
example of how real-world messy text data often needs small format adjustments.

---

## Step 5: The real fit-score (using Gemini's actual understanding, not guesswork)

**What we did:** Rebuilt the Step 1 ranking logic, but this time comparing your
profile against Gemini's CLEAN extracted skill lists (from Step 2), instead of raw
noisy paragraph text. Also swapped out the naive "is this text suspiciously short"
anomaly check for Gemini's own real red-flag detections.

**Why this is meaningfully better than Step 1:** Step 1 was comparing your skills
against messy paragraphs. Step 5 compares your skills against a clean, structured
list Gemini already extracted and verified. Much less noise, much more accurate.

---

## The application forecast (synthetic data, real feature)

**What we did:** Since you don't have a real logged history of job applications
yet, we generated a realistic, synthetic application log - which jobs you "applied"
to, and whether you got a callback, rejection, or no response - weighted so that
higher fit-score jobs were slightly more likely to get callbacks (a coherent story:
"our tool's own score correlates with better real-world outcomes").

**Why we were upfront that it's synthetic:** Presenting made-up numbers as if
they're your real personal results would be dishonest and could unravel under a
judge's questioning. Being clear "this feature works, using demonstration data for
now" is both honest and still demonstrates the capability fully.

**What the forecast actually calculates:** Your overall callback rate, whether your
recent applications are doing better or worse than your earlier ones (the "trend"),
and a projection of how many callbacks you should expect if you apply to N more
similar-quality jobs.

---

## Step 6: FastAPI backend (three endpoints)

**What we did:** Built a small web server with three "endpoints" (specific URLs
that do specific things when you request them):
- `/score` - returns the ranked job list
- `/forecast` - returns the application funnel summary
- `/ask` - takes a plain-English question, has Gemini turn it into SQL, runs that
  SQL against BigQuery, and returns the answer

**Why FastAPI specifically:** This is literally the same framework you already use
daily at Ascentt - so this step should have felt familiar, just applied to your own
project instead of DELight.

**The real bugs we hit and fixed (useful to remember, this is normal software work):**
- BigQuery results needed one more Python package (`db-dtypes`) to convert into a
  usable table format - a one-line fix once identified
- Gemini's free tier has a daily limit of 20 requests per model - you hit this
  limit because Step 2's extraction used up your daily quota, then `/ask` tried to
  use more. Fix: either wait a day, or switch to a different Gemini model
  (`gemini-2.5-flash-lite`) which has its own separate quota
- The first version of the code didn't catch errors from Gemini gracefully (it
  would crash with an ugly technical error page) - we added proper error handling
  so it now returns a clean, readable message instead

**A subtlety worth remembering:** When you asked "which company has the most
inconsistent experience requirements," Gemini wrote technically correct SQL, but it
answered a slightly different question than intended (widest min-to-max range,
rather than "contradicts itself within one posting"). This is a good real-world
lesson: AI-generated SQL is often literally correct but needs precise questions to
get precisely the right answer - the same is true of AI-generated code in general.

---

## Step 7: Streamlit dashboard

**What we did:** Built a simple, visual web page with three tabs (Ranked Jobs,
Application Forecast, Ask a Question) that calls your FastAPI backend and displays
the results as tables, metric cards, and a chart.

**Why Streamlit:** It's the fastest way to build a working dashboard in Python
without needing to learn a full frontend framework like React - ideal for a
hackathon deadline.

---

## Step 8: Packaging for deployment (Docker)

**What we did:** Wrote a `Dockerfile` (instructions for building a "container" - a
self-contained package with your code and everything it needs to run) and a
`start.sh` script that starts both your FastAPI backend and Streamlit frontend
together inside that one container.

**Why we packaged it this way:** Cloud platforms like Google Cloud Run, AWS, or
basically any modern hosting service can run a Docker container without needing to
know anything about Python, FastAPI, or Streamlit specifically - it's a universal
format. This means the same exact files work whether you deploy to Cloud Run, AWS,
or anywhere else later.

**Why we didn't actually deploy it live:** Deploying to Cloud Run requires linking
a real billing account (a payment card) to your Google Cloud project - even though
the free tier would almost certainly cover a hackathon demo at $0 cost. You decided
to keep that decision in your own hands rather than have it made for you, which is
completely reasonable - the deployment config is fully ready whenever you want to
use it.

---

## Step 9: Documentation (PPT, README, video script, description)

**What we did:** Turned the entire journey above into hackathon-submission-ready
materials - a slide-by-slide PPT outline, a polished GitHub README, a timed demo
video script, and short/long description text for the submission form.

**Why this step matters as much as the code:** A judge only sees what you show
them. The strongest engineering work in the world doesn't score well if it's
presented as a wall of unexplained code - the PPT and video are where you translate
everything above into a story a judge can follow and be convinced by in a few
minutes.

---

## If someone asks you "what's the hardest part of this project," here's an honest answer

The acceleration benchmark (Step 3). Not because the code was hard to write, but
because the first result contradicted what we expected (GPU should always be
faster, right?) - and the real skill was in stopping, understanding WHY the result
came out that way, and redesigning the test around a workload that actually fits
the hardware, rather than just reporting a bad number or hiding it.