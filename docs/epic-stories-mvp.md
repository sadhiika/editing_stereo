# StereoWipe MVP Epic Stories: Research-Focused Implementation

## Critical Review of Original Plan

### Reality Check
The original epic-stories.md assumes resources typical of a well-funded startup:
- Multiple full-time developers
- Significant infrastructure budget
- 3+ year timeline
- Enterprise customers

For a research project, we need to be brutally honest about constraints:
- 1-2 part-time contributors
- Minimal infrastructure budget
- Need publishable results in 6-12 months
- Primary users are researchers, not enterprises

### 80-20 Analysis
**20% of features that deliver 80% of research value:**
1. Working LLM judges (currently only mocks exist)
2. Basic human validation data collection
3. Simple preference collection mechanism
4. Reproducible evaluation pipeline
5. Published datasets with paper

**80% of features that add complexity without research value:**
- Real-time anything (WebSockets, live updates)
- User accounts and authentication
- Multiple database systems
- Enterprise features
- Fancy UI/UX
- Complex analytics dashboards

---

## Research-Focused MVP Epics

### Epic 0: Technical Foundation (Prerequisites)
**Goal:** Get the basics working before anything else

**Minimal Viable Tasks:**
1. Implement real OpenAI judge (1 week)
2. Add SQLite for basic data storage (3 days)
3. Fix the broken upload endpoint (1 day)
4. Add basic progress logging to CLI (2 days)

**Deliverables:**
- Working LLM judge with actual API calls
- Simple data persistence
- Functional web viewer

---

### MVP Epic 1: Core Research Validation (Month 1-2)
**Goal:** Validate LLM-as-Judge methodology with minimal complexity

**Simplified User Stories:**
- As a researcher, I need to collect human ratings to compare with LLM judges
- As a researcher, I need statistically significant comparison data

**MVP Implementation:**
1. **Simple Annotation Tool:**
   - Basic Flask form for human annotations
   - SQLite storage for responses
   - CSV export for analysis
   - No user accounts - just session IDs

2. **Comparison Study:**
   - 100 prompts × 5 model responses = 500 evaluations
   - Get 3 human ratings per evaluation (volunteers)
   - Run same through LLM judge
   - Calculate inter-rater reliability

**Deliverables:**
- Simple web form: `/annotate`
- SQLite schema: `annotations.db`
- Analysis notebook: `human_vs_llm_analysis.ipynb`
- First research finding on LLM judge reliability

---

### MVP Epic 2: Preference Arena Prototype (Month 3-4)
**Goal:** Test preference-based evaluation with minimal infrastructure

**Simplified Implementation:**
1. **Dead Simple Arena:**
   - Static HTML page with two model outputs
   - Submit preference to SQLite database
   - No real-time updates - just batch processing
   - Model names revealed after voting

2. **Basic Leaderboard:**
   - Daily batch job calculates Elo scores
   - Static HTML page updated once per day
   - Simple table, no fancy visualizations

**Technical Shortcuts:**
- Use GitHub Pages for static hosting
- SQLite database file in repo (for small scale)
- Python script for daily Elo calculation
- No WebSockets, no React, no complexity

**Deliverables:**
- `/arena` page with basic A/B comparison
- `calculate_elo.py` batch script
- Static leaderboard page

---

### MVP Epic 3: Dataset & Reproducibility (Month 5-6)
**Goal:** Create research artifacts others can use

**Simplified Implementation:**
1. **Dataset Creation:**
   - Curate 200 high-quality prompts
   - Focus on 5 most important bias categories
   - Simple JSON format, version controlled

2. **Evaluation Package:**
   - Pip-installable package
   - Clear documentation
   - Jupyter notebook examples
   - Reproducible results

**Deliverables:**
- `stereowipe-data` v1.0 on HuggingFace
- `pip install stereowipe`
- Research paper draft
- Reproducibility package

---

## What We're NOT Building (And Why)

### Cut from Phase 1:
- ❌ **Multi-judge ensemble** → Just use GPT-4 for now
- ❌ **Custom judge prompts** → One good default is enough
- ❌ **Real-time progress** → Batch processing is fine
- ❌ **Web-based upload** → CLI is sufficient for researchers

### Cut from Phase 2:
- ❌ **User accounts** → Anonymous sessions work fine
- ❌ **Real-time leaderboard** → Daily updates are enough
- ❌ **WebSocket updates** → Page refresh is acceptable
- ❌ **Prompt-to-leaderboard** → Cool but not essential

### Cut from Phase 3:
- ❌ **Multi-domain** → Focus on stereotyping only
- ❌ **Enterprise SDK** → Open source tools are enough
- ❌ **Complex analytics** → Jupyter notebooks work fine
- ❌ **Collaboration features** → GitHub is sufficient

---

## Realistic Timeline

### Month 1: Foundation
- Week 1-2: Implement real LLM judge
- Week 3: Add SQLite, fix web viewer
- Week 4: Create annotation tool

### Month 2: Data Collection
- Week 1-2: Recruit volunteer annotators
- Week 3-4: Collect human validation data

### Month 3: Analysis
- Week 1-2: Statistical analysis of human vs LLM
- Week 3-4: Write up initial findings

### Month 4: Arena Prototype
- Week 1-2: Build simple preference collection
- Week 3-4: Implement Elo calculation

### Month 5: Dataset
- Week 1-2: Curate prompt dataset
- Week 3-4: Package for release

### Month 6: Paper
- Week 1-4: Write research paper

---

## Success Metrics (Revised)

### Research Success:
- ✓ Published paper on LLM-as-Judge reliability
- ✓ Open dataset with 200+ validated prompts
- ✓ 1000+ human annotations for comparison
- ✓ Working prototype others can use

### What We're NOT Measuring:
- ❌ Daily active users
- ❌ API response times
- ❌ Enterprise adoption
- ❌ Revenue

---

## Resource Requirements

### Human Resources:
- 1 graduate student (20 hrs/week)
- 1 advisor (2 hrs/week)
- 10-20 volunteer annotators

### Technical Resources:
- $100/month OpenAI API budget
- Free GitHub Pages hosting
- Local SQLite (no cloud database)
- Existing university compute for analysis

### Total Budget:
- ~$1000 for API costs
- ~$500 for volunteer incentives
- $0 for infrastructure

---

## Key Insights from 80-20 Analysis

1. **Static > Dynamic**: For research, daily updates beat real-time
2. **Simple > Scalable**: SQLite beats PostgreSQL for <10k records
3. **Notebooks > Dashboards**: Researchers prefer reproducible analysis
4. **Anonymous > Authenticated**: Reduces complexity dramatically
5. **Batch > Streaming**: Much easier to debug and analyze

This MVP focuses on answering the core research questions with minimal complexity. Every feature directly contributes to publishable results.