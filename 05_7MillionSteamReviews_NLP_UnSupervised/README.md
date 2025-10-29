# Voice of the Player: Unsupervised Insight Mining on 7M Steam Reviews

### What this project does
I built an unsupervised pipeline to extract product feedback, pain signals, and delight signals from millions of real Steam game reviews — without any labeled data.

### Why this matters
Studios, publishers, and platform teams drown in qualitative feedback:
- "Servers are unplayable"
- "Cheaters ruin every match"
- "This story changed my life"
- "Predatory DLC pricing"

This project:
1. Finds those themes automatically.
2. Quantifies how angry / happy each theme is.
3. Tells you which themes are worth fixing first.

### How it works
1. Ingested ~7M Steam reviews.
2. Removed duplicates, bots, spammy low-effort text.
3. Built a high-signal working set of ~60K detailed reviews (helpful / long / unique).
4. Cleaned and lemmatized text (spaCy).
5. Ran two styles of unsupervised modeling:
   - **TF-IDF + NMF** to get interpretable topics with clear keywords.
   - **Sentence-BERT embeddings + K-Means, Agglomerative, and DBSCAN** to cluster reviews by semantic meaning.
6. Visualized clusters in PCA space.
7. For each cluster, extracted:
   - Top terms players use,
   - Example raw reviews,
   - Average recommendation score.
   
  
### 📂 Data
This project uses the [Steam Review Dataset (2017)](https://zenodo.org/records/1000885).
Download and place it in the `data/` folder before running.
Contains 6,417,106 rows x 5 columns.

### Key findings (example)
- **"Unplayable / Cheater / Performance Rage" Cluster (~6.6K reviews, avg score ≈ -0.11)**  
  Players use language like `lag`, `fps`, `crash`, `server`, `cheater`, `anti-cheat`, `refund`.  
  This is the highest-churn risk: "I want my money back."

- **"Price / Value / DLC Friction" Cluster (~7.4K reviews, avg score ≈ 0.31)**  
  Language around `worth`, `money`, `sale`, `expensive`, `dlc`.  
  Players say it's only worth buying on discount — clear monetization perception problem.

- **"Needs Polish / QoL" Cluster (~9.6K reviews, avg score ≈ 0.50)**  
  Tone is "I enjoy it BUT fix [content balance / bugs / progression / PvE/PvP issues]."  
  This is where live-ops and patch teams can win goodwill fast.

- **"Core Gameplay Experience" Cluster (~10.2K reviews, avg score ≈ 0.65)**  
  People talk about intense fights, horror moments, co-op sessions with friends, 50+ hour runs.  
  This is your engagement loop.

- **"Story / Immersion / Replayability" Cluster (~14.1K reviews, avg score ≈ 0.72)**  
  Language around `story`, `characters`, `voice acting`, `atmosphere`, `recommend`.  
  This is what actually sells — players literally say "I recommend this."

- **"Superfans / Evangelists" Cluster (~11.9K reviews, avg score ≈ 0.79)**  
  Pure hype: "masterpiece", "best ever", "buy this", "hundreds of hours".

### Who uses this?
- Product leads: prioritize what to fix (performance, cheaters).
- Monetization / pricing: understand value perception.
- Marketing: identify what players gush about (story, immersion, co-op moments).
- Community & trust/safety: escalate cheating/toxicity clusters for moderation.

### Tech stack
- pandas, numpy
- spaCy for lemmatization & cleanup
- scikit-learn (TF-IDF, NMF, K-Means, Agglomerative, PCA, Silhouette)
- sentence-transformers (all-MiniLM-L6-v2 for semantic embeddings)
- matplotlib / seaborn for visualization

### Why this stands out
- Completely unsupervised.
- Scales to millions of reviews.
- Produces *directly actionable* guidance:
  - "Fix this first."
  - "Market this harder."
  - "Here are 3 verbatim player quotes we can show leadership."

### Notebook
See `steam_reviews_voice_of_player.ipynb` for:
- Data cleaning
- Text normalization
- Topic modeling
- Embedding-based clustering
- PCA visualization
- Business interpretation
- Cluster export
