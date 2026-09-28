# 002 — taxonomy proposal (evidence for the team)

## Question

Where are the 12 current categories too coarse? Which **new master categories** does the corpus
support (e.g. `Geopolítica` out of `Política`), and which **subcategories** carry real volume under
each master?

This experiment trains nothing and labels nothing. It produces evidence for a team decision; the
proposal for the team is [proposal.pt-BR.md](proposal.pt-BR.md).

## Method

| Step | Script | Output (`data/002/`) |
|---|---|---|
| 1 | `01_topics.py` | `topics.parquet`, `centroids.npy`: every topic (45,970, covering all 85,586 ingested articles) with subject, scope, member count and `centroid` vector from Qdrant `topics` |
| 2 | `02_categorise.py` | `topics_categorised.parquet`: each topic under one of the 12 categories by the 001 student, refit on all 3,075 labelled 001 articles |
| 3 | `03_clusters.py` | `topics_clustered.parquet`, `clusters.md`: k-means inside each category (PCA 256, weighted by member count, k = topics/400 clipped to 3–30), 115 clusters with share, scope mix and the subjects nearest each centre |

- The topic `centroid` is the mean of its members' article vectors, the same space 001 trained on;
  most topics have one member, so topics stand in for the whole corpus with a readable subject.
- Shares are shares of **articles** (topics weighted by member count).
- Candidate groupings were read from `clusters.md` and summed by cluster id; the sums are
  estimates (the student is 83% accurate vs a human, and k-means clusters are not pure).

```bash
uv run python experiments/002-taxonomy-proposal/01_topics.py
uv run python experiments/002-taxonomy-proposal/02_categorise.py
uv run python experiments/002-taxonomy-proposal/03_clusters.py
```

## Result (2026-09-28, dev)

**Category shares** (student, all articles) match the 001 teacher's uniform sample: sports 27.1%,
politics 22.5%, economy 13.9%, public_safety 10.3%, culture 6.7%, technology 6.5%, health 5.1%,
environment 3.9%, human_rights 2.0%, transport 1.1%, education 0.7%, housing 0.2%.

**A missing master does not show up as low confidence** (only 1.4% of articles score < 0.5): the
student always files an item somewhere. It shows up as a large, coherent cluster that does not
belong under its parent.

| Candidate | Clusters | Share of all articles |
|---|---|---|
| **Geopolitics** (relations between states, wars, diplomacy) | politics:1,2,3,8,9,11,14,17,21,22 | **8.7%** (39% of politics) |
| + the same wars filed elsewhere | public_safety:9, human_rights:2 | +1.2% |
| Foreign domestic politics (US, UK, Germany, Portugal/EU, Africa) | politics:4,6,13,15,18,19 | 5.7% (25% of politics) |
| Brazilian politics | politics:0,5,7,10,12,16,20 | 8.1% (36% of politics) |
| **Entertainment / celebrities** (famosos, novelas, streaming, athletes' love lives) | culture:3,4,5,9, sports:13 | **4.0%** |
| **Lotteries** | economy:8 | **1.3%** (more than education + housing) |
| Accidents and disasters | public_safety:2,4,8 | 1.9% |
| Weather | environment:1,2 | 1.3% |
| Automotive (split between economy and transport) | economy:2, transport:2 | 1.2% |
| Lifestyle (recipes, horoscope, tourism) | culture:0 | 1.1% |
| Games | technology:5 | 0.6% |

- **Geopolitics is real and large**: at ~10% it would be the 4th-largest master, ahead of culture.
  Today the same wars are scattered across politics, public_safety and human_rights.
- **Geography is a separate axis.** Much of what looks like a split (foreign vs Brazilian politics,
  European vs Brazilian football) is geography, not subject. Topics already carry a `scope`; the
  proposal keeps geography out of the taxonomy.
- **Housing (0.2%) and education (0.7%) are tiny**, and part of transport is urban works.
- **Caveat:** dev's sources lean on UK/Portuguese feeds (cricket, Welsh rugby, Labour leadership).
  Production may weigh subcategories differently; masters are less sensitive to that.

**Conclusion:** the corpus supports at least two new masters (`Geopolítica`, `Entretenimento`) and a
decision on lotteries; subcategory candidates per master are in the proposal.
