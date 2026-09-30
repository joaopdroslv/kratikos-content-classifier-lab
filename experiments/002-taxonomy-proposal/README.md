# 002 — taxonomy proposal (evidence for the team)

## Question

Where are the 12 current categories too coarse? Which **new master categories** does the corpus
support, and which **subcategories** should each master have?

This experiment trains no model. It produces a taxonomy for a team decision, curated in
[`src/lab/taxonomy_v2.py`](../../src/lab/taxonomy_v2.py); the proposal for the team is
[proposal.pt-BR.md](proposal.pt-BR.md).

## Method

Two parts: steps 1–3 find the master categories from vectors; steps 4–7 induce the
subcategories from text, then validate the curated result.

| Step | Script | Output (`data/002/`) |
|---|---|---|
| 1 | `01_topics.py` | `topics.parquet`, `centroids.npy`: every topic (45,970, covering all 85,586 ingested articles) with subject, scope, member count and `centroid` vector from Qdrant `topics` |
| 2 | `02_categorise.py` | `topics_categorised.parquet`: each topic under one of the 12 v1 categories by the 001 student, refit on all 3,075 labelled 001 articles |
| 3 | `03_clusters.py` | `clusters.md`: k-means inside each category, 115 clusters with share, scope mix and nearest subjects |
| 4 | `04_extra_sample.py` | `extra_sample.parquet`: 750 more articles for the small masters, picked by the student |
| 5 | `05_tag.py` | `tags.jsonl`: per article, a v2 master, a `main_tag` and an optional `aspect_tag`, both generic (no place, person, club, event) |
| 6 | `06_consolidate.py` | `subcategories.json`: per master, faceted subcategories proposed by an LLM from the tags, checked against rules, revised up to twice |
| 7 | `07_validate.py` | `validation.md`: the curated taxonomy applied to 1,400 unseen articles |
| 8 | `08_review_sheet.py` | `taxonomia_v2_validacao.xlsx`: one row per subcategory with its validation share, for the team to approve |

```bash
uv run python experiments/002-taxonomy-proposal/01_topics.py        # ... through 07_validate.py
```

### Part 1 — master categories (steps 1–3)

- The topic `centroid` is the mean of its members' article vectors, the same space 001 trained
  on. Most topics have one member, so topics stand in for the whole corpus.
- Shares are shares of **articles** (topics weighted by member count), summed over the clusters
  read as one candidate; they are estimates (student 83% accurate vs a human, clusters not pure).

### Part 2 — subcategories (steps 4–7)

The step-3 clusters could not give subcategories: the vectors split by **geography** (the Premier
League and the Brasileirão share no names) and by **single stories** (one soap opera, one
concert). So the subcategories are induced from text, in the TnT-LLM way (Wan et al., 2024): an
LLM writes generic tags per article, another consolidates them per master, and the result is
reviewed by hand.

Design decisions, each taken with the product owner during the experiment:

- **Multi-label subcategories, organised in facets.** A master has a **primary** facet split
  along one axis, and may have a **cross-cutting** facet of aspects that recur across it. An
  article can take several subcategories from either facet: "futebol" + "transferências". The
  first consolidation (one flat list per master) mixed axes, "futebol" next to "transferências",
  and its boundaries overlapped.
- **No geography and no named entities** in any subcategory. The reader's location is a separate
  filter (topics already carry a `scope`).
- **Sports by modality; politics by subject**, like every other master (the institution axis was
  considered and rejected).
- **`Loterias` and `Estilo de Vida` are masters**; lotteries have no subcategories.

Lessons from the runs, kept for the next taxonomy work:

- A free "1 to 3 tags" list came back with 3 almost every time; a required `main_tag` plus a
  nullable `aspect_tag` works, but only with `gpt-4.1` (`gpt-4.1-mini` filled the aspect on 39 of
  40 trial articles, with details such as "gols" or "superação").
- Asking one call to propose subcategories AND map hundreds of tag numbers made `gpt-4.1` loop
  until the output limit. Step 6 proposes in one call and assigns each tag in its own call,
  through an enum of the proposed slugs.
- A minimum share per subcategory stops story-sized fragments, but on an enumerable axis it
  backfires: every sport but football is under 4% of sports, so the rule merged them into
  "individual sports". Size is a hard rule only at the bottom and never for modalities.
- Automated revisions got the taxonomy most of the way; the rest were editorial decisions
  (which side of a boundary a subject falls on) that no revision round settles.

## Re-running

```
01 ─► 02 ─► 03                                   (read by a person → Result below)
       └─► 04 ─► 05 ─► 06 ─► [hand curation] ─► src/lab/taxonomy_v2.py ─► 07 ─► 08
```

| Step | Behaviour |
|---|---|
| `01_topics` | **Reads the live database and Qdrant.** Re-run later, it sees a different corpus; `sampled_at` records when |
| `02_categorise`, `03_clusters` | Deterministic (seeded). Changing a parameter of 03 renumbers the clusters, and the cluster ids quoted under Result go stale |
| `04_extra_sample` | **Reads the live database**; seeded, same draw only while the database is unchanged. Depends on 02 |
| `05_tag` | **Resumable, costs API.** Skips ids already in `tags.jsonl`; a new prompt needs a new `PROMPT_VERSION` and a fresh file |
| `06_consolidate` | **Costs API, not deterministic**: `gpt-4.1` may propose slightly different subcategories. Tag assignments are resumable per proposal. Overwrites `subcategories.json` and `consolidation_log.jsonl` |
| `07_validate` | Draws its sample once (`validation_sample.parquet`; delete it to redraw). **Resumable, costs API**; answers go to a file named by the prompt's hash, so a changed taxonomy starts a new file |
| `08_review_sheet` | Deterministic, no API. Reads step 7's 2.0 answers and carries them to the current taxonomy through `REMAP`; fails if a 2.0 subcategory has no mapping. **Overwrites the xlsx**: move a filled copy out first |

Re-running 06 does **not** change 07: 07 reads the curated `taxonomy_v2.py`, which only a person
edits.

## Result (2026-09-28, dev)

### Master categories (steps 1–3)

**v1 category shares** (student, all articles): sports 27.1%, politics 22.5%, economy 13.9%,
public_safety 10.3%, culture 6.7%, technology 6.5%, health 5.1%, environment 3.9%, human_rights
2.0%, transport 1.1%, education 0.7%, housing 0.2%.

**A missing master shows up as a large, coherent cluster under the wrong parent**, not as low
confidence (only 1.4% of articles score < 0.5).

| Candidate | Clusters | Share of all articles |
|---|---|---|
| **Geopolitics** (relations between states, wars, diplomacy) | politics:1,2,3,8,9,11,14,17,21,22 | **8.7%** (39% of politics) |
| + the same wars filed elsewhere | public_safety:9, human_rights:2 | +1.2% |
| Foreign domestic politics (US, UK, Germany, Portugal/EU, Africa) | politics:4,6,13,15,18,19 | 5.7% (25% of politics) |
| Brazilian politics | politics:0,5,7,10,12,16,20 | 8.1% (36% of politics) |
| **Entertainment / celebrities** | culture:3,4,5,9, sports:13 | **4.0%** |
| **Lotteries** | economy:8 | **1.3%** (more than education + housing) |
| Accidents and disasters | public_safety:2,4,8 | 1.9% |
| Weather | environment:1,2 | 1.3% |
| Automotive (split between economy and transport) | economy:2, transport:2 | 1.2% |
| Lifestyle (recipes, horoscope, tourism) | culture:0 | 1.1% |

Adopted: `Geopolítica`, `Entretenimento`, `Loterias`, and later `Estilo de Vida` (step 5 also
returned travel and tourism as the most frequent missing master). Weather, accidents and
automotive became subcategories.

### Subcategories (steps 4–6)

3,825 articles tagged (3,075 from 001 + 750 extra), every master above 100 articles but
lotteries. The v2 master mix of the tagged sample confirms part 1: geopolitics (357) is larger
than politics (319).

The faceted consolidation (`subcategories.json`) covered 89–100% of each master's articles with
a primary subcategory, at 1.2–1.5 subcategories per article. Health, transport, human rights,
technology and housing came out close to final. It failed where an editorial call was needed:

| Master | Consolidation output | Curated (`taxonomy_v2.py`) | Why |
|---|---|---|---|
| Sports | Futebol 75%, "esportes individuais", "coletivos", "multiesportivos e temas gerais" | 13 modalities + cross-cutting transfers, management, **women's sport** | The share rule merged every sport but football; readers follow a modality |
| Politics | 7 subcategories on a "process or institution" axis, incl. "sistema político e direitos cívicos" | 7 subjects: elections, corruption, democracy and institutions, government, bills and reforms, parties, polarisation | Institution axis rejected; vague umbrellas cut |
| Geopolitics | Diplomacy 48%, armed conflicts 47% | Wars; peace talks; diplomacy; defence and weapons; sanctions; international bodies; migration | Two giants split along the same axis |
| Culture | Incl. "agenda e estilo cultural" (recipes, horoscope, tourism) and cinema | 5 disciplines + cultural policy | Lifestyle became a master; film goes to entertainment |
| Entertainment | Incl. "música e shows" | Celebrities; TV, soaps and reality; film and series | Music belongs to culture |
| Education | "Política, gestão e temas transversais" 41% as a stage | 4 stages + cross-cutting exams, teachers, policy | Policy is an aspect, not a stage |
| Transport | Cross-cutting "segurança e acidentes" | Accidents removed | Accidents belong to public_safety |
| Environment | Climate change, weather and natural disasters overlapping | Boundaries written: long-term climate / weather / disasters with damage | Same subject in three places |
| Public safety | Violence against women as cross-cutting, coverage 89% | Violence against women and sexual violence as primary types | They are types of occurrence |

Curated total: **16 masters, 116 subcategories** (taxonomy 2.0). Boundaries between masters are
written in the `taxonomy_v2.py` docstring.

### Validation (step 7)

Run on 2026-09-29 with `gpt-4.1` on taxonomy 2.0: 1,390 of 1,394 articles answered (the rest
hit rate limits). Full report in `data/002/validation.md`.

- **Coverage holds.** 92–100% of each master's articles got a subcategory of their primary master
  (lotteries 0% by design: no subcategories), "none fits" at 0–11%, and subcategories chosen
  outside the article's masters at 0–1%: the masters do not bleed into each other.
- **Uniform-stratum mix:** sports 26%, politics 13%, geopolitics 12%, economy 11%, public_safety
  9%, technology 7%, environment 5%, health 5%; every other master under 4%.
- **Empty or tiny subcategories:** basketball and volleyball (0%, likely the UK-leaning dev feeds),
  deforestation, teachers and homelessness (0%, on masters of 28–68 articles), and a few at 3–4%.
- **Blurriest same-facet pairs** (co-occurring in 6–7 articles): elections + parties, diplomacy +
  international bodies, diplomacy + wars, financial markets + macroeconomy.

### Manual review → taxonomy 2.1 (2026-09-29)

Decided with the product owner after reading the validation, without a new run (step 8 carries
the 2.0 answers over by merging subcategories):

| Master | 2.0 | 2.1 | Why |
|---|---|---|---|
| Human rights | Primary facet by protected group: racial/indigenous/religious, LGBTQIA+, women | By type of issue: one `Discriminação e igualdade` for every group; `Crianças e grupos vulneráveis`, `Ativismo` renamed shorter | No group gets a theme of its own; coverage kept (the umbrella holds 61% of the master) |
| Culture | 5 disciplines | + `Religião` (faith and institutions) | Religion news had no home; religious freedom stays in human rights |
| Housing | Real estate, rent, social housing, homelessness separate | `Mercado imobiliário e aluguel` + `Acesso à moradia` (tenants' rights, programmes, evictions, homelessness) | A small master; homelessness at 0% |
| Environment | `Clima e previsão do tempo` | `Tempo e previsão` | "Clima" meant both climate and weather; the subcategories stay apart: climate policy is debated, forecasts are not |
| Health | `SUS e sistema de saúde` | `Sistemas e serviços de saúde` (SUS named in the definition) | Geography is not a category axis |

Total: **16 masters, 113 subcategories**. `religion` has no validation share yet, and
`housing_access` counts every 2.0 `rent` article, including rents as a market.

**Caveat:** dev's sources lean on UK and Portuguese feeds (cricket, rugby, Labour). Subcategory
shares may differ in production; `Críquete` in particular may not earn its place there.

**Conclusion so far:** the corpus supports four new masters (`Geopolítica`, `Entretenimento`,
`Loterias`, `Estilo de Vida`) and a faceted, multi-label, geography-free subcategory layer.
On unseen articles the curated subcategories hold (step 7); 2.1 awaits the team's review
(step 8's sheet).

**Update (2026-09-30):** the team validated 2.1. Experiment 004 then swept the whole corpus for
missing subcategories and added one, `economy.cryptocurrencies`: taxonomy **2.2, 114
subcategories**.
