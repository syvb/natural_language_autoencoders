# Gate G — example data points (Stage 0)

What the **AV verbalizes about h₂₂** (the layer-22 "after" state), which becomes
the `text` arm fed to the h₂₀-conditioned critic. The critic also sees h₂₀
(injected) and must reconstruct h₂₂; so the *useful* part of each description
is whatever it says about h₂₂ that h₂₀ doesn't already determine.

All positions below are **held-out documents** (FineWeb, gate_c slice). The AV
saw only the single h₂₂ vector — never the surrounding text. The "CTX" shown is
the real left-context (for the reader's benefit only) and the `token@pos` is the
token whose residual stream we read.

Each description has the AV's characteristic three-part shape: **register/format**
→ **local content** → **next-token expectation**. The third part is notable —
h₂₂ is two layers deeper than h₂₀ and encodes more of the model's forward
prediction, which is exactly the kind of content we hope survives the h₂₀
subtraction.

---

### 1 · classical-music catalog  (`token@pos = "ski"`, in "Wieniaw·ski")
**CTX:** …Concerto for Two Violins in D Minor (BWV 1043) — reproduced from an authoritative edition. / Caprices and Etudes for Solo Violin by Nicolò Paganini, Henryk Wieniaw**ski**

**AV(h₂₂):** *Classic reference book listing format with authoritative music publishing imprint, presenting works by great composers in hardcover editions. The second title introduces a catalog of composers with "100 sensational virtuoso pieces," suggesting a similar catalog of solo piano works by another virtuoso…*

---

### 2 · Unix package-manager mailing list  (`token@pos = " packages"`)
**CTX:** …List: tech-pkg … How to update all installed pkgs using binary packages? So, if my installed **packages**

**AV(h₂₂):** *Technical Unix/Perl mailing list post asking about `pkg_add -u` behavior, implying a Unix package manager context with versioning and dependency questions. The phrase "Say I have my installed packages" sets up an expected scenario describing current package versions or desired upgrade list, likely continuing with "are version X.Y.Z" or "include packages that have changed."*

---

### 3 · small-cap stock forum  (`token@pos = "2"`, in "$2")
**CTX:** …This is a short sellers manipulation, I wouldn't be surprised if the stock recovers at least $**2**

**AV(h₂₂):** *Stock market/technical analysis post tone with investor sentiment discussion around a small cap stock company, implying speculative trading context. The phrase "…I would expect the stock to drop another $5-6 and see shares drop ~2" suggests a specific numerical claim about stock price decline…*

---

### 4 · newspaper paywall widget  (`token@pos = " read"`)
**CTX:** …purchase a single-day subscription for only $0.75 to **read**

**AV(h₂₂):** *Local newspaper article format following a subscription/publishing platform interface displaying access restrictions. The phrase "Sign up today to become a USA TODAY NETWORK subscriber to read unlimited articles for only $4.99 a month" mirrors a standard access-widget pattern, completing with the familiar "7-day free trial" clause…*

---

### 5 · cooking blog, ratio tip  (`token@pos = " keep"`)
**CTX:** …you can halve the Maple Syrup and Stock Powder for a less intense flavour, but if you are used to strong sugary beans, you might want to **keep**

**AV(h₂₂):** *British cooking blog format with recipe notes sharing ingredient variations. The sentence "…you can reduce the amount of honey but if you want the full flavour then keep" is explaining a ratio adjustment choice, implying a contrast between reduced and full recipe…*

---

### 6 · horse-racing history column  (`token@pos = " handicap"`)
**CTX:** …The Virginia homebred trained by R.E. Handlen was the 3-year-old filly and **handicap** mare champion in 1954 and **handicap**

**AV(h₂₂):** *Sports racing newspaper column with historical racing statistics, listing Hall of Fame achievements for a champion mare. The sentence "In 1972, she was the American champion turf filly and handicap mare, handicap" mirrors earlier descriptions, continuing a list of championship titles…*

---

**Reading these as data:** the descriptions are accurate about register and topic
and make a concrete next-token guess. The Stage-0 question is whether the *slice*
of this that h₂₀ doesn't already pin down (the layer-20→22 refinement) measurably
helps reconstruct h₂₂ — and whether it beats the raw-context control. Stage-0
point estimates: text **0.881** > none **0.867** > cat **0.862** (significance via
doc-clustered bootstrap pending a per-position retrain).
