# Problem-Set-3-Econ_5200_PS3.ipynb
# ECON 5200 Problem Set 3 — The Rollout That Looked Like It Worked

## Objective
I studied how staggered treatment adoption affects Difference-in-Differences (DiD) estimates and how to improve causal inference using clean ATT and clustered standard errors.

## Methodology
- Reproduced Card and Krueger's minimum-wage DiD analysis using the original dataset.
- Simulated a panel of 40 counties across 8 regional boards over 12 years with different treatment adoption times.
- Applied TWFE and used the Goodman-Bacon decomposition to examine forbidden comparisons.
- Calculated clean ATT using never-treated and not-yet-treated counties as controls.
- Conducted an event study to examine pre-treatment and post-treatment trends.
- Compared default, county-clustered, and board-clustered standard errors.
- Used AI in Phase 4 to implement a wild cluster bootstrap and evaluate the results.

## Key Findings
- The Card and Krueger DiD estimate was +2.754 FTE employees, close to the published +2.76.
- The TWFE estimate was 2.084, compared with the true ATT of 4.426.
- Forbidden comparisons accounted for 37.0% of the Goodman-Bacon decomposition weight.
- In an additional simulation, TWFE estimated -5.546 even though all true treatment effects were positive.
- Clean ATT was 4.774, closer to the true ATT than TWFE.
- The default, county-clustered, and board-clustered standard errors were 0.207, 0.517, and 1.197.
- The naive board-clustered p-value was 0.1254, while the wild cluster bootstrap p-value was 0.1641. Neither was below 0.05.

## Conclusion
I learned that TWFE can produce misleading estimates when treatment starts at different times and effects change over time. Clean ATT can improve the estimate by avoiding already-treated control groups, but it still depends on the parallel trends assumption. I also learned that clustering at the treatment assignment level matters for statistical inference. Based on my results, I could not establish a statistically significant program effect using the reported TWFE tests.
