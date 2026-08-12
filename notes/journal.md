## Session notes — day 1

- The freeze frame data is richer than I expected; for many shots there
  are 15+ players visible. This should make defender-counting features
  really powerful.
- shot_end_location has a z coordinate for height — could be useful
  for a separate "where do keepers concede" visualisation later.
- The World Cup final had only 2-3 headers; sample sizes from one match
  are tiny, need full dataset before trusting conversion rate numbers.
- Question: do penalties have freeze frames? They seemed sparse —
  worth checking at scale.
- Grouping shots by categories in dictionaries allows for better analysation, especially as data sizes scale.


## Session notes - Session 2 01/07/2026

### What I did

   - Picked project back up
   - Hit a ModuleNotFoundError on restart, turned out Jupyter was running on a different Python than my venv. Fixed by running Jupyter from the same terminal environment.
   - Re-ran full notebook top to bottom to confirm everything from session 1 still worked
   - Scaled up from a single match (the final) to all 64 World Cup 2022 matches, looped over match IDs, pulled events for each, filtered to shots, concatenated into one dataframe (1453 shots total)
   - Merged in team names and competitions stage from the matches table
   - Found that the shots database was including penalty shootout kicks in its penalty data.
   - Moved the shootout exclusion filter to early in the pipeline, right after the full shots dataframe construction.
   - Re-ran feature engineering (x, y, distance_to_goal) and shot_groups filters on the full, cleaned dataset.



### Observations

  - At full tournament scale, conversion rates and avg xG line up closely (11.6% conversion vs 0.107 avg xG)
  - Single match sample sizes from session 1 were far noisier than I realised, penalties, headers etc. look very different at n=1453 vs n=25
  - Lesson learned: data cleaning decisions need to happen as early as possible, or different cells in the notebook end up silently inconsistent with each other



### Next Session

  - Proper feature engineering, angle to goal, defenders in the shot-to-goal triangle, distance to nearest defender, goalkeeper position, etc. - using freeze frame data
  - Consider other competitions besides WC2022 for more training data
  - Move shot_filters and feature-computation functions out of the notebook into src/ now that they're stable and reusable


## Session notes - Session 3 09/08/2026 - 12/08/2026

### What I did  

  - Built defenders_in_triangle feature using freeze frame data and a
  point-in-triangle geometry test (sign method / cross products) —
  verified with sanity checks (conversion rate dropped cleanly from
  38.7% with 0 defenders to ~0-4% with 3+) and a visual pitch plot
  overlaying the triangle
  - Added dist_to_nearest_defender, goalkeeper positioning features
  (gk_dist_to_line, gk_y_offset), and several StatsBomb event tags:
  under_pressure, shot_open_goal, shot_first_time, is_counter_attack,
  shot_one_on_one — 11 features total
  - Checked missingness per feature — 22 shots (mostly penalties)
  missing all freeze-frame-derived features, which is consistent and
  explainable
  - Built the modelling dataset, dropping those 22 rows
  - Did a grouped train/test split (by match_id, to avoid leakage
  across shots from the same game)
  - Trained a baseline logistic regression
  - Compared directly against StatsBomb's own xG on the same test shots
  - Generated a calibration plot comparing my model vs StatsBomb,
  saved to reports/calibration_comparison.png
  - Updated README with a results summary table and next steps

### Results

  - My model — log loss: 0.2470, AUC: 0.8664
  - StatsBomb xG — log loss: 0.2428, AUC: 0.8542
  - Essentially matched StatsBomb on log loss and beat it on AUC, with
  far fewer features and one tournament of data. However, with one test
  split and one competition it is not a general claim of beating StatsBomb

### Investigations

  - angle_to_goal had a near-zero coefficient in the full model despite
  being a strong predictor alone — checked correlation with
  distance_to_goal, found -0.75. Points to multicollinearity, distance 
  absorbs most of the shared signal
  - gk_y_offset came out with a negative coefficient (unexpected as more
  off-centre keeper should mean easier chance). Bucket analysis showed
  the expected positive trend for most of the range, but the top
  bucket (very few shots) collapsed to 0% and dragged things off.

### Next session

  - Try XGBoost/LightGBM as a stronger model, compare to
  the logistic regression baseline
  - Consider pulling in more competitions for a larger training set
  - Eventually move shot_filters, feature functions, and the modelling
  pipeline out of the notebook into src/
  - Get back into the habit of committing/journaling every session,
  a few sessions ran together this time.
