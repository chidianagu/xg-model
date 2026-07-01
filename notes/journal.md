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
