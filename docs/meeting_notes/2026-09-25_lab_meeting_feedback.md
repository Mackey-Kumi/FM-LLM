# Feedback from the September 2026 DISAL progress meeting (25 Sep 2026)

Feedback from the September meeting, to shape the Accra brownout early-warning application.

1. **Shorter horizons.** Don't predict too far into the future. He wants something people can use in the **next day or
   two**.
2. **Don't present uncertain predictions as certain.** People will make decisions based on what we tell them. We
   suggested showing a confidence level, and his answer was "maybe".
3. **Be careful with average error metrics (MSE/MAE).** It's not that ours are wrong. The problem is that averages hide
   individual experiences: one person gets a bad forecast, another a good one, and the average still looks good for
   everyone.
4. **Experiment with different look-back windows.** For example: take the 14 days of history before the most recent 7
   days, and leave those last 7 days out (21 days in total). Does that change the model's prediction? Try that kind of
   variation.
