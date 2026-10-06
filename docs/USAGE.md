# Using the software

Day-to-day use of the application. To download or install it first, see the
[README](../README.md).

1. **Load the video** — *Charger la vidéo* button.
2. **Draw the line** — *Tracer la ligne* button, then two clicks on the image,
   at the top and bottom of the line you care about. Drawing is "armed": a
   stray click outside that gesture is ignored. The detection band follows the
   line wherever you put it.
3. **Set the direction** — in *Ligne de franchissement*, choose the counted
   direction. The green arrow shows the active direction. **If the count stays
   at zero after a few seconds, this is almost always why** — the software
   shows a reminder rather than failing silently.
4. **Adjust the sensitivity** — the parameter that matters most is *Frames de
   confirmation*: 1 counts immediately (sensitive to false positives), 5 only
   validates a person seen over several consecutive frames. Every parameter has
   a tooltip explaining what it does and which value suits this scene; the
   panel's *Afficher l'aide* button shows them all at once.
5. **Run** — the counter updates continuously, the line flashes on every
   crossing. The band and the line cannot be moved once the run starts.
6. **Read the summary** — when the analysis stops, a section appears between
   the status bar and the settings: total, duration, average rate, peak rate
   and a curve of people per minute.

## The end-of-analysis summary

The summary reports only what was measured:

- **Total** — people counted in the chosen direction.
- **Average rate** — total over the analysed duration.
- **Peak rate** — the busiest 60 s window, with when it happened. On analyses
  shorter than a minute the window shrinks, and the figure is reported as
  `12 pers / 40 s` rather than extrapolated to a rate that was never measured.

There is **no automatic error rate**. There is no ground truth without manual
annotation, and the software does not invent one. The only way to know how far
a total is from the truth is to count a segment by hand and compare.

## Audit mode

During a replay, the *Audit* panel lets you flag errors by eye. The error rate
shown is only valid for the portion you actually checked: it is a
**measurement**, not an automatic estimate.

## Reading the green boxes

When a person is counted, their box turns **green** and is **filled**, hiding
the counted head. The green box **follows the person** as they walk away,
and disappears the moment they leave the detection band on the far side of the
line — a clean exit, no lingering frame.

This is not decoration — **it is the only way to see a mistake without ground
truth.** Nobody can recount a demonstration by hand, and the counter will never
say "I missed one": the total is just a number, and a wrong number looks exactly
like a right one. But a person walking across the line *without* their box
turning green is a visible miss. Twenty seconds of watching tells you whether
the count can be trusted, and nothing else on screen tells you that.

So when you check a result, don't only read the total. Watch the line for a
while and count the green boxes yourself. If a head crosses and stays amber, you
have found a miss — and you know roughly how far off the number is.
