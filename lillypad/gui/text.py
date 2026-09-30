"""User-facing strings shared by more than one part of the interface.

Only strings that appear in two or more places belong here. A message used once
stays where it is used — the point of this module is that a sentence the
operator reads in several contexts says exactly the same thing in each, not to
collect every string in the program.
"""

# Said in three places (the auto-scale checkboxes) about one gesture, so it is
# written once — FrogCanvas.reset_axes is what implements it.
RIGHT_CLICK_HINT = ("Right-click the plot to toggle this: off fits the full "
                    "range and follows it live, on freezes the view where it "
                    "is.")


# Shown whenever a device operation has to be refused because the live feed did
# not release the hardware in time.
FEED_BUSY_MSG = ("Live feed is still mid-acquisition — try again in a moment "
                 "(or stop the feed first).")
