"""Raw vendor SDK wrappers.

Code in here knows NOTHING about Lillypad and is meant to stay that way — it
should be liftable into another project unchanged. Where a wrapper needs to know
something about its host (where the program is installed, say), the host tells it
through a hook rather than the wrapper importing upwards. See
avaspec.add_search_dir.

The adapters that translate these into SpectrometerBase / StageBase live one
level up, in spectrometers/ and stages/.
"""
