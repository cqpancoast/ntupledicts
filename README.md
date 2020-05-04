#ntupledict

A package for dealing with CMS TrackTrigger ntuples as dictionaries. Designed with machine learning studies in mind.

There are three parts to this code:

- Getting Data.
    One or more ntuples is read in. If more than one ntuple is
    read in, they are all concatenated together. More on the data
    structures used below.
- Applying Cuts.
    Place arbitrary cuts on the event collection on any property of
    any track type.
- Plotting.
    The matplotlib library is (for some) easier to work with than
    the ROOT plotting library. However, matplotlib does not know you
    are a pysicist! This is not intended as a replacement for ROOT,
    just an alternative.


## Data Definitions and Scheme

All event data is stored in an object called an ntuple dictionary. This is a
dictionary from track types ("trk", "tp", "matchtrk", "matchtp") to dicts
from track properties ("eta", "chi2", "nmatch") to lists of properties. (These
smaller dicts within the ntuple dictionaries are called "track property dicts".)
For example, a simple ntuple dictionary might look like this:

    {"trk": {"pt": [1, 2, 3], "eta": [0, 2.2, 1.1]}, "tp": {"nmatch": ...}}

The values here are the track property dictionaries.

In the main method, you can specify which properties you want to access from
the ntuple to have in your track type dictionary. This is nice, because you don't
have to deal with hauling around any more data than you have to!


## Applying Cuts

To perform a cut on a track type dictionary... TODO docs!!!


## Plotting

All plotting functions follow certain rules:
    - All return an axes object
    - None clear or save the figure
    - Some set labels (x-axis, y-axis, title) but some do not


