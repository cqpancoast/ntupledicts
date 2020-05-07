# ntupledicts

A package for dealing with CMS TrackTrigger ntuples as dictionaries.
Designed with machine learning studies in mind.


## Working with Ntuples as Dicts

Event data is stored in an object called an ntuple dictionary. This is a
dictionary from track types ("trk", "tp", "matchtrk", "matchtp") to dicts
from track properties ("eta", "chi2", "nmatch") to lists of properties. (These
smaller dicts within the ntuple dictionaries are called "track property dicts".)
For example, a simple ntuple dictionary might look like this:

```
{"trk": {"pt": [1, 2, 3], "eta": [0, 2.2, 1.1]}, "tp": {"nmatch": ...}}
```

The values here are the track property dictionaries.

### Creating an ntuple dictionary

Here's a sample of code where I make an ntuple dictionary from root ntuples:

```
# Open ntuples
event_sets = []
for input_file in input_files:
    event_sets.append(next(iter(uproot_open(input_file).values()))["eventTree"])

# Specify desired properties
properties_by_track_type = {"trk": ["pt", "eta", "genuine"],
                            "matchtrk": ["pt", "eta", "nstub", "chi2rphi", "chi2rz"],
                            "tp": ["pt", "eta", "nstub", "dxy", "d0", "eventid", "nmatch"]}

# Create ntuple properties dict from event set
ntuple_dict = ndops.ntuples_to_ntuple_dict(event_sets, properties_by_track_type)
```

### Applying cuts to an ntuple dictionary

Now say I want to apply some cuts to the ntuple dictionary. Cuts are performed using
objects called "selectors", functions which take in a value and spit out true or false.
For example, a selector might be:

```
lambda eta: eta <= 2.4 and eta => -2.4
```

However, there's a convenient function in the `ntupledicts.operations` library that
transforms that into this:

```
ntupledicts.operations(-2.4, 2.4)
```

These selectors are collected into "selector dictionaries" which
have the same format as ntuple dictionaries and tracks properties dictionaries, but
replace their list of values with selectors.

So, to apply a cut to tracking particles in an ntuple dictionary, I'd do this:

```
from ntupledicts.operations import select as sel
from ntupledicts.operations import cut_ntuple

ntuple_dict_selector = {"tp": {"eta": sel(-2.4, 2.4), "pt": sel(2, 100), "eventid": sel(0)}}
ntuple_dict = cut_ntuple(ntuple_dict, general_cut_dicts)
```

One convenient thing about `sel()` here is that it can select a particular value as well
as a range, for track properties that take discrete rather than continuous values.

### Other functions of note in ntupledicts.operations

(`import * from ntupledicts.operations`)

- Ntuple dictionaries with the same track types and properties can be added
together with `add_ntuple_dicts`.
- `select_indices` returns the indices in a track properties dictionary selected
by a selector of the same form.
- `ntuple_dict_length` returns a dictionary from track type to number of tracks.
Some sample output might be `{"trk": 101, "tp": 89}`
- `reduce_ntuple_dict` takes in a dictionary from track types to track property
value list lengths and cuts those lists to the given sizes.

Also, note that most funcitons that do something to ntuple dictionaries have
corresponding functions that do that thing to track property dictionaries.

### Plotting

There is also a small library of plots that someone might find useful, but
this hasn't been given as much attention as the operations portion of the code.

If you're interested, check it out in `ntupledicts.plot`


## For Machine Learning




