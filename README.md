# ntupledicts

A package for dealing with CMS TrackTrigger ntuples as Python dictionaries.
Designed with machine learning studies in mind.


## What you'll need

I'll turn this into a package to make this easier soon, but before
that, here are the dependencies so far:

- `tensorflow 2.0.0` or above for machine learning
- `sklearn` for more machine learning
- `numpy` for all your various numpy needs (trying to get rid of this)
- `uproot` to read in the ROOT ntuples

Note that this package does not depend upon any particular version of CMSSW
or have any requirements for track properties in the imported ntuples.


## Working with ntuples as Python dictionaries

Event data is stored in an object called an ntuple dictionary (or ntuple dict).
This is a dictionary from track types ("trk", "tp", "matchtrk", "matchtp") to dicts
from track properties ("eta", "chi2", "nmatch") to lists of properties. (These
smaller dicts within the ntuple dicts are called "track property dicts".)
For example, a simple ntuple dict might look like this:

```
{"trk": {"pt": [1, 2, 3], "eta": [0, 2.2, 1.1]}, "tp": {"nmatch": ...}}
```

The ntuple dict's values here (e.g., `{"pt": [1, 2, 3], ...}`) are track property dicts.
In the code, the lists of track property values are called value lists.

### Creating an ntuple dictionary

Here's a sample of code where I make an ntuple dict from root ntuples:

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

Now say I want to apply some cuts to the ntuple dict. Cuts are performed using
objects called "selectors", functions which take in a value and spit out true or false.
For example, a selector might be:

```
lambda eta: eta <= 2.4 and eta => -2.4
```

However, there's a convenient function in the `ntupledicts.operations` library that
transforms that into this:

```
ntupledicts.operations.select(-2.4, 2.4)
```

These selectors are collected into "selector dictionaries" which have the same
format as ntuple dicts and tracks properties dicts, but replace their list of 
values with selectors.

So, to apply a cut to tracking particles in an ntuple dict, I'd do this:

```
from ntupledicts.operations import select as sel
from ntupledicts.operations import cut_ntuple

ntuple_dict_selector = {"tp": {"eta": sel(-2.4, 2.4), "pt": sel(2, 100), "eventid": sel(0)}}
ntuple_dict = cut_ntuple(ntuple_dict, general_cut_dicts)
```

One convenient thing about `sel()` here is that it can select a particular value as well
as a range, for track properties that take discrete rather than continuous values. This
is shown above in the case of eventid.

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

Also, note that most funcitons that do something to ntuple dicts have
corresponding functions that do that thing to track property dicts.

### Plotting

There is also a small library of plots that someone might find useful, but
this hasn't been given as much attention as the operations portion of the code.
It's really just stuff that accumulated during my time making plots of this sort.

If you're interested, check it out in `ntupledicts.plot`.


## For Machine Learning

Contained in `ntupledicts.ml` is everything you'll need to make a machine learning
model, configure it, train it on data, test its performance, and plot the result
of those tests.

### Data

All of the models we'll use are from `tensorflow` or `sklearn`, which deal with
tensorflow tensors and numpy arrays respectively. Because tensorflow tensors can
turn into numpy arrays, the primary function for turning ntuple dicts into
datasets converts them into tensors. If I wanted to create train, eval, and test
datasets using a given a given ntuple dict, I'd do this:

```
from ntupledicts.ml.data import make_datasets_from_track_prop_dict as make_datasets

(data_properties, label_property), (train_data, train_labels),\
             (eval_data, eval_labels), (test_data, test_labels) \
             = make_datasets(ntuple_dict["trk"])
```

By default, `make_datasets` splits the data into three datasets with relative sizes
of `.7`, `.2`, and `.1`. Note that I am only creating the dataset using the `trk`
track property dict of the ntuple dict. An improvement I am thinking of making is
to have a dedicated function to merge trk-matchtp or tp-matchtrk track property
dicts. However, this isn't high on the priority lists, as the actual models will
only have access to trk infomation.

### Models

There are some convenient wrapper functions for common networks. For example, for
a tensorflow neural network, rather than building it yourself, you can simply
specify hidden layers:

```
from ntupledicts.ml.models import make_neuralnet

NN = make_neuralnet(train_data, train_labels,
             validation_data=(eval_data, eval_labels), hidden_layers=[14, 6], epochs=10)
```

However, you are by no means restricted to using these functions to create your models.

### Plotting

`ntupledicts.ml.plot` is "under construction" right now — this is envisioned 
as a set of functions that evaluate and compare the performance of networks
in some basic ways. Ideally, these would play nicely with the results of testing
networks that will be in the also-under-construction subpackage
`ntupledicts.ml.test`.


## #TODO: Potential improvements

### General

- Greater cut sophistication: selectors that can operate on more than one track
property at a time
- Saving models and datasets for future use instead of just returning them
- `reduce_ntuple_dict` should have an option to shuffle lists before cutting to
ensure randomly selected tracks
- Clean up generic plotting library

### ML

- More model configurability from the model creation wrapper functions — it's
hard to know what's too much configurability and what isn't enough
- Support for more than one track property to contribute to a label
- Obviously, support for as many models as possible

