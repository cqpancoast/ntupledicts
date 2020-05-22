# ntupledicts

Author: Casey Pancoast  
Email: <cqpancoast@gmail.com>

**A package for dealing with CMS TrackTrigger ntuples as Python dictionaries.  
Designed with machine learning studies in mind.**

Info on the CMS TrackTrigger can be found [here](https://arxiv.org/abs/1705.04321).
Info on CMS as a whole can be found [here](https://home.cern/science/experiments/cms).

I'd like to thank [Claire Savard](https://github.com/cgsavard) for her previous
work in machine learning for the track trigger.
All plots in the `ntupledicts.ml.plot` module are based off of ones that she developed.


## What you'll need

I'll turn this into a package to make this easier soon, but before
that, here are the dependencies so far:

- `tensorflow 2.0.0` or above for machine learning
- `sklearn` for more machine learning
- `uproot` to read in the ROOT ntuples

Note that this package does not depend upon any particular version of CMSSW
or have any requirements for track properties in the imported ntuples.


## Working with ntuples as Python dictionaries

Event data is stored in an object called an **ntuple dictionary** (or **ntuple dict**).
This is a dictionary from track types ("trk", "tp", "matchtrk", "matchtp") to dicts
from track properties ("eta", "chi2", "nmatch") to lists of properties. (These
smaller dicts within the ntuple dicts are called "**track property dicts**".)
For example, a simple **ntuple dict** might look like this:

```
{"trk": {"pt": [1, 2, 3], "eta": [0, 2.2, 1.1]}, "tp": {"nmatch": ...}}
```

The **ntuple dict**'s values here (e.g., `{"pt": [1, 2, 3], ...}`) are **track property dicts**.
In the code, the lists of track property values are called **value lists**.

The whole formula looks about like this:

```
val_list = ntuple_dict[track_type][track_property]
track_prop_dict = ntuple_dict[track_type]
val_list = track_prop_dict[track_property]
```

***Note***: I choose to define **track property dicts** such that even **value lists** that are not
drawn directly from the input ntuples are valid. For example, if you wanted a machine
learning model's prediction to be a track property **value list**, that would be perfectly
valid, *as long as the predictions are indexed by the same tracks that made the track
property dict in the first place*.

### Creating an ntuple dictionary

Here's a sample of code where I make an **ntuple dict** from root ntuples:

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

(`from ntupledicts.operations import select as sel`)

Now, say I want to apply some cuts to the **ntuple dict**. Cuts are performed using objects called **selectors**, functions which take in a value and spit out true or false.
For example, a **selector** might be:

```
lambda eta: eta <= 2.4 and eta => -2.4
```

However, there's a convenient function in the `ntupledicts.operations` library that transforms that into this:

```
sel(-2.4, 2.4)
```

These **selectors** are collected into "**selector dicts**" which have the same
format as **ntuple dicts** and **track properties dicts**, but replace their value lists with **selectors**.

So, to apply a cut to tracking particles in an **ntuple dict**, I'd do this:

```
from ntupledicts.operations import select as sel
from ntupledicts.operations import cut_ntuple

ntuple_dict_selector = {"tp": {"eta": sel(-2.4, 2.4), "pt": sel(2, 100), "eventid": sel(0)}}
ntuple_dict = cut_ntuple(ntuple_dict, general_cut_dicts)
```

One convenient thing about `sel()` here is that it can select a particular value as well as a range, for track properties that take discrete rather than continuous values.
This is shown above in the case of eventid.

To logical `AND` with **selector**s, simply apply two **selector**s.
To logical `OR`, pass your desired **selector**s to logical `OR` into `sel` as a list, like so:

```
sel([sel(0), sel(1, 4)])
```

This will select zero and any value between one and four, inclusive.
To "reverse" any **selector**, simply add the keyword arg `invert=True` into a composed **selector**.
For example, `sel([sel(1, 3)], invert=True)` will select all values outside of the inclusive range one through three.

### Other functions of note in ntupledicts.operations

(`import * from ntupledicts.operations`)

- **Ntuple dicts** with the same track types and properties can be added together with `add_ntuple_dicts`.
- `select_indices` returns the indices in a **track properties dict** selected by a **selector** of the same form.
- `ntuple_dict_length` returns a dictionary from track type to number of tracks. Some sample output might be `{"trk": 101, "tp": 89}`
- `reduce_ntuple_dict` takes in a dictionary from track types to track property **value list** lengths and cuts those lists to the given sizes.
- `shuffle_ntuple_dict` shuffles the **ntuple dict**, respecting the association between tp/matchtrk and trk/matchtp tracks.

Also, note that most functions that do something to **ntuple dict**s have
corresponding functions that do that thing to **track property dict**s.

### Plotting

The main plotting library includes some functions for making histograms of track properties and making a(n) ROC curve out of different sets of cuts.

All functions in `ntupledicts.plot` (and in `ntupledicts.ml.plot`) accept and return an axes object for ease of use in overlaying.

## For Machine Learning

Contained in `ntupledicts.ml` is everything you'll need to make a machine learning model, configure it, train it on data, test its performance, and plot the result of those tests.

### Data

(`from ntupledicts.ml.data import TrackPropertiesDataset`)

All data is stored in a `TrackPropertiesDataset`, which is essentially a track
properties dict with some ML-focus functionality.
It separates the data contained in an input track properties dict into data and labels, in accordance with standard machine learning practice.

```
tpd = ntupledict["trk"]  # make a track properties dict
tpd.keys()  # ["pt", "eta", "nmatch", "genuine"]
active_data_properties = ["pt", "eta"]  # set pt and eta as data to train on
label_property = "genuine"  # have genuine be the property that a model trains on

tpds = TrackPropertiesDataset(tpd, active_data_properties, label_property)
tpds.get_active_data_properties()  # ["pt", "eta"]
tpds.get_available_data_properties()  # ["pt", "eta", "nmatch", "genuine"]
tpds.get_label_property()  # "genuine"
```

The label property and the active data property can also be set in an already instantiated dataset, though this is less common.

To get the active data and labels, simply run:

```
tpds.get_data()  # Tensorflow array of data
tpds.get_labels()  # Tensorflow array of labels
tpds.get_data(["pt", "nstub"])  # Tensorflow array of only pt and nstub data
```


### Models

```
from ntupledicts.ml.models import make_neuralnet
from ntupledicts.ml.models import make_gbdt
```

There are some convenient wrapper functions for common networks.
For example, for a tensorflow neural network, rather than building it yourself, you can simply specify hidden layers:

```
NN = make_neuralnet(train_ds, validation_data=eval_ds, hidden_layers=[14, 6], epochs=10)
GBDT = ndmlmodels.make_gbdt(train_ds)
```

However, you are by no means restricted to using these functions to create your models.


### Prediction

(`import ntupledicts.ml.predict as ndmlpred`)

Just like there are wrappers to create models, there are also wrappers to run them on data. These will create lists of probabilities of label predictions.

```
pred_labels = ndmlpred.predict_labels(GBDT, test_ds.get_data())
```

`TrackPropertiesDataset`s are capable of storing predictions, previous ones of which can be accessed by label like so:

```
test_ds.add_prediction("NN", ndmlpred.predict_labels(NN, test_ds.get_data()))
test_ds.get_prediction("NN")  # Tensorflow array of labels predicted by model NN
```

There is also support for having a selector (or, in common speak, a set of cuts) predict labels. This is done like so:

```
some_track_property = "pt"  # a track property to cut on
some_selector = sel(0, 10)  # only accept values between zero and ten
cut_pred_labels = ndmlpred.predict_labels_cuts({some_track_property: some_selector})
  # returns a list of 1's corresponding to tracks with pts below 10, 0's above
```

`ndmlpred` also has functions `true_positive_rate()` and `false_positive_rate()` (or `tpr` and `fpr`) that calculate exactly what you'd expect if given a threshhold value to turn probablistic predictions into binary predictions. 
These functions are used often in the plots below.


### Plotting

`ntupledicts.ml.plot` consists of a function that plots the ROC curve of a model and a couple functions that split a `TrackPropertiesDataset` into bins and then compute `tpr`/`fpr` for each bin.
This ascertains the performance of a model on different types of tracks.
Say, for example, the model did really well for high pt and really bad for low pt.
You might see high `tpr` and low `fpr` for high pt and the reverse
for low pt.

All of the plotting functions in `ntupledicts.ml.plot` as of now are generalizations of ones developed by [Claire Savard](https://github.com/cgsavard), a grad student in high energy physics at CU Boulder. Props to her!


## #TODO: Potential improvements

### General

- Greater cut sophistication: selectors that can operate on more than one track property at a time
- Saving models and datasets for future use

### ML

- More model configurability from the model creation wrapper functions — it's hard to know what's too much configurability and what isn't enough
- Support for more than one track property to contribute to a label, if desired
  - "Composite labels"?
- Obviously, support for as many models as possible

