# ntupledicts

Author: Casey Pancoast  
Email: <cqpancoast@gmail.com>

**A package for dealing with CMS TrackTrigger ntuples as Python dictionaries.  
Designed with machine learning studies in mind.**

Info on the CMS TrackTrigger can be found [here](https://arxiv.org/abs/1705.04321).
Info on CMS as a whole can be found [here](https://home.cern/science/experiments/cms).

I'd like to thank [Claire Savard](https://github.com/cgsavard) for her previous work in machine learning for the track trigger.
All plots in the `ntupledicts.ml.plot` module are based off of ones that she developed.


## Install and Run

It's on pip — `pip install ntupledicts`.

Note that this package does not depend upon any particular version of CMSSW or have any requirements for track properties in the imported ntuples.
I've been thinking about separating this package into CMS/track-trigger functionality and more general machine learning functionality in order to more easily apply it to my personal use.


## Working with ntuples as Python dictionaries

Event data is stored in an object called an **ntuple dictionary** (or **ntuple dict**).
This is a dictionary from track types ("trk", "tp", "matchtrk", "matchtp") to dicts
from track properties ("eta", "chi2", "nmatch") to lists of properties. (These
smaller dicts within the ntuple dicts are called "**track property dicts**".)
For example, a simple **ntuple dict** might look like this:

```python
{"trk": {"pt": [1, 2, 3], "eta": [0, 2.2, 1.1]}, "tp": {"nmatch": ...}}
```

The **ntuple dict**'s values here (e.g., `{"pt": [1, 2, 3], ...}`) are **track property dicts**.
In the code, the lists of track property values are called **value lists**.

The whole formula looks about like this:

```python
val_list = ntuple_dict[track_type][track_property]
track_prop_dict = ntuple_dict[track_type]
val_list = track_prop_dict[track_property]
```

***Note***: I choose to define **track property dicts** such that even **value lists** that are not drawn from the input ntuples are valid.
Now you can add custom track properties (see "analyzing the contents of an ntuple dict) and ML model predictions (see the section on machine learning) without worrying about whether it's considered hacky to do so.

### Creating an ntuple dictionary

```python
import ntupledicts.load as ndload
```

Here's a sample of code where I make an **ntuple dict** from root ntuples:

```python
input_files = ["TTbar_PU200_D49.root", "QCD_PU200_D49.root"]

# Specify desired properties
properties_by_track_type = {"trk": ["pt", "eta", "genuine"],
                            "matchtrk": ["pt", "eta", "nstub", "chi2rphi", "chi2rz"],
                            "tp": ["pt", "eta", "nstub", "dxy", "d0", "eventid", "nmatch"]}

# Create ntuple properties dict from event set
ntuple_dict = ndload.root_files_to_ntuple_dict(input_files, properties_by_track_type)
```

It is as easy as this: choose whichever samples you wish, choose the properties you want from them, and shove this into a function.

Note that this function, `root_files_to_ntuple_dict`, by default cuts tracks with invalid values like `inf` or `nan` upon creation.
As this takes time, this can be disabled with a keyword argument, as those values do not appear frequently.
However, it is the default, as even one `inf` or `nan` can ruin a machine learning train session.

### Applying cuts to an ntuple dictionary

```python
from ntupledicts.operations import select as sel
```

Now, say I want to apply some cuts to the **ntuple dict**. Cuts are performed using objects called **selectors**, functions which take in a value and spit out true or false.
For example, a **selector** might be:

```python
lambda eta: eta <= 2.4 and eta => -2.4
```

However, there's a convenient function in the `ntupledicts.operations` library that transforms that into this:

```python
sel(-2.4, 2.4)
```

These **selectors** are collected into "**selector dicts**" which have the same
format as **ntuple dicts** and **track properties dicts**, but replace their value lists with **selectors**.

So, to apply a cut to tracking particles in an **ntuple dict**, I'd do this:

```python
from ntupledicts.operations import select as sel
from ntupledicts.operations import cut_ntuple

ntuple_dict_selector = {"tp": {"eta": sel(-2.4, 2.4), "pt": sel(2, 100), "eventid": sel(0)}}
ntuple_dict = cut_ntuple(ntuple_dict, general_cut_dicts)
```

One convenient thing about `sel()` here is that it can select a particular value as well as a range, for track properties that take discrete rather than continuous values.
This is shown above in the case of eventid.

To logical `AND` with **selector**s, simply apply two **selector**s.
To logical `OR`, pass your desired **selector**s to logical `OR` into `sel` as a list.

```python
sel([sel(0), sel(1, 4)])
```

This will select zero and any value between one and four, inclusive.
To "reverse" any **selector**, that is, make it select everything but what is specified, add the keyword arg `invert=True` into a composed **selector**.
For example, `sel([sel(1, 3)], invert=True)` will select all values outside of the inclusive range one through three.

#### Other functions of note in ntupledicts.operations

```python
import * from ntupledicts.operations
```

- **Ntuple dicts** with the same track types and properties can be added together with `add_ntuple_dicts`.
- `select_indices` returns the indices in a **track properties dict** selected by a **selector** of the same form.
- `ntuple_dict_length` returns a dictionary from track type to number of tracks. Some sample output might be `{"trk": 101, "tp": 89}`
- `reduce_ntuple_dict` takes in a dictionary from track types to track property **value list** lengths and cuts those lists to the given sizes.
- `shuffle_ntuple_dict` shuffles the **ntuple dict**, respecting the association between tp/matchtrk and trk/matchtp tracks.

Also, note that most functions that do something to **ntuple dict**s have
corresponding functions that do that thing to **track property dict**s.

### Analyzing the contents of an ntuple dict

```python
import * from ntupledicts.analyze
```

The analyze module includes functions for getting the efficiency of a sample from a track properties dict, getting the proportion of a dict selected by some selector, and binning a dict by some track property or another.
The most interesting part of the module is the `StubInfo` class, which allows you to make custom track properties based on stub information associated with the stub.

You would find the number of missing 2S or PS stubs associated with a track and create a new track property for it like this:

```python
missing_2S_layer = lambda expected, hit, ps: not ps and expected and not hit
missing_PS_layer = lambda expected, hit, ps: ps and expected and not hit

track_prop_dict["missing2S"] = create_stub_info_list(track_prop_dict,
        ndanl.basic_process_stub_info(missing_2S_layer))
track_prop_dict["missingPS"] = create_stub_info_list(track_prop_dict,
        ndanl.basic_process_stub_info(missing_PS_layer))
```

`create_stub_info_list()` is a function that uses the eta and hitpattern associated with each track in a track properties dict (assuming those track properties have been included) to generate stub information.
The eta is used to find which layers are expected to be hit and whether each is a PS or 2S module.
The hitpattern is then used to find which of those expected layers were hit.

Using both of these, `basic_process_stub_info()` is able to take in the lambda expressions `missing_2S_layer` and `missing_PS_layer` (as well as any that can be defined using that form) to find that information for each track, creating a list that can simply be added to the `track_prop_dict`.

For more information about the internals of this process, see the `StubInfo` class in `ntupledicts.analyze`.

### Plotting

The main plotting library includes some functions for making histograms of track properties and making a(n) ROC curve out of different sets of cuts.

All functions in `ntupledicts.plot` (and in `ntupledicts.ml.plot`) accept and return an axes object for ease of use in overlaying.


## For Machine Learning

Contained in `ntupledicts.ml` is everything you'll need to make a machine learning model, configure it, train it on data, test its performance, and plot the result of those tests.

### Data

```python
from ntupledicts.ml.data import TrackPropertiesDataset
```

All data is stored in a `TrackPropertiesDataset`, which is essentially a track properties dict with its own class and some ML-focused functionality.
It separates the data contained in an input track properties dict into data and labels, in accordance with standard machine learning practice.

```python
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

To get the active data and labels, run:

```python
tpds.get_data()  # Tensorflow array of data
tpds.get_labels()  # Tensorflow array of labels
tpds.get_data(["pt", "nstub"])  # Tensorflow array of only pt and nstub data
```

By default, `get_data()` normalizes the data for each property, for better use in model training.
This can be disabled with the keyword argument `normalize=False`.


### Models

```python
from ntupledicts.ml.models import make_neuralnet
from ntupledicts.ml.models import make_gbdt
```

There are some convenient wrapper functions for common networks.
For example, for a tensorflow neural network, rather than building it yourself, you can specify hidden layers:

```python
NN = make_neuralnet(train_ds, validation_data=eval_ds, hidden_layers=[14, 6], epochs=10)
GBDT = ndmlmodels.make_gbdt(train_ds)
```

However, you are by no means restricted to using these functions to create your models.
These may lack the configurability required for your research.


### Prediction

```python
import ntupledicts.ml.predict as ndmlpred
```

Just like there are wrappers to create models, there are also wrappers to run them on data.
These will create lists of probabilities of label predictions.

```python
pred_labels = ndmlpred.predict_labels(GBDT, test_ds.get_data())
```

`TrackPropertiesDataset`s are capable of storing predictions, previous ones of which can be accessed by label.

```python
test_ds.add_prediction("NN", ndmlpred.predict_labels(NN, test_ds.get_data()))
test_ds.get_prediction("NN")  # Tensorflow array of labels predicted by model NN
```

There is also support for having a selector (or, in common speak, a set of cuts) predict labels.

```python
some_track_property = "pt"  # a track property to cut on
some_selector = sel(0, 10)  # only accept values between zero and ten
cut_pred_labels = ndmlpred.predict_labels_cuts({some_track_property: some_selector})
  # returns a list of 1's corresponding to tracks with pts below 10, 0's above
```

`ndmlpred` also has functions `true_positive_rate()` and `false_positive_rate()` (or `tpr` and `fpr`) that calculate exactly what you'd expect if given a threshold value to turn probablistic predictions into binary predictions.
These functions are used often in the plots below.


### Plotting

`ntupledicts.ml.plot` consists of a function that plots the ROC curve of a model and a couple functions that split a `TrackPropertiesDataset` into bins and then compute `tpr`/`fpr` for each bin.
This ascertains the performance of a model on different types of tracks.
Say, for example, the model did wonderfully for high pt and terribly for low pt.
You might see high `tpr` and low `fpr` for high pt and the reverse
for low pt.

All of the plotting functions in `ntupledicts.ml.plot` as of now are generalizations of ones developed by [Claire Savard](https://github.com/cgsavard), a grad student in high energy physics at CU Boulder. Props to her!


## Potential Improvements / "How can I contribute?"

If `ntupledicts` develops a usership, I'd be happy to add this functionality; I just haven't found it useful for my own work.
But you know what's even better than me adding functionality?
Someone else doing it, and submitting a pull request!

### General

- There are many places in which 2-tuples of a value and some error are returned. There should be an option in all functions like this to return only the value.
- Greater cut sophistication: selectors that can operate on more than one track property at a time.
- There just aren't that many types of plots. I only made the plots that were relevant to my work; more might be necessary for others (and this is a super easy topic to submit a pull request for).
- Boring but important: `root_files_to_ntuple_dict()`, the "entry point" to `ntupledicts`, is a bit hacky and hard-coded with the way it extracts ntuples from the files. It's worked for me so far, but you're welcome to check it out for yourself.

### ML

- Making tensorflow deterministic. It's harder than you'd think.
- Saving models and datasets for future use. I never needed to do this, as I usually ran stuff in a Jupyter Notebook.
- More model configurability from the model creation wrapper functions — it's hard to know what's too much configurability and what isn't enough.
- And, obviously, support for as many models as possible. I only needed NN's and GBDT's, but others sure do exist.
- [Hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization): check it out!
- Multi-class learning - currently, most functions only have support for binary classification (genuine == 1 vs. genuine == 0 being the canonical example).
However, models can be trained to classify other discrete variables, such as pdgid. Right now, you can't use `ntupledicts` to make a model that picks electrons or muons out of a slurry of particles.

