# (Our repository for) Machine Learning Studies for the CMS Track-Trigger

This repo holds results of various ML studies aimed at improving CERN's CMS's upcoming track-trigger by letting the outer tracker send additional information created by ML along to the L1 Trigger for it to make decisions. Depending on a bunch of things, this repo might be split into a few in the future, or it might be moved to a repo with a different name.

## What's going on here?

__In a sentence:__ our only current project is studying how new chi2 variables added to each track outputted by the track-trigger algorithm can improve performance of track-quality selection algorithms based on machine learning.

Currently, the only study-in-progress in this repository is one in which ML machines are built to make use of chi2 variables newly accessible to the L1 Trigger. Originally, there was one chi2 variable per track that showed the track's adherence to the stub sequence it was generated from, but now there is a chi2 for each track measuring its fit with in the phi plane and another for the r-z plane. _This is useful because_ electrons in particular have super high chi2 values in the phi plane when compared to other particles (like muons), but have about the same in the r-z plane. Below are some graphs that make this clear:

__TODO__ add chi2/DOF graphs for singleMuon/singleElectron found from TTbar samples.
