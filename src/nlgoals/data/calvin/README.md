# CALVIN Dataset Code

This directory contains the code for the datasets, datamodules and evaluation
associated with the CALVIN dataset.

Unfortunately it is a bit of a mess because I started writing my own code as you
normally would with a dataset (see [legacy](./legacy/)).

However, the CALVIN dataset presents a series of non-trivial design choices
which ultimately made it very difficult to work on it from scratch. As a result,
I ended up porting the original CALVIN repository code here (see
[repo](./repo/), which I then modified a little bit to fit my requirements.

The rest of the code in this repository should be slightly better I hope.
