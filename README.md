# Conformal Prediction

Conformal prediction is a framework for providing accuracy guarantees on the predictions
of a base predictor.

------------------


## Installation

Conformal uses the following dependencies:

- numpy,
- pyyaml
- HDF5 and h5py (optional, required if you use model saving/loading functions)

To install Conformal, `cd` to the conformal folder and run the install command:
```sh
sudo python setup.py install
```

You can also install Conformal from PyPI:
```sh
sudo pip install conformal
```

------------------


## Usage
```sh
cf = ConformalPrediction(model_prediction, Y_test, 5, measure=SoftMax(), threshold_mode=0)
cf_prediction = cf.predict(model_prediction)
cf_accuracy = cf.evaluate(cf_prediction, Y_test)
```

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## References:

1. http://ieeexplore.ieee.org/document/4410411/