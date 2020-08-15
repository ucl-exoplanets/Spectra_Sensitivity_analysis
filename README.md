# Interpreting deep learning models for exoplanet atmospheric retrievals
_Work in Progress_


## Summary
This is a code repository for the paper: Peeking inside the Black Box: Interpreting Deep Learning Models for Exoplanet Atmospheric Retrievals
Inside you will find detailed implementation of the 3 DNNs, and how to carry out sensitivity test for the trained model. You can find the source data file here.

### List of authors
_Kai Hou Yip, Quentin Changeat, Nikolaos Nikolaou, Mario Morvan, Billy Edwards, Ingo P. Waldmann, and Giovanna Tinetti_

## How to use it
There are two ways to use the code:

### Model Training + Sensitivity test
For those who would like to train a model and use the sensitivity test, you can run the following command in the terminal: 

```
python runfile.py --config 'config.yaml' --epochs 100 --batch_size 64 --lr 0.01
```
Finer settings are available to edit via config.yml. 

### Sensitivty test only
For those who would like to only apply sensitivity test on their own trained model , you may do so by:
```
import sensitivity
test_result = sensitivity.compute_sensitivty_org(
    model,  
    y_test,
    org_spectrum,
    org_error,
    y_data_mean,
    y_data_std,
    gases,
    no_spectra,
    repeat,
    x_mean,
    x_std,
    abundance)
```
Description of model arguments:

**model**: Trained keras model, make sure you can use .predict method of the model. If you have trained other models via Tensorflow or PyTorch, you can change the relevant line ( model.predict) to the appropiate equivalent method, as long as it can take in input and output numpy arrays.

**org_spectrum**: original spectra (without preprocessing such as standardising )from your test set, Shape: (N x wl_bins).

**org_error**: corresponding error for each wavelength bins (wl_bins) Shape: (N xx wl_bins)

**x_mean**: Overall mean for spectrum, used for preprocessing purposes. Shape=(1,)

**x_std**: Overall s.d. for spectrum, used for preprocessing purposes. Shape=(1,)

**y_test**: Ground truth AMPs values, for sorting abundance only ( only used when abundance is not None).

**y_data_mean**: Overall mean for each AMP, used for preprocessing purposes.  Shape=(number of AMPs)

**y_data_std**: Overall s.d. for each AMP, used for preprocessing purposes.  Shape=(number of AMPs)

**gases**: For pre-selection only, if None, it will look for label in constants.py .

**no_spectra**: Number of example spectra to use (default= 100)

**repeat**: Number of shuffling for each spectrum (default= 500).

**abundance**: only include spectra within certain log abundances range , currently it will apply to all the gases. If None, every spectra will be considered.

The normalisation factors (mean , s.d. ) are needed in order to normalise the shuffled spectrum before going into the model. Readers might need to change the normalisation method if they are using a different preprocessing procedure. 

## Sample Figures

## Citation
If our tool has been useful to your research/work, please kindly cite us:

## License
