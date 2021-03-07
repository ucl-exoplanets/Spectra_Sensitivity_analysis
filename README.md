[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4587377.svg)](https://doi.org/10.5281/zenodo.4587377)



# Peeking inside the Black Box: Interpreting Deep Learning Models for Exoplanet Atmospheric Retrievals

## Summary
This is a code repository for the paper: Peeking inside the Black Box: Interpreting Deep Learning Models for Exoplanet Atmospheric Retrievals[https://arxiv.org/abs/2011.11284]

Inside you will find detailed implementation of the 3 DNNs, and how to carry out sensitivity test for the trained model. You can find the source data file here.

### List of authors
_Kai Hou Yip, Quentin Changeat, Nikolaos Nikolaou, Mario Morvan, Billy Edwards, Ingo P. Waldmann, and Giovanna Tinetti_

# Key Requirements
- Python (3.8.8)
- Keras (2.3.1)
- Tensorflow (2.2.0)
- h5py (2.10.0)
- pandas (1.2.2)
- matplotlib (3.3.4)
- numpy (1.19.2)

To replicate the exact computing environment, please install the `pip install` packages according to `requirements.txt`.
## Installation
1. Download the code using `git clone`, or as a `zip` file
2. Install the package using `pip install . `
3. To replicate the exact computing environment: `pip install -r requirements.txt`

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
@ARTICLE{2020arXiv201111284H,
       author = {{Hou Yip}, Kai and {Changeat}, Quentin and {Nikolaou}, Nikolaos and {Morvan}, Mario and {Edwards}, Billy and {Waldmann}, Ingo P. and {Tinetti}, Giovanna},
        title = "{Peeking inside the Black Box: Interpreting Deep Learning Models for Exoplanet Atmospheric Retrievals}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Machine Learning},
         year = 2020,
        month = nov,
          eid = {arXiv:2011.11284},
        pages = {arXiv:2011.11284},
archivePrefix = {arXiv},
       eprint = {2011.11284},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201111284H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

