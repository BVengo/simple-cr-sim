# Simple CR Sim
This program is designed to provide simulated imaging data with background noise, stars, and cosmic rays for the purpose of benchmarking cosmic ray excision algorithms. It has been developed as a 'pipeline' of sorts, with the ability to chain together various steps in the simulation. For example:

```py
img = (
  SImage()  # The base simulation image object
  .set_rng(...)  # Define the random number generator for reproducible results
  .set_data(np.zeros(IMG_SIZE))  # Set the initial dataset to be empty zeros
  .enable_history(...)  # Enable the saving of interim datasets from this point on, included in the final FITS file
  .add_stars(...)  # Add stars onto the image
  .add_noise(...)  # Add background noise onto the image (important this occurs _after_ stars, so the PSF properties are applied)
  .add_cosmics(...)  # Add cosmic rays (important this occurs _after_ noise, given they hit the CCD directly
  .add_to_header(...)  # Add other information into the fits header
)
```
It was planned that a function be included for spectral data, but unfortunately time constraints have prevented this.

There are three notebooks included, each of which retains the outputs from the final run-through required for the paper that this program was written for.
1. **create_img**: Generates the simulated data
2. **gen_masks**: Runs each image through the algorithms LACosmic, PyCosmic, and AstroScrappy, exporting their results as FITS files
3. **analyse_img**: Runs additional analysis against each output. Extracts average time per algorithm, a truth table, and comparison metrics
