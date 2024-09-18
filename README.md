# Siemens TWIX Recon
Perform GRAPPA-ND reconstruction of Siemens TWIX data.
Twix Recon is a command-line interface (CLI) tool designed for reconstructing Siemens TWIX raw data files. It processes .dat files, applies gGRAPPA reconstruction, and saves the results in NIfTI format. This tool supports GPU acceleration and offers options for tuning reconstruction parameters.

## Features

- **Reconstruct Siemens TWIX raw data files (.dat)**
- **Customize kernel size for reconstruction**
- **Batch processing of reconstruction windows**
- **Adjust regularization strength**
- **Enable GPU acceleration**

## Clone
You need to clone the repository by using the following command:
```bash
  git clone git@github.com:mbertrait/siemens_twix_grappa_recon.git
```

## Install
After cloning, you just need to install the tool:
```bash
  cd siemens_twix_recon
  pip install .
```

## Usage

After installation, a new command will be available: `twix_reco`.
Run it from the command line with the required and optional arguments:

```bash
 twix_reco [scan_path] [options]
```

### Arguments

- `scan_path` (required): Path to the TWIX raw data file (.dat) or directory containing `.dat` files.

### Optional Arguments

- `-o`, `--output_dir`: Path to the directory where the reconstructed image will be saved. Defaults to the current working directory.
- `-k`, `--kernel_size`: Kernel size as a comma-separated list of three integers (ky, kz, kx). Default is `4,4,5`.
- `-b`, `--batch_size`: Number of reconstruction windows to process at once. Default is `1`.
- `-l`, `--lambda`: Regularization term strength for pseudo-inverse. Default is `1e-4`.
- `--gpu`: Enable GPU acceleration.
- `--dcm_range`: Rescale the intensities of the reconstructed scan to DICOM range.
- `--gpu_mode`: Specify GPU mode - `all`, `application`, or `estimation`. Default is `all`.

### Examples

1. Reconstruct a single `.dat` file and save the result in the current directory:

    ```bash
     twix_reco /path/to/data.dat
    ```

2. Reconstruct files in a directory, set a specific output directory, and use GPU acceleration:

    ```bash
    twix_reco /path/to/data_directory -o /path/to/output_directory --gpu
    ```

3. Set a custom kernel size (ky,kz,kx): (5,5,3), batch size: 2, and adjust regularization: 1e-5:

    ```bash
    twix_reco /path/to/data.dat -k 5,5,3 -b 2 -l 1e-5
    ```

4. Rescale intensities to DICOM range and use a specific GPU mode: here GPU only for GRAPPA kernel weights estimation:

    ```bash
    twix_reco /path/to/data.dat --dcm_range --gpu --gpu_mode estimation
    ```
