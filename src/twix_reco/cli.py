import argparse
import os

from .twix import SiemensTwixReco


def parse_kernel_size(value):
    try:
        kernel_size = tuple(map(int, value.split(',')))
        if len(kernel_size) == 3:
            return kernel_size
        else:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Kernel size must be a comma-separated list of three integers, e.g., 5,5,3"
        )


def main():
    default_output_path = os.path.join(os.getcwd())
    parser = argparse.ArgumentParser(description="Twix Recon")

    parser.add_argument(
        'scan_path',
        type=str,
        help='The twix raw data path (.dat)'
    )

    parser.add_argument(
        '-o', '--output_dir', 
        type=str,
        default=default_output_path,
        help="Path to the directory where to save the reconstructed image."
    )

    parser.add_argument(
        '-k', '--kernel_size', 
        type=parse_kernel_size, 
        default=(4, 4, 5),  # Default kernel size
        help="Kernel size as a comma-separated list of three integers, in the ordr: ky, kz, kx e.g., 5,5,3 (default: 4,4,5)"
    )

    parser.add_argument(
        '-b', '--batch_size', 
        type=int, 
        default=1,
        help="The number of reconstruction window to process at once. (default: 1)"
    )

    parser.add_argument(
        '-l', '--lambda', 
        type=float, 
        default=1e-4,
        help="Pseudo-inverse regularization term strength"
    )

    parser.add_argument(
        '--gpu', 
        action='store_true',
        help="Enable GPU acceleration"
    )

    parser.add_argument(
        '--dcm_range', 
        action='store_true',
        help="Rescale the intensities of the reconstructed scan to DICOM ones."
    )

    parser.add_argument(
        '--gpu_mode', 
        type=str, 
        choices=['all', 'application', 'estimation'],
        default='all',
        help="Specify GPU mode: 'all', 'application', or 'estimation' (default: 'all'). \n\"all\" means both estimation and application of GRAPPA kernel is GPU based."
    )

    args = parser.parse_args()

    reco = SiemensTwixReco(
        filepath=args.scan_path
    )
    reco.runReco()
    reco.saveToNifTI(os.path.join(args.output_dir, 'reco'), to_dicom_range=args.dcm_range)


if __name__ == "__main__":
    main()