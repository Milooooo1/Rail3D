from utils.config import Config
from pathlib import Path
from models.KPFCNN import *
from datasets.Rail3D import Rail3DDataset, Rail3DSampler, Rail3DCollate
from torch.utils.data import DataLoader
import argparse


SUPPORTED_FILE_FORMATS = ['.las', '.laz', '.ply']

def cloud_segment(net, test_loader, config):
    pass


def main():
    # ==================================================================================================
    #                                    Parse command line arguments
    # ==================================================================================================
    parser = argparse.ArgumentParser(description="Process a point cloud file or a directory of point cloud files.")
    parser.add_argument('-i', '--input',          type=str, required=True,  help='Path to a point cloud file or a directory containing point cloud files')
    parser.add_argument('-o', '--output',         type=str, required=False, help='Path to save the output file(s)')
    parser.add_argument('-os', '--output_suffix', type=str, required=False, help='Suffix to append to the output file(s)', default='_predicted')
    parser.add_argument('-m', '--model',          type=str, required=False, help='Path to the model file')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)
    
    files = []
    if input_path.is_file() and input_path.suffix in SUPPORTED_FILE_FORMATS:
        files.append(input_path)
    elif input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix in SUPPORTED_FILE_FORMATS]
    else:
        print(f"\033[91mERROR:\033[0m {input_path} is not a valid file or directory.")
        exit()

    if not model_path.exists() or not model_path.is_file() or not model_path.suffix == '.tar':
        print(f"\033[91mERROR:\033[0m {model_path} is not a valid model file.")
        exit()



    # ==================================================================================================
    #                                    Load the model
    # ==================================================================================================

    # Set up config
    config = Config()
    config.load(model_path)

    config.in_radius = 3
    config.validation_size = 50
    config.input_threads = 2

    # Set up dataloader
    dataset = Rail3DDataset(config, set='test', use_potentials=True)
    sampler = Rail3DSampler(dataset)
    collate_fn = Rail3DCollate

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             sampler=sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    sampler.calibration(data_loader, verbose=True)

    net = KPFCNN(config, dataset.label_values, dataset.ignored_labels)

    cloud_segment(net, data_loader, config)

if __name__ == "__main__":
    main()