from utils.config import Config
from models.KPFCNN import *
from dataloaders.Rail3D import Rail3DDataset, Rail3DSampler, Rail3DCollate
from utils.metrics import IoU_from_confusions, fast_confusion


from torch.utils.data import DataLoader
from os import makedirs, listdir
from os.path import exists, join
from pathlib import Path

import argparse


SUPPORTED_FILE_FORMATS = ['.ply'] # TODO '.las', '.laz',

def cloud_segment(net, test_loader, config):
    """
    Test method for cloud segmentation models
    """

    ############
    # Initialize
    ############

    # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    test_smooth = 0.95
    test_radius_ratio = 0.7
    softmax = torch.nn.Softmax(1)

    # Number of classes including ignored labels
    nc_tot = test_loader.dataset.num_classes

    # Number of classes predicted by the model
    nc_model = config.num_classes

    # Initiate global prediction over test clouds
    test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]

    # Choose to train on cpu or gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net.to(device)

    # Test saving path
    if config.saving:
        test_path = join('test', config.saving_path.split('/')[-1])
        if not exists(test_path):
            makedirs(test_path)
        if not exists(join(test_path, 'predictions')):
            makedirs(join(test_path, 'predictions'))
        if not exists(join(test_path, 'probs')):
            makedirs(join(test_path, 'probs'))
        if not exists(join(test_path, 'potentials')):
            makedirs(join(test_path, 'potentials'))
    else:
        test_path = None

    # If on validation directly compute score
    if test_loader.dataset.set == 'validation':
        val_proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in test_loader.dataset.label_values:
            if label_value not in test_loader.dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                for labels in test_loader.dataset.validation_labels])
                i += 1
    else:
        val_proportions = None

    #####################
    # Network predictions
    #####################

    test_epoch = 0
    last_min = -0.5

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(1)

    # Start test loop
    while True:
        print('Initialize workers')
        for i, batch in enumerate(test_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if i == 0:
                print('Done in {:.1f}s'.format(t[1] - t[0]))

            if 'cuda' in device.type:
                batch.to(device)

            # Forward pass
            outputs = net(batch, config)

            t += [time.time()]

            # Get probs and labels
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            s_points = batch.points[0].cpu().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            in_inds = batch.input_inds.cpu().numpy()
            cloud_inds = batch.cloud_inds.cpu().numpy()
            torch.cuda.synchronize(device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                points = s_points[i0:i0 + length]
                probs = stacked_probs[i0:i0 + length]
                inds = in_inds[i0:i0 + length]
                c_i = cloud_inds[b_i]

                if 0 < test_radius_ratio < 1:
                    mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                    inds = inds[mask]
                    probs = probs[mask]

                # Update current probs in whole cloud
                test_probs[c_i][inds] = test_smooth * test_probs[c_i][inds] + (1 - test_smooth) * probs
                i0 += length

            # Average timing
            t += [time.time()]
            if i < 2:
                mean_dt = np.array(t[1:]) - np.array(t[:-1])
            else:
                mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                print(message.format(test_epoch, i,
                                        100 * i / config.validation_size,
                                        1000 * (mean_dt[0]),
                                        1000 * (mean_dt[1]),
                                        1000 * (mean_dt[2])))

        # Update minimum od potentials
        new_min = torch.min(test_loader.dataset.min_potentials)
        print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
        #print([np.mean(pots) for pots in test_loader.dataset.potentials])

        # Save predicted cloud
        if last_min + 1 < new_min:

            # Update last_min
            last_min += 1

            # Show vote results (On subcloud so it is not the good values here)
            if test_loader.dataset.set == 'validation':
                print('\nConfusion on sub clouds')
                Confs = []
                for i, file_path in enumerate(test_loader.dataset.files):

                    # Insert false columns for ignored labels
                    probs = np.array(test_probs[i], copy=True)
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            probs = np.insert(probs, l_ind, 0, axis=1)

                    # Predicted labels
                    preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                    # Targets
                    targets = test_loader.dataset.input_labels[i]

                    # Confs
                    Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                # Regroup confusions
                C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                # Remove ignored labels from confusions
                for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                    if label_value in test_loader.dataset.ignored_labels:
                        C = np.delete(C, l_ind, axis=0)
                        C = np.delete(C, l_ind, axis=1)

                # Rescale with the right number of point per class
                C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                # Compute IoUs
                IoUs = IoU_from_confusions(C)
                mIoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * mIoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                print(s + '\n')

            # Save real IoU once in a while
            if int(np.ceil(new_min)) % 1 == 0:

                # Project predictions
                print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                t1 = time.time()
                proj_probs = []
                for i, file_path in enumerate(test_loader.dataset.files):

                    # print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)

                    # print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                    # print(test_loader.dataset.test_proj[i][:5])

                    # Reproject probs on the evaluations points
                    probs = test_probs[i][test_loader.dataset.test_proj[i], :]
                    proj_probs += [probs]

                    # Insert false columns for ignored labels
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)

                t2 = time.time()
                print('Done in {:.1f} s\n'.format(t2 - t1))

                # Show vote results
                if test_loader.dataset.set == 'validation':
                    print('Confusion on full clouds')
                    t1 = time.time()
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Get the predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                        # Confusion
                        targets = test_loader.dataset.validation_labels[i]
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print('-' * len(s))
                    print(s)
                    print('-' * len(s) + '\n')

                # Save predictions
                print('Saving clouds')
                t1 = time.time()
                for i, file_path in enumerate(test_loader.dataset.files):

                    # Get file
                    points = test_loader.dataset.load_evaluation_points(file_path)

                    # Get the predicted labels
                    preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                    # Increment each value in preds by one
                    preds += 1

                    # Save plys
                    cloud_name = file_path.split('\\')[-1]
                    test_name = join(test_path, 'predictions', cloud_name)
                    write_ply(test_name,
                                [points, preds],
                                ['x', 'y', 'z', 'preds'])
                    test_name2 = join(test_path, 'probs', cloud_name)
                    prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                                    for label in test_loader.dataset.label_values]
                    write_ply(test_name2,
                                [points, proj_probs[i]],
                                ['x', 'y', 'z'] + prob_names)

                    # Save potentials
                    pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                    pot_name = join(test_path, 'potentials', cloud_name)
                    pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                    write_ply(pot_name,
                                [pot_points.astype(np.float32), pots],
                                ['x', 'y', 'z', 'pots'])

                    # Save ascii preds
                    if test_loader.dataset.set == 'test':
                        if test_loader.dataset.name.startswith('Semantic3D'):
                            ascii_name = join(test_path, 'predictions', test_loader.dataset.ascii_files[cloud_name])
                        else:
                            ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                        np.savetxt(ascii_name, preds, fmt='%d')

                t2 = time.time()
                print('Done in {:.1f} s\n'.format(t2 - t1))

        test_epoch += 1

        # Break when reaching number of desired votes
        if test_epoch > 200:
            break

    return


def main():
    # ==================================================================================================
    #                                    Parse command line arguments
    # ==================================================================================================
    parser = argparse.ArgumentParser(description="Process a point cloud file or a directory of point cloud files.")
    parser.add_argument('-i', '--input',          type=str, required=True,  help='Path to a point cloud file or a directory containing point cloud files')
    parser.add_argument('-o', '--output',         type=str, required=False, help='Path to save the output file(s)', default='.')
    parser.add_argument('-os', '--output_suffix', type=str, required=False, help='Suffix to append to the output file(s)', default='_predicted')
    parser.add_argument('-m', '--model',          type=str, required=True, help='Path to the model file')
    
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    model_path = Path(args.model).resolve()
    
    files = []
    if input_path.is_file() and input_path.suffix in SUPPORTED_FILE_FORMATS:
        files.append(input_path)
    elif input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix in SUPPORTED_FILE_FORMATS]
    else:
        print(f"\033[91mERROR:\033[0m {input_path} is not a valid file or directory.")
        exit()

    if not model_path.exists() or not model_path.is_dir():#or not model_path.suffix == '.tar':
        print(f"\033[91mERROR:\033[0m {model_path} is not a valid model.")
        exit()

    output_path = files[0].parent if args.output == '.' else Path(args.output).resolve()
    
    if not output_path.is_dir():
        print(f"\033[91mERROR:\033[0m {output_path} is not a valid directory.")
        exit()

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

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
    dataset = Rail3DDataset(config, files, output_path, use_potentials=True)
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