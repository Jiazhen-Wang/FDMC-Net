import os
import os.path as path
import argparse
from adn.utils import \
    get_config, update_config, save_config, \
    get_last_checkpoint, add_post, Logger
from adn.datasets import get_dataset
from adn.models import ADNTrain,ADNTrain1
from torch.utils.data import DataLoader

import torch
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train a frequency-decomposed motion correction network")
    parser.add_argument("--run_name",  default= 'motion_image', help="name of the run")
    parser.add_argument("--default_config", default="config/fdmcnet.yaml", help="default configs")
    parser.add_argument("--run_config", default="runs/fdmcnet.yaml", help="run configs")
    args = parser.parse_args()

    # Get ADN options
    opts = get_config(args.default_config)
    run_opts = get_config(args.run_config)
    if args.run_name in run_opts and "train" in run_opts[args.run_name]:
        run_opts = run_opts[args.run_name]["train"]
        update_config(opts, run_opts)
    run_dir = path.join(opts["checkpoints_dir"], args.run_name)
    if not path.isdir(run_dir): os.makedirs(run_dir)
    save_config(opts, path.join(run_dir, "train_options.yaml"))
    # torch.autograd.set_detect_anomaly(True)

    # Get dataset
    def get_image(data):
        dataset_type = dataset_opts['dataset_type']
        if dataset_type == "deep_lesion":
            if dataset_opts[dataset_type]['load_mask']: return data['lq_image'], data['hq_image'], data['mask']
            else: return data['lq_image'], data['hq_image']
        elif dataset_type == "spineweb":
            return data['a'], data['b']
        elif dataset_type == "nature_image":
            return data["artifact"], data["no_artifact"]
        elif dataset_type == "motion_image":
            return data["motion"], data["clean"]
        else:
            raise ValueError("Invalid dataset type!")

    dataset_opts = opts['dataset']
    train_dataset = get_dataset(**dataset_opts)
    train_loader = DataLoader(train_dataset,
        batch_size=opts["batch_size"], num_workers=opts['num_workers'], shuffle=True)
    train_loader = add_post(train_loader, get_image)

    # Get checkpoint
    if opts['last_epoch'] == 'last':
        checkpoint, start_epoch = get_last_checkpoint(run_dir)
    else:
        start_epoch = opts['last_epoch']
        checkpoint = path.join(run_dir, "net_{}".format(start_epoch))
        if type(start_epoch) is not int: start_epoch = 0

    # Get model
    model = ADNTrain(opts['learn'], opts['loss'], **opts['model'])
    model1 = ADNTrain1(opts['learn'], opts['loss'], **opts['model'])
    if opts['use_gpu']:
        print("=> use gpu id: '{}'".format(opts['gpus']))
        os.environ["CUDA_VISIBLE_DEVICES"] = opts['gpus']
        model.cuda()
        model1.cuda()
        if path.isfile(checkpoint):
            model1.resume(checkpoint)
            model1.netT = model1._get_target_network()

        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # Get logger
    logger = Logger(run_dir, start_epoch, args.run_name)
    logger.add_loss_log(model.get_loss, opts["print_step"], opts['window_size'])
    logger.add_iter_visual_log(model.get_visuals, opts['visualize_step'], "train_visuals")
    logger.add_save_log(model.save, opts['save_step'])


    # Test the model
    # model.evaluate(train_loader, psnr)

    # Train the model
    for epoch in range(start_epoch, opts['num_epochs']):
        for data in logger(train_loader):
            model.optimize(*data, model1.netT)
        model.update_lr()
