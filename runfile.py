import argparse
from train_model import run_DNN

parser = argparse.ArgumentParser(description='Parser for run_model.py.')

parser.add_argument('--config', type=str, nargs='?', default='config.yaml',
                    help='path to config file')
parser.add_argument('--batch_size', type=int, nargs='?', default=64,
                    help='batch size for the DNN')
parser.add_argument('--lr', type=float, nargs='?', default=0.001,
                    help='initial learning rate for the DNN')
parser.add_argument('--epochs', type=int, nargs='?', default=50,
                    help='Number of training epochs')

args = parser.parse_args()
run_DNN(config_path=args.config,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        )
