from argparse import ArgumentParser
from visualisation import create_fig_training

parser = ArgumentParser(description="Command you can run for this model", epilog='\n')

parser.add_argument('command', type=str, description='Command you want to run')
parser.add_argument('--checkpoint', type=str, default=None, help='need for visualize command')
parser.add_argument('--name_fig', type=str, default=None, help='name of the figure')

args = vars(parser.parse_args())

if args['command'] == 'visualize':
    assert args.get('checkpoint') is not None, 'Some options are missing: --checkpoint'
    create_fig_training(args['checkpoint'], args.get('name_fig', ''))