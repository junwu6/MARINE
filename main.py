import argparse
from inits import *
from model import run_model
from utils import load_data
from evaluation import node_classification


def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run MARINE (unsupervised embedding).")

    parser.add_argument('--input', nargs='?', default='citeseer',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='graph.embeddings',
                        help='Output node embeddings of the graph')

    parser.add_argument('--dimension', type=int, default=128,
                        help='Embedding dimension. Default is 128.')

    parser.add_argument('--lamb', type=float, default=1.0,
                        help='Parameter lambda in objective. Default is 1.0.')

    parser.add_argument('--eta', type=float, default=0.005,
                        help='Parameter lambda in objective. Default is 0.005.')

    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='Learning rate for Adam. Default is 5e-3.')

    parser.add_argument('--epoch', type=int, default=20,
                        help='Training epochs. Default is 20.')

    parser.add_argument('--gpu_fraction', type=float, default=0.20,
                        help='Memory usage of the GPU. Default is 0.20.')

    parser.add_argument('--batchsize', type=int, default=1024,
                        help='Size of one mini-batch. Default is 1024.')

    parser.add_argument('--print_every_epoch', type=int, default=1,
                        help='Print the objective every k epochs. Default: 1.')
    return parser.parse_args()


def main():
    args = parse_args()
    adj, edges, feats, labels = load_data(args.input)
    embeddings = run_model(edges, adj, feats, args.lamb, args.eta, args.dimension, args.learning_rate, args.epoch,
                           gpu_fraction=args.gpu_fraction,
                           batchsize=args.batchsize,
                           print_every_epoch=args.print_every_epoch,
                           scope_name='default')
    # save the embeddings
    np.savetxt(args.output, embeddings, delimiter=',')

    # performance evaluation
    indice = np.random.permutation(labels.shape[0])
    acc = node_classification(embeddings, labels, indice)
    print("Node classification: ACC={}".format(acc))


if __name__ == "__main__":
    main()
