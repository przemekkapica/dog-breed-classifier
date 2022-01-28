import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--image-dim', dest='image_dim', default=40, type=int)
parser.add_argument('-lr', '--learning-rate', dest='learning_rate', default=0.001, type=float)
parser.add_argument('-e', '--epochs', dest='epochs', default=100, type=int)
parser.add_argument('-o', '--optimizer', dest='optimizer', default='Adam', type=str)
parser.add_argument('-l', '--epsilon', dest='epsilon', default=1e-07, type=float)
parser.add_argument('-k', '--kernel-size', dest='kernel_size', default=3, type=int)
parser.add_argument('-d', '--dense-neurons', dest='dense_neurons', default=80, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    print(args.image_dim)
    print(args.learning_rate)
    print(args.epochs)
    print(args.epsilon)
    print(args.optimizer)
    print(args.kernel_size)
    print(args.dense_neurons)