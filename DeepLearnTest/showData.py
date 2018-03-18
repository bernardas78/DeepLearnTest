import argparse
import random

#parser = argparse.ArgumentParser()
#parser.add_argument("--id", default=None, type=int,
#                    help="ID (position) of the letter to show")
#parser.add_argument("--training", action="store_true",
#                    help="Use training set instead of testing set")

#parser.add_argument("--data", default="./data",
#                    help="Path to MNIST data dir")

#args = parser.parse_args()

#print ("len(args)", len(args))
#mn = MNIST(args.data)

#if args.training:
#    img, label = mn.load_training()
#else:
#    img, label = mn.load_testing()

#if args.id:
#    which = args.id
#else:
which = random.randrange(0, len(labels))

print('Showing num: {}'.format(labels[which]))
print(mndata.display(images[which]))