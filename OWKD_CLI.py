import runOWKD
import argparse

parser = argparse.ArgumentParser()

# Default args
parser.add_argument('--DBPath', type=str, default='E:\\JMS\\thirdPaper\\thermalDB')
parser.add_argument('--inputDB', type=str, default='DBPerson-Recog-DB1_thermal')
# parser.add_argument('--inputDB', type=str, default='SYSU-MM01_thermal')

parser.add_argument('--Fold', type=str, default='Fold1,Fold2')
parser.add_argument('--modelType', type=str, default='original,attention')

# init training args0
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--trainBatchSize', type=int, default=8)
parser.add_argument('--numEpoch', type=int, default=3)

parser.add_argument('--GCE_lr', type=float, default=1e-3)
parser.add_argument('--GCE_wd', type=float, default=1e-3)

# init validation args
parser.add_argument('--valBatchSize', type=int, default=8)

# init test args
parser.add_argument('--testBatchSize', type=int, default=256)

args = parser.parse_args()


def main():
    args.Fold = args.Fold.replace(" ", "").split(',')
    runOWKD.run(args)


if __name__ == "__main__":
    main()
