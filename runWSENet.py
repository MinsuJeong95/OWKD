import Training
import Validation
import validationAcc
import Test
import calculate.APCalculate as APCalculate
import calculate.RankCalculate as RankCalculate


def run(args):
    for Fold in args.Fold:
        Training.training(args, Fold)
        Validation.validation(args, Fold)
        validationAcc.accCalculate(args, Fold)
        Test.test(args, Fold)

        APCalculate.apCalculate(args, Fold)
        RankCalculate.rankClaculate(args, Fold)