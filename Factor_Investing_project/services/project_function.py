from services.strategies import *
import numpy as np


def project_function(periodReturns, periodFactRet, x0= None):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    # Strategy = OLS_MVO()
    # x = Strategy.execute_strategy(periodReturns, periodFactRet)
    model_weights = [0.25,0.25,0.25,0.25]
    Strategy = Ensemble_MVO (NumObs=36, model_weights=model_weights)
    #Strategy = Lasso_MVO (NumObs=48, model_weights=model_weights)
    x, R2_Lasso, adj_R2_Lasso, R2_OLS, adj_R2_OLS, R2_FF3, adj_R2_FF3, R2_BSS , adj_R2_BSS = Strategy.execute_strategy(periodReturns, periodFactRet)
    # Strategy = PCA_MVO()
    # x, R2, adj= Strategy.execute_strategy(periodReturns, periodFactRet)
    return x

'''
def project_function(periodReturns, periodFactRet, x0=None):
    if STRATEGY_NAME == "OLS":
        Strategy = OLS_MVO()
    elif STRATEGY_NAME == "FF3":
        Strategy = FF3_MVO()
    elif STRATEGY_NAME == "Lasso":
        Strategy = Lasso_MVO()
    elif STRATEGY_NAME == "BSS":
        Strategy = BSS_MVO(K=3)
    elif STRATEGY_NAME == "Ensemble":
        Strategy = Ensemble_MVO(model_weights=[0.4,0.1,0.4,0.1])
    else:
        raise ValueError(f"Invalid strategy: {STRATEGY_NAME}")
    
    return Strategy.execute_strategy(periodReturns, periodFactRet)

'''
