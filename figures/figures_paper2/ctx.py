import ctypes
from unittest import result
import pandas as pd
from copy import deepcopy

pd.set_option('display.max_rows', None)

ctxdata = pd.read_csv('../states/ctxacc.csv', usecols=['PredictionTag', 'ComposeByCompositionFirst',
                      'ComposeByPotentialsFirst', 'Divergence', 'Refractory', 'CtxLoss', 'CtxAcc', 'CtxPotDist'])

ctxdata = ctxdata[ctxdata['CtxAcc'] >= 0.75]
ctxdata = ctxdata[ctxdata['CtxPotDist'] >= 0.75]
ctxdata = ctxdata[ctxdata['CtxLoss'] <= 0.2]
# ctxdata = ctxdata[lambda x: ctxdata['PredictionTag']]
# ctxdata.loc[lambda x: "MV" in x['PredictionTag']]
mv3 = deepcopy(ctxdata[ctxdata.apply(
    lambda x: True if "MV3" in (x['PredictionTag']) else False, axis=1)])
sin3 = deepcopy(ctxdata[ctxdata.apply(
    lambda x: True if "SIN3" in (x['PredictionTag']) else False, axis=1)])
saw3 = deepcopy(ctxdata[ctxdata.apply(
    lambda x: True if "SAW3" in (x['PredictionTag']) else False, axis=1)])


def printresults(tag, results):
    results.sort_values(by=['CtxLoss'], ascending=True, inplace=True)
    results.sort_values(by=['CtxPotDist'], ascending=False, inplace=True)
    results.sort_values(by=['CtxAcc'], ascending=False, inplace=True)
    print(f'\n[{tag} CtxAcc]\n', results)

    # results.sort_values(by=['CtxLoss'], ascending=True, inplace=True)
    # print(f'\n[{tag} CtxLoss]\n', results[:8])

    # results.sort_values(by=['CtxPotDist'], ascending=False, inplace=True)
    # print(f'\n[{tag} CtxPotDist]\n', results[:8])


def generatetableforoverleaf():
    def entry(conf):
        print(r"\textit{"+ conf+r"} & \scriptsize $S0\ [0,1]\ \ (1.0)$ \\\n\textbf{ } & \scriptsize $S1\ [0,1]\ \ (1.0)$ \\\n\hline")
    print("\n\n")
    print(r"\begin{table*}[htbp]")
    print(r"\caption{stuff to say}")
    print(r"\vspace*{4mm}")
    print(r"\begin{tabular*}{\hsize}{c|ccccccc}")
    print("\hline")
    print(r"\textbf{$Configurations$} && $PotFirst$ && $CompFirst$ && $Divergence$ && $RefracPeriod$ && $CtxLoss$ && $CtxAcc$ && $CtxPotDist$")
    entry("B2L3sdfsfdsfdfd")
    print("\end{tabular*}\n\label{context_data}\n\end{table*}")



printresults('sin3', sin3)
printresults('saw3', saw3)
printresults('mv3', mv3)

generatetableforoverleaf()

