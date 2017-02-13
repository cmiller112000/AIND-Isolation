import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("results.csv","w") as output:
        print("heuristic,id_improved_pct_win,student_pct_win,student_wins,id_improved_wins", file=output)
        for name in glob.glob('results_game_agent.py_*.out'):
            heuristic = name.replace('results_game_agent.py_','').replace('.out','')
            heuristic = re.sub(r"_\d+_\d+","",heuristic)
            with open(name,'r') as f:
                for l in f:
                    if "ID_Improved" in l:
                        l = re.sub(r"ID_Improved\s+","",l)
                        idpct = float(re.sub(r"\%\s*$","",l))
                        out=str(idpct)
                    elif "Student" in l:
                        l = re.sub(r"Student\s+","",l)
                        spct = float(re.sub(r"\%\s*$","",l))
                        swin=0
                        idwin=0
                        if spct >= idpct:
                            swin=1
                        else:
                            idwin=1
                        out = out + "," + str(spct) + "," + str(swin) + "," + str(idwin)
                        print(heuristic+","+out, file=output)
    df=pd.read_csv("results.csv",dtype={"id_improved_pct_win":"float","student_pct_win":"float","student_wins":"int","id_improved_wins":"int"})
    df_pct=df[["heuristic","id_improved_pct_win","student_pct_win"]]
    df_pct.set_index(["heuristic"])
    df_wins=df[["heuristic","student_wins","id_improved_wins"]]
    df_wins.set_index(["heuristic"])

    dfagg_pct=df_pct.groupby("heuristic").agg([np.min,np.mean,np.max,np.var])
    dfagg_wins=df_wins.groupby("heuristic").mean()
    df_pct1=df_pct[df_pct.heuristic=='edgemoves_offense']
    df_pct1.set_index(["heuristic"])
    df_pct1.reset_index(drop=True, inplace=True)
    p1=df_pct1.plot(title="EdgeMoves Offense heuristic",rot=0).legend(bbox_to_anchor=(1,0.5),loc="center left",title="").figure.savefig("p1.png",bbox_inches="tight")
    df_pct2=df_pct[df_pct.heuristic=='onedge_offense']
    df_pct2.set_index(["heuristic"])
    df_pct2.reset_index(drop=True, inplace=True)
    p2=df_pct2.plot(title="On Edge Offense heuristic",rot=0).legend(bbox_to_anchor=(1,0.5),loc="center left",title="").figure.savefig("p2.png",bbox_inches="tight")
    df_pct3=df_pct[df_pct.heuristic=='onedge']
    df_pct3.set_index(["heuristic"])
    df_pct3.reset_index(drop=True, inplace=True)
    p3=df_pct3.plot(title="On Edge heuristic",rot=0).legend(bbox_to_anchor=(1,0.5),loc="center left",title="").figure.savefig("p3.png",bbox_inches="tight")
    p4=dfagg_pct.plot.bar(title="Average Tournament Win Percentage",rot=0).legend(bbox_to_anchor=(1,0.5),loc="center left",title="").figure.savefig("p4.png",bbox_inches="tight")
    p5=dfagg_wins.plot.bar(title="Average Total Tournament Wins",rot=0).legend(bbox_to_anchor=(1,0.5),loc="center left",title="").figure.savefig("p5.png",bbox_inches="tight")

if __name__ == "__main__":
    main()

