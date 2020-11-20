import pandas as pd

def get_bags_from_mimic(
    labevents = "https://github.com/fvillena/matbio/blob/master/data/LABEVENTS.csv?raw=true", 
    d_labitems = "https://raw.githubusercontent.com/fvillena/matbio/master/data/D_LABITEMS.csv"
    ):
    labevents = pd.read_csv(labevents)
    d_labitems = pd.read_csv(d_labitems)
    data = labevents.merge(d_labitems[d_labitems.fluid.isin(["Blood","Urine"])].dropna(subset=["loinc_code"]),how="inner",on="itemid")
    data["instant"] = (pd.to_datetime(data.charttime).astype(int)/(10e6*60*10)).astype(int)
    bags = []
    for name, group in data.groupby(by=["subject_id","instant"]):
        bag = tuple(set(group.label.to_list()))
        if len(bag)>1:
            bags.append(bag)
    bags = list(set(bags))
    return bags

def cut_bag(bag):
    x,y = ([],[])
    for cut in range(1,len(bag)):
        current_x = bag[:cut]
        current_y = bag[cut:]
        x.append(current_x)
        y.append(current_y)
    return (x,y)