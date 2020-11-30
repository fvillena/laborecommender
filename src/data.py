import pandas as pd

def get_bags_from_mimic(
    labevents: str = "https://github.com/fvillena/matbio/blob/master/data/LABEVENTS.csv?raw=true", 
    d_labitems: str = "https://raw.githubusercontent.com/fvillena/matbio/master/data/D_LABITEMS.csv"
    ) -> list:
    """
    Constructs a list of laboratory test bags from mimic tables.

    From the MIMIC tables `LABEVENTS` and `D_LABEVENTS` this function creates a list
    of bags of laboratory test (namely laboratory tests that are requested at the 
    same time).

    Parameters
    ----------
    labevents : str, default 'https://github.com/fvillena/matbio/blob/master/data/LABEVENTS.csv?raw=true'
        URL for the csv of the `LABEVENTS` MIMIC table
    d_labitems : str, default 'https://raw.githubusercontent.com/fvillena/matbio/master/data/D_LABITEMS.csv'
        URL for the csv of the `D_LABEVENTS` MIMIC table
    
    Returns
    -------
    list of list of str
        A list of list of str laboratory test bags.

    """
    labevents = pd.read_csv(labevents)
    d_labitems = pd.read_csv(d_labitems)
    data = labevents.merge(d_labitems[d_labitems.fluid.isin(["Blood","Urine"])].dropna(subset=["loinc_code"]),how="inner",on="itemid")
    data["instant"] = (pd.to_datetime(data.charttime).astype(int)/(10e6*60*10)).astype(int)
    bags = []
    for _, group in data.groupby(by=["subject_id","instant"]):
        bag = tuple(set(group.label.to_list()))
        if len(bag)>1:
            bags.append(bag)
    bags = list(set(bags))
    return bags

def cut_bag(bag: list) -> tuple:
    """
    Cuts a laboratory test bag into differnt subsections.

    From a laboratory test bag, this functions iterates over each element and in each
    element it creates a cutting point to separate the bag in two subsections.

    Parameters
    ----------
    bag : list of str
        List of laboratory tests.
    
    Returns
    -------
    x : list of str
        Left side of the cutted bag
    y : list of str
        Right side of the cutted bag
    
    Examples
    --------
    >>> bag = ["a","b","c","d"]
    >>> cut_bag(bag)
    (
        [
            ['a'],              ['a', 'b'],     ['a', 'b', 'c']
        ], 
        [
            ['b', 'c', 'd'],    ['c', 'd'],     ['d']
        ]
    )

    """
    x,y = ([],[])
    for cut in range(1,len(bag)):
        current_x = bag[:cut]
        current_y = bag[cut:]
        x.append(current_x)
        y.append(current_y)
    return (x,y)

def make_supervised_dataset(bags: list) -> tuple:
    """
    Constructs features and labels from a list of bags.
    
    From a list of bags, this function iterates over each bag and cut the bag to create
    a features bag (left side of the cutted bag) and a labels bag (right side of the 
    cutted bag). This is the function to create a testing dataset.

    Parameters
    ----------
    bags : list of list of str
        List of bags.
    
    Returns
    -------
    features : list of list of str
        List of features.
    labels : list of list of str
        List of labels.
    
    Examples
    --------
    >>> bags = [["a","b","c"],["d","e","f"]]
    >>> make_supervised_dataset(bags)
    (
        [['a'],         ['a', 'b'],     ['d'],          ['d', 'e']],
        [['b', 'c'],    ['c'],          ['e', 'f'],     ['f']]
    )

    """
    test_bags_x = []
    test_bags_y = []
    for bag in bags:
        x,y = cut_bag(bag)
        test_bags_x.extend(x)
        test_bags_y.extend(y)
    return (test_bags_x,test_bags_y)