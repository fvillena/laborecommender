import src.data
import src.features
import sklearn.model_selection
import json
import os

project_dir= os.path.dirname(os.path.realpath(__file__))

bags = src.data.get_bags_from_mimic()
train_bags, test_bags = sklearn.model_selection.train_test_split(bags)

test_bags_x = []
test_bags_y = []
for bag in test_bags:
    x,y = src.data.cut_bag(bag)
    test_bags_x.extend(x)
    test_bags_y.extend(y)

train_bags_filepath = os.path.join(project_dir,"data/interim/train_bags.json")
test_bags_x_filepath = os.path.join(project_dir,"data/interim/test_bags_x.json")
test_bags_y_filepath = os.path.join(project_dir,"data/interim/test_bags_y.json")

with open(train_bags_filepath,"w",encoding="utf-8") as tr, open(test_bags_x_filepath,"w",encoding="utf-8") as tex, open(test_bags_y_filepath,"w",encoding="utf-8") as tey:
    tr.write(json.dumps(train_bags,ensure_ascii=False,indent=2))
    tex.write(json.dumps(test_bags_x,ensure_ascii=False,indent=2))
    tey.write(json.dumps(test_bags_y,ensure_ascii=False,indent=2))