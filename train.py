import src.model
import json
import os

project_dir= os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(project_dir,"data/interim/train_bags.json")) as j:
    train_bags = json.load(j)
with open(os.path.join(project_dir,"data/interim/test_bags.json")) as j:
    test_bags = json.load(j)

labo_recommender = src.model.LaboRecommender()
labo_recommender.fit(train_bags)

print(labo_recommender.predict(test_bags))