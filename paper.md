---
title: 'LaboRecommender: A crazy-easy to use Python-based recommender system for laboratory tests'
tags:
  - Python
  - medical informatics
  - recommender systems
  - clinical decision support systems
authors:
  - name: Fabián Villena
    orcid: 0000-0002-8759-466X
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Independent Researcher
   index: 1
date: 02 October 2021
bibliography: paper.bib

---

# Summary

Laboratory tests play a significant role in clinical decision making because they are essential for the confirmation of diagnostics suspicions and influence medical decisions. The number of laboratory tests available to physicians in our age has been expanding rapidly due to the rapid advances in laboratory technology. To find the correct desired tests within this expanding plethora of elements, the Health Information System must provide a powerful search engine. The practitioner needs to remember the exact name of the laboratory test to select the bag of tests to order correctly. Recommender systems are platforms that suggest appropriate items to a user after learning the users' behaviour. 
A neighbourhood-based collaborative filtering recommender system library was developed to ease the development of clinical decision support systems for laboratory tests.

# Statement of need

Laboratory tests play a significant role in clinical decision making because they are essential for the confirmation of diagnostics suspicions and influence medical decisions in general. A large proportion of clinical encounters, inpatient and outpatient, requires laboratory testing. Henceforth, the value of these diagnostic procedures is significant. Literature dictates that overall, 35 % of the clinical encounters requires at least one laboratory test, and for inpatient, the proportion goes up to 98 % [@ngo_frequency_2017].

The number of different laboratory tests available to physicians in our age has been expanding very rapidly to nearly 3000 different laboratory tests, and this number will continue to increase due to the rapid advances in laboratory technology [@wians_clinical_nodate]. The Health Information System (HIS) must provide a powerful search engine to find the correct desired tests within this expanding plethora of elements. The practitioner needs to remember the exact name of the laboratory test to select the bag of tests to order correctly. 

The process of selection of the laboratory tests, taking into account the attributes described above, is time-consuming and prone to errors by misselection or omission in the clinical order entry process [@zhi_landscape_2013].

Typically laboratory tests are ordered through a logic of *order sets*, this term refers to a selection of a group of laboratory tests related to each other to answer a specific clinical question. These clinical questions are often operationalized in clinical guidelines based on the state of the art evidence to describes the best route to manage specific clinical events. Therefore is expected and suggested to order laboratory tests in bags [@chan_order_2012].

Recommender systems are platforms that suggest appropriate items to a user after learning the users' behaviour. In this work, I want to recommend to the practitioner, the addition of a laboratory test to a bag of laboratory tests, based on already added laboratory tests. These systems use information filtering to recommend information of interest to a user [@alyari_recommender_2018].

Recommender systems are a technology that has been used in a variety of fields, ranging from social media to healthcare. Specifically in medical informatics, these systems have been developed to support clinical decisions. For example, recommender systems are used in food recommendations for diabetic patients [@norouzi_mobile_2018], suggesting cardiac diagnostics [@mustaqeem_modular_2020] and suggesting where to publish medical papers [@feng_deep_2019].

The basic principle of recommender systems is that significant dependencies exist between users and items. These dependencies can be modelled in a data-driven manner through historical interaction data. Many different approaches can be used to model this process, and the most used method is *collaborative filtering*, which refers to the use of interaction from multiple users to predict future interactions of similar users. In this software, a neighbourhood-based collaborative filtering method was used [@aggarwal_recommender_2016].

A Python-based library to model recommender systems for laboratory tests was developed and released to the research community to encourage research in the area.

# References