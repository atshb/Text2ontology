import pandas as pd

ex1 =  [("video cable","hdmi video cable"    ,1),
        ("video cable","rgb video cable"     ,1),
        ("hdmi video cable","rgb video cable",3)]

item = ["animal","mammal",
          "bird","reptile",
          "ostrich","chicken",
          "eagle","dog",
          "bull","tiger",
          "turtle","crocodile","reptile"]

ex2 = []
for i in item:
    for j in item:
        ex2.append((i, j, 0))


pd.DataFrame(ex1).to_csv('../result/ex1.csv')
pd.DataFrame(ex2).to_csv('../result/ex2.csv')
print("成功！")
