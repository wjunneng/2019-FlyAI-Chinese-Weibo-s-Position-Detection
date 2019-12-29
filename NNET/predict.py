# -*- coding: utf-8 -*
"""
实现模型的调用
"""
from flyai.dataset import Dataset
from NNET.model import Model

data = Dataset()
model = Model(data)

# p = model.predict(TARGET='IphoneSE', TEXT='已经等不及想要去看看这款手机了')
# print(p)

p = model.predict_all(
    [{"TARGET": "深圳禁摩限电",
      "TEXT": "#深圳禁摩限电# 自行车、汽车也同样会引发交通事故——为何单怪他们？（我早就发现：交通的混乱，反映了“交管局”内部的混乱！）必须先严整公安交管局内部！——抓问题的根本！@深圳交警@中国政府网@人民日报"},
     {"TARGET": "俄罗斯在叙利亚的反恐行动",
      "TEXT": "我教普大帝几招。 出地面部队灭了土库曼人，赶IS进土耳其。土耳其不是说土库曼人是他们亲人吗。然后大力扶持库而德人独立建国。土有几百万库而德人，就算不灭土也要挖一块出来。东突大本营早该收拾"},
     {"TARGET": "IphoneSE",
      "TEXT": "iPhone SE印度没人要：贵的离谱"},
     {"TARGET": "开放二胎",
      "TEXT": "#姚晨怀二胎#开始借着儿子大肆洗白了，这个出轨还倒打一耙黑前夫的贱婢"}])
print(p)