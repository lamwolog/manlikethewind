# manlikethewind
## （1）
用户基本信息：姓名、年龄、性别、职业、联系方式、常住城市。
购车偏好：意向车型（轿车 / SUV 等）、品牌倾向、预算范围、动力类型（燃油 / 电动等）、车身颜色偏好、座位数需求。
消费行为：过往购车次数、置换 / 全款 / 贷款选择、年均行驶里程、保养支出习惯、关注促销活动频率、线上线下购车渠道倾向。
（2）
常见评估数据参数指标有准确性、完整性、一致性、时效性、有效性。
数据审核流程：先制定计划，接着预处理数据，再实施审核，记录反馈问题，整改后复查，最后撰写报告。
审核方法有人工审核、逻辑检查、对比分析、统计分析、程序审核，可依数据特点灵活选用。

## 2、数据库查询
（2）select * from t_s;
（3）select * from t_s where salary is not null;
（4）select * from t_s inner join t_r on t_r.id=t_s.no;

## 3、水果分类
(2)#以下1行需要背 
from torchvision import datasets
#划分数据集：
folder=dataset.ImageFolder(root='C:/水果种类智能训练考试文件/水果图片',transform=trans_compose); n=len(folder);n1=int(n*0.8);n2=n-n1;
train,test=random_split(folder,[n1,n2]);
（3）device = torch.device("cpu");  batchX= batchX.to(device); batchY = batchy.to(device)
model = model.to(device);#清零梯度optimizer.zero_grad()
#前向传播outputs = model(batchX) #计算损失loss = lossf(outputs, batchY)
#反向传播loss.backward()# 更新参数optimizer.step()
计算准确率preds = torch.argmax(outputs,dim=1)metricsf.update(preds,batchY)
#每个 epoch 结束后打印损失和准确率epoch loss= loss.item()
epoch accuracy =metricsf.compute()print(f'Epoch {i + 1},Loss: {epoch loss:.4f}, Accuracy: {epoch accuracy: .4f}')
重置评估指标
metricsf.reset();torch.save(model.state dict(),'2-2model test.pth')
print("模型已保存为 2-2model test.pth")
## 4、神经网络
（2）class MyNet(nn.Module):def init (self):super(MyNet, self). init ();
self.fc1 = nn.Linear(287,128);self.bn1 = nn.BatchNorm1d(128);self.relu1 = nn.ReLU();
self.fc2 = nn.Linear(128,256);self.bn2 = nn.BatchNorm1d(256);self.relu2 = nn.ReLU();
self.fc3 = nn.Linear(256,1);
def forward(self, x):x = self.fc1(x);x = self.bn1(x);x = self.relu(x);
x = self.fc2(x);x = self.bn2(x);x = self.relu(x);out = self.fc3(x);
return out;
## 5、数据采集培训方法（注意修改不可雷同）
（1）基础认知：明晰数据采集概念、重要性与应用场景。
方法技巧：讲授多种采集方式，涵盖网络、传感器等，分享实操窍门。
工具运用：熟练掌握 Excel、Python 等工具用于数据获取与整理。
（2）常见问题及解决方法（选2条，不可雷同）
目标不明确
问题：未清晰界定采集数据的用途与范围，导致收集大量无关数据，遗漏关键信息。比如市场调研时，不清楚要分析用户哪类消费行为，盲目收集。
解决方法：项目启动前，组织跨部门会议，与业务、分析团队深入沟通，基于业务需求和分析目的，详细梳理数据需求清单，明确数据用途、范围、字段及预期成果。
隐私与合规问题
问题：采集敏感个人信息未获授权，或违反行业法规，面临法律风险。
解决方法：设立数据合规官，负责解读法规政策；采集前向用户明确告知数据用途、范围、存储方式，获用户同意；加密敏感数据，遵循 “最小必要” 原则采集。
