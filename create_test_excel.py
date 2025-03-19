import pandas as pd; data = {"问题": ["什么是人工智能？", "机器学习的主要应用领域有哪些？", "深度学习与传统机器学习的区别是什么？"]}; df = pd.DataFrame(data); df.to_excel("test_questions.xlsx", index=False); print("测试Excel文件已创建")
