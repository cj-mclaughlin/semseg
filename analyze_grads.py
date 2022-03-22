import pickle
import numpy as np

from scipy.spatial.distance import cosine

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

grads = pickle.load(open("grads_f8_trained.pkl", "rb"))

grad_list = []
grad_names = []

samples = 101

cosine_similarity = {layer: {} for layer in grads.keys()}

for task in grads.keys():
    grads[task] = np.asarray(grads[task])[:samples].reshape(samples, -1)

for task1 in grads.keys():
    for task2 in grads.keys():
        similarities = []
        similarity = None
        if task1 == task2:
            similarity = 1

        else:
            for sample in range(samples):
                g0 = grads[0][sample]
                g1, g2 = grads[task1][sample], grads[task2][sample]

                if task1 != 0:
                    g1g0 = g1.dot(g0)
                    if g1g0 < 1:
                        print("n")
                        g1 -= (g1g0) * (g0 / np.linalg.norm(g0))
                
                if task2 != 0:
                    g2g0 = g2.dot(g0)
                    if g2g0 < 1:
                        print("n")
                        g2 -= (g2g0) * (g0 / np.linalg.norm(g0))

                similarity = 1 - cosine(grads[task1][sample], grads[task2][sample])
                similarities.append(similarity)
            
            similarity = np.min(similarities)
            # similarity over mean gradient
            # similarity = 1 - cosine(np.mean(grads[task1], axis=0), np.mean(grads[task2], axis=0) )

        if task1 not in cosine_similarity.keys():
            cosine_similarity[task1] = {task2: similarity}
        else:
            cosine_similarity[task1][task2] = similarity

        if task2 not in cosine_similarity.keys():
            cosine_similarity[task2] = {task1: similarity}
        else:
            cosine_similarity[task2][task1] = similarity

l1 = pd.DataFrame.from_dict(cosine_similarity)
sns.heatmap(l1, annot=True)
plt.title("Cosine Similarity (F8 Trained)")
plt.show()

# l2 = pd.DataFrame.from_dict(cosine_similarity['layer2'])
# sns.heatmap(l2, annot=True)
# plt.title("Cosine Similarity (Layer2 ResNet)")
# plt.show()

# l3 = pd.DataFrame.from_dict(cosine_similarity['layer3'])
# sns.heatmap(l3, annot=True)
# plt.title("Cosine Similarity (Layer3 ResNet)")
# plt.show()

# l4 = pd.DataFrame.from_dict(cosine_similarity['layer4'])
# sns.heatmap(l4, annot=True)
# plt.title("Cosine Similarity (Layer4 ResNet)")
# plt.show()



# for i in range(len(grad_list)):
#     for j in range(len(grad_list)):
#         if i == j:
#             similarity = 1
#         else:
