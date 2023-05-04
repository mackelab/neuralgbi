import pickle
import torch


thetas = []
for i in range(100):
    with open(f"data/theta_{i}.pkl", "rb") as handle:
        theta = pickle.load(handle)
        thetas.append(theta)
thetas = torch.cat(thetas)


allsummstats = []
for i in range(100):
    with open(f"data/summstats_{i}.pkl", "rb") as handle:
        summstats = pickle.load(handle)
        allsummstats.append(torch.as_tensor(summstats, dtype=torch.float32))
allsummstats = torch.cat(allsummstats)


with open("data/theta.pkl", "wb") as handle:
    pickle.dump(thetas, handle)


with open("data/summstats.pkl", "wb") as handle:
    pickle.dump(allsummstats, handle)
