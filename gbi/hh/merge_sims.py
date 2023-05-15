import pickle
import torch


thetas = []
for i in range(1000):
    with open(f"allen_data/allen_theta_{i}.pkl", "rb") as handle:
        theta = pickle.load(handle)
        thetas.append(theta)
thetas = torch.cat(thetas)


allsummstats = []
for i in range(1000):
    with open(f"allen_data/allen_summstats_{i}.pkl", "rb") as handle:
        summstats = pickle.load(handle)
        allsummstats.append(torch.as_tensor(summstats, dtype=torch.float32))
allsummstats = torch.cat(allsummstats)


with open("allen_data/allen_theta.pkl", "wb") as handle:
    pickle.dump(thetas, handle)


with open("allen_data/allen_summstats.pkl", "wb") as handle:
    pickle.dump(allsummstats, handle)
