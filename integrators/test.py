import os
import pickle
import matplotlib.pyplot as plt

curr_file = os.path.dirname(__file__)
netfile = curr_file + f"/../networks/swe/net_16_10.0000_100_1000_1000_10_32_MSE_30000_"

f = open(netfile, "rb")
net = pickle.load(f)

plt.semilogy(net.loss)



netfile = curr_file + f"/../networks/swe/backup/net_16_10.0000_100_1000_1000_10_32_MSE_30000_"

f = open(netfile, "rb")
net2 = pickle.load(f)

plt.semilogy(net2.loss)


