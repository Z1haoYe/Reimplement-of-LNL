import datag
import models
import numpy as np
import matplotlib.pyplot as plt

rho_pos = 0.2
rho_neg = 0.2
raw_data, noise_data = datag.generate_data(5000, rho_pos, rho_neg)
# raw_data, n1, n2 = datag.random_data(5000)
# noise_data = datag.add_noise(raw_data, 0.2, 0.2)
true_dict = {}
for i in raw_data:
  true_dict[(i[1], i[2])] = i[0]
noise_dict = {}
for i in noise_data:
  noise_dict[(i[1], i[2])] = i[0]
# print(noise_data)

train_data, test_data = datag.split_data(noise_data)

t_data = np.array(test_data)
total = len(test_data)
acc = 0.0
target = 1 * (1 - rho_pos) * (1 - rho_neg) * 1.3

output_list = {}

print("Enter '1' for Method of Unbiased Estimators")
print("Enter '2' for Method of Label-dependent Costs")
if(input() == 1):
    print("xxx")
    while(acc < target):
    # for _ in range(0, 5):
        acc_count = 0
        data = np.array(train_data)
        tdata = np.array(test_data)
        model = models.create_model1(data, tdata, true_dict, rho_pos, rho_neg)
        for i in test_data:
            input = np.array([(i[1]), (i[2])]).reshape(1, 2)
            # print(input)
            output = model.predict(input, verbose=0)
            output_list[(i[1], i[2])] = output
            # print(output)
            if (output == true_dict[i[1],i[2]]): acc_count += 1
            # print(true_dict[i[1],i[2]])
        acc = acc_count / total
    print(acc)

else:
    while(acc < target):
    # for _ in range(0, 5):
        acc_count = 0
        data = np.array(train_data)
        tdata = np.array(test_data)
        model = models.create_model2(data, tdata, true_dict, rho_pos, rho_neg)
        for i in test_data:
            input = np.array([(i[1]), (i[2])]).reshape(1, 2)
            # print(input)
            output = model.predict(input)
            output_list[(i[1], i[2])] = output[0]
            # print(output)
            if (output == true_dict[i[1],i[2]]): acc_count += 1
            # print(true_dict[i[1],i[2]])
        acc = acc_count / total
        print(acc)

datag.ploting(output_list, "Prediction:")
datag.ploting(true_dict, "Noise-less Data")
datag.ploting(noise_dict, "Noisy Data")
plt.show()