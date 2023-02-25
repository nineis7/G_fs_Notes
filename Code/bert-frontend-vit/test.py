import numpy as np

def transform(data, bias):
    ret = data.reshape([64, 49, 288])
    ret = np.add(ret, bias)
    ret = ret.reshape([64, 49, 3, 3, 32])
    ret = np.transpose(ret, (2, 0, 3, 1, 4))
    print(ret.shape)
    return ret

def take(data, scalar):
    ret = np.take(data, 0, axis=0)
    print(ret.shape)
    ret = ret * scalar
    return ret

a = np.load("./golden_layer/intermediate_3_after.npy")
a1 = np.load("./golden_layer/bias_3_after.npy")
# print(a.shape)
out1 = transform(a, a1)
out1 = take(out1, 1)
# print(out1.shape)
# print(out1)

b = np.load("./golden_layer/intermediate_3_before.npy")
b1 = np.load("./golden_layer/bias_3_before.npy")
# print(b.shape)
out2 = transform(b, b1)
out2 = take(out2, 0.176777)
# print(out2.shape)
# print(out2)

print(out1 - out2)
print(np.mean(np.abs(out1 - out2)))

a = np.load("./dump/gather2_18.npy")
print(a.shape)
print(a[0])