
import pickle

s=0
for i in range(5):
    file='trainHistoryDict_'+str(i)
    with open(file, "rb") as f:   # Unpickling
        hist = pickle.load(f)
        print(max(hist['val_acc']))
        s+=max(hist['val_acc'])
print('avg:')
print(s/5)
