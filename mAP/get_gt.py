import os
import shutil
classes = ["person"]

if __name__ == '__main__':
    truth = "./ground-truth"
    if os.path.exists(truth):
        shutil.rmtree(truth)
    os.mkdir(truth)
    with open("../low_test.txt") as lines:    # 训练集
        for line in lines:
            tmp_line = line.strip().split(" ")
            imagefile = os.path.split(tmp_line[0])[1]
            truthfile = truth + os.sep + imagefile.split(".")[0]+".txt"
            print(truthfile)
            truthfiles = open(truthfile, "w")
            for i, a in enumerate(tmp_line):
                if i != 0:
                    cc = classes[int(a.split(",")[-1])] + " " +" ".join([i for i in a.split(",")[0:4]])
                    truthfiles.write(cc + "\n")
            truthfiles.close()
            print("*" * 20)