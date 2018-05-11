with open('./output__.txt', 'r+') as f:
    lines = f.readlines()

a = []
for line in lines:
    a.append(line.strip().split())

for i in range(len(a)):
    for j in range(len(a[i])):
        if a[i][j] == "PAD":
            a[i][j] = "<PAD>"
        elif a[i][j] == "UNK":
            a[i][j] = "<UNK>"
b = []
for row in a:
    b.append([" ".join(row)])
print(b)
with open('./output_fix.txt', 'w') as f:
    for row in b:
        f.writelines(row)
        f.write('\n')
        
