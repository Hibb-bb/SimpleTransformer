score = [2, 4.3, 4.3, 4.3, 3.7, 4.3, 4, 4, 3.7, 4, 4.3, 2.3, 4.3, 4.3, 4, 4, 4.3, 4.3, 4, 4, 4.3, 3.7, 3, 4, 4.3, 4.3, 4.3, 3.7, 4]
w = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 1, 3, 3, 1, 2, 3, 1, 2, 3, 3, 3, 3, 3] 
# 4.3

score=[4, 4, 2, 4 ,4, 4, 4, 4, 4, 4, 4, 4, 4, 4 ,4 ,4, 4, 3, 4, 4, 4, 4, 4, 4, 4]
w=[3, 3, 3, 2, 3, 3, 2, 2, 3, 1, 3, 3, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3, 3, 3, 3]
# 4.0 senior/junior

score=[2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4]
w = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 1, 3, 3, 1, 2, 3, 1, 2, 3, 3, 3, 3, 3] 
# 4.0 major
print(len(score), len(w))
assert len(score) == len(w)

s = 0
c = 0

for i in range(len(score)):
	s += score[i]*w[i]
	c += w[i]

print(s/c)