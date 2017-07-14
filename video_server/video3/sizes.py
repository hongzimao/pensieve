import os, sys

chunks = os.listdir('.')
sizes = []

for i in range(1,98):
	size = os.path.getsize(str(i) + '.m4s')
	sizes.append(size)
print sizes

