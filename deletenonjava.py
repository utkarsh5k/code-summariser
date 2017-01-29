import os
import sys

project_folder = sys.argv[1]

for root, subdirs, files in os.walk(project_folder):
	for f in files:
		if ".java" not in f:
			file_path = os.path.join(root, f)
			os.remove(file_path)
