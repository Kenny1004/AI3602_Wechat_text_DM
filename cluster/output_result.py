import json
result_path="/home/zlhu/data2/zlhu/DM_project/data/withpf/result.json"
with open(result_path, "r") as fin:
    print(json.load(fin))