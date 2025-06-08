import json

fw = open("...jsonl", "w",encoding="utf-8")

i=-1
with open("...json","r",encoding="utf-8") as f:
    for line in f:
        input_json = json.loads(line)
        i=i+1
        dic_={}
        dic_["id"]=str(i)
        answers=input_json["answer"]

        A=[]
        for k, v in answers.items():
            A.append(v)
        dic_["answers"] =A

        # dic_["answers"]=[x for x in input_json["response"].strip().split(",")]


        fw.writelines(json.dumps(dic_) + "\n")
        fw.flush()

