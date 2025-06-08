import json

fw = open("...jsonl", "w",encoding="utf-8")

i=-1
with open(".....json","r",encoding="utf-8") as f:
    for line in f:
        input_json = json.loads(line)
        i=i+1
        P=set()
        gold_triples=set()
        question=input_json["question"]
        # print(input_json["predicted_answers"].split("'answer':")[1])
        Q=input_json["predicted_answers"].replace("\"","").replace("{","").replace("}","").replace("'","")

        # print(Q)
        # print(json.loads(Q.strip()))
        # print(Q)
        predicted_answers=[x.strip() for x in Q.split("answer:")[1].strip().split(",")]

        # if len(Q)!=2:
        #     print(Q)
        # if len(Q)!=2:
        #     i=i+1
# print(i)
            # print(question)
        # predicted_answers=input_json["predicted_answers"].split("'answer':")[1].replace("}","").replace("'","").split(",")
        # print(predicted_answers)
        # print("--------")
        # break
        # print(json.loads(predicted_answers))
        # predicted_answers=[x.strip() for x in input_json["predicted_answers"].replace("Output","").replace(": ","").split(",")]
        #
        #
        dic_={}
        dic_["id"]=str(i)
        dic_["answers"]=predicted_answers


        fw.writelines(json.dumps(dic_) + "\n")
        fw.flush()
