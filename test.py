import ast

line = '{"code":"OK","worldId":1,"runId":"70003","reward":-10000,"scoreIncrement":-0,"newState":null}'

res = ast.literal_eval(line)
print(res)