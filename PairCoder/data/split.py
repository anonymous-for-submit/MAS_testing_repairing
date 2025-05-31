import json
def code_split(func):
    '''
    Split code into signature, comment and function body
    '''
    func = func.replace("\r\n", "\n")
    before_func = func[:func.rfind("def ")]
    code = func[func.rfind("def "):]

    is_comment = False
    comments = []
    
    statements = code.split("\n")
    for s_idx, s in enumerate(statements):
        s = s.strip()
        if s.startswith("def"):
            signature = statements[:s_idx+1]
            method_name = s.split("def ")[1].split("(")[0]
            func_body_idx = s_idx+1
            tmp_statement = statements[func_body_idx].strip()
            if not tmp_statement.startswith("'''"):
                break
        elif s.startswith("'''") and not is_comment:
            is_comment = True

        elif is_comment:
            if s.startswith("'''"):
                is_comment = False
                func_body_idx = s_idx+1
                break
            comments.append(s)
    func_body = statements[func_body_idx:]
    return method_name, "\n".join(signature), "\n".join(comments), "\n".join(func_body), before_func


file  ='/data/zlyuaj/muti-agent/fuzzing/data/HumanEval_test_case_ET.jsonl'
with open(file,'r') as f:
    lines = [json.loads(line) for line in f]
    # initial_score = [True, True, False, True, True, True, True, True, True, True, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, False, False, True, True, True, True, False, True, True, True, True, True, False, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, False, True, False, True, True, False, True, True, True, False, False, True, False, False, True, False, False, True, True, False, False, True, True, True, True, True, True, False, True, True, False, True, True, True, False, True, True, False, True, False, True, True, True, True, False, False, True, True, False, False, True, False, True, True, False, True, False, False, True, False, True, False, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, False]
        
    for index,i in enumerate(lines):
        # before_func
        # if initial_score[index]==False:
        #     continue
        examples,code,nl,explaination='','','',''
        p = i['prompt']
        # print('*'*50)
        # print('-'*50)
        # print(p)
        pp=code_split(p)
        # print(pp)
        # break
        method_name = pp[0]
        code=pp[1]
        # p=pp[3]
        y=['\"\"\"','\'\'\'']
        for yy in y:
            if yy in p:
                p=p.split(yy)
                # print(p)
                p=p[-2]
                
                break
        
        x=[ 'For example','for example','For Example','example','Example','Examples','samples:','>>>']
        o=method_name+'('
        name_idx=1e5
        if o in p and 'define' not in p:
            name_idx=p.find(o)
        
        for xx in x:
            if xx in p:
                idx=p.find(xx)
                if idx>name_idx:
                    continue
                if name_idx!=1e5 and name_idx - idx>30:
                    continue
                examples=p[idx:]
                p=p[:idx]
                break
        if examples=='':
            if o in p:
                idx=p.find(o)
                examples=p[idx:]
                p=p[:idx]
            
        
        nl=p
        lines[index]['nl']=nl
        lines[index]['func']=code
        lines[index]['examples']=examples
    with open ('a.jsonl','w+') as ff:
        for i in lines:
            ff.write(json.dumps(i) + '\n')
        # print('#'*50)
        # # print('-'*50)
        # # print(i['prompt'])
        # print('-'*50)
        # print(code)
        # print('-'*50)
        # print(nl)
        # # print('-'*50)
        # # print(explaination)
        # print('-'*50)
        # print(examples)
        
        # print('-'*50)

        # print(code+'\t\n\'\'\''+ nl+examples+'\'\'\'')

        
        # break

# x='1 def def def 2'
# i=x.find('def')
# print(i)
# print(x[i:])
# print(x[:i])


# with open('/data/zlyuaj/muti-agent/fuzzing/data/MBPP_ET.jsonl','r') as f:
#     lines = [json.loads(line) for line in f]
#     for line in lines:
#         print(line['text'])


'''
107
--------------------------------------------------
def fibfib(n: int):
--------------------------------------------------
The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    
--------------------------------------------------
fibfib(0) == 0
    fibfib(1) == 0
    fibfib(2) == 1
    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).
    Please write a function to efficiently compute the n-th element of the fibfib number sequence.
    >>> fibfib(1)
    0
    >>> fibfib(5)
    4
    >>> fibfib(8)
    24

105
--------------------------------------------------
def fib4(n: int):
--------------------------------------------------
The Fib4 number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    
--------------------------------------------------
fib4(0) -> 0
    fib4(1) -> 0
    fib4(2) -> 2
    fib4(3) -> 0
    fib4(n) -> fib4(n-1) + fib4(n-2) + fib4(n-3) + fib4(n-4).
    Please write a function to efficiently compute the n-th element of the fib4 number sequence.  Do not use recursion.
    >>> fib4(5)
    4
    >>> fib4(6)
    8
    >>> fib4(7)
    14
'''