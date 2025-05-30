from roles import Analyst, Coder, Tester
from utils import find_method_name
from monitor import monitor
import time
from utils import code_truncate

def printinfo(info, head, round=0):
    print('---'*10)
    print('round: '+str(round))
    print(head)
    print(info)

class Session(object):
    def __init__(self, TEAM, ANALYST, PYTHON_DEVELOPER, TESTER, requirement, model='gpt-35-turbo', repair_prompt = False,repair_code = False,add_monitor=False, majority=1, max_tokens=512,
                                temperature=0.0, top_p=1.0, max_round=4, before_func=''):

        self.session_history = {}
        # 默认为2轮
        self.max_round = max_round
        self.before_func = before_func
        self.requirement = requirement
        # 初始化角色，这一步就已经进行了第一次分析
        self.analyst = Analyst(TEAM, ANALYST, requirement, model, majority, max_tokens, temperature, top_p)
        self.coder = Coder(TEAM, PYTHON_DEVELOPER, requirement, model, majority, max_tokens, temperature, top_p)
        self.tester = Tester(TEAM, TESTER, requirement, model, majority, max_tokens, temperature, top_p)
        self.repair_prompt = repair_prompt
        self.add_monitor = add_monitor
        self.repair_code =repair_code
        self.model = model
    def monitor_plan(self,plan):
        max_try=3
        for i in range(max_try):
            try:
                # print(f'in session.monitor plan, model = {self.model}')
                res = monitor(plan,self.requirement,model=self.model,task= 'repair_plan')
                if type(res) == list and len(res)>0:
                    more_plan=res[0]
                    break
                else:
                    more_plan=''
            except Exception as e:
                print(e)
                more_plan=''
        if not more_plan:
            print('fail to generate interperated plan!')
            more_plan = ''
        INTEPERATE_PROMPT = '\nPlease read and understand the following inteperation before coding\n'
        # split_plan = more_plan.split('\n')
        # cleaned_plan = ''
        # for plan_line in split_plan:
        #     if '2.' in plan_line and ('No' in plan_line or 'no' in plan_line):
        #         continue
        #     if len(plan_line)>3 and 

        return plan + INTEPERATE_PROMPT +  more_plan

    def monitor_code(self,code,plan):
        try:
            # print(f'in session.monitor_code, model = {self.model}')
            res = monitor(plan,self.requirement,code,self.model,task= 'judge_code')
            if type(res) == list and len(res)>0:
                code_analysis=res[0]
            else:
                code_analysis=''
        except Exception as e:
            print(e)
            code_analysis=''
        need_regenerate = False
        if type(code_analysis)==str and '[YES]' in code_analysis:
            need_regenerate = True
        return code_analysis ,need_regenerate
        

    def run_session(self,need_second_round,finally_pass):
        # 首先让分析师进行分析
        plan = self.analyst.analyze()
        # print(self.requirement)
        # printinfo(plan,'plan')



        report = plan
        is_init=True
        
        # print(plan)

        # if self.repair_prompt:
        #     # print(report)
        #     report=self.repair_plan(report)
        #     # print(report)
        # print(f'try to monitor plan, model = {self.model}')
        if self.add_monitor and self.repair_prompt:
            report=self.monitor_plan(report)
        # else:
        #     print('Do not repair prompt')
        self.session_history["plan"] = report

        code = ""
        # print(report)

        # 两轮迭代
        for i in range(self.max_round):
            # 第一轮代码生成
            # print('in coding')
            naivecode = self.coder.implement(report, is_init)

            need_regenerate=False
            if self.add_monitor and self.repair_code:
                # print(f'try to monitor code, model = {self.model}')
                code_monitor_result,need_regenerate=self.monitor_code(naivecode,report)
            
            if need_regenerate:
                naivecode = self.coder.implement(report, is_init)
            
            # print('end of coding')
            # print(naivecode)
            
            method_name = find_method_name(naivecode)
            # print('pass')
            # if method_name:
                # code = naivecode


                
            # 一个string
            code = naivecode
            
            # 保存代码
            if code.strip() == "":
                if i == 0:
                    code = "error"
                else:
                    code = self.session_history['Round_{}'.format(i-1)]["code"]
                break
            # printinfo(code,'code',i)
            
            
            
            # 测试
            # print('in testing')
            tests = self.tester.test(code)
            # print('end of testing')
            test_report = code_truncate(tests)
            # printinfo(tests,'test report',i)
            answer_report = unsafe_execute(self.before_func+code+'\n'+test_report+f'check({method_name})', '')
            report = f'The compilation output of the preceding code is: {answer_report}'
            # printinfo(report,'test result',i)

            is_init = False
            self.session_history['Round_{}'.format(i)] = {"code": code, "report": report}

            if (plan == "error") or (code == "error") or (report == "error"):
                code = "error"    
                break
            
            if answer_report == "Code Test Passed.":
                # print('Passed ')
                finally_pass+=1
                break
            
            if i == self.max_round-1:
                self.session_history['Round_{}'.format(i)] = {"code": code}
                break
            need_second_round+=1

        self.analyst.itf.clear_history()
        self.coder.itf.clear_history()
        self.tester.itf.clear_history()
        # print(code)
        
        return code, self.session_history, need_second_round, finally_pass

    def run_analyst_coder(self):
        plan = self.analyst.analyze()
        is_init=True
        self.session_history["plan"] = plan
        code = self.coder.implement(plan, is_init)

        if (plan == "error") or (code == "error"):
            code = "error"

        self.analyst.itf.clear_history()
        self.coder.itf.clear_history()
        self.tester.itf.clear_history()

        return code, self.session_history


    def run_coder_tester(self):
        report = ""
        is_init=True
        code = ""
        
        for i in range(self.max_round):

            naivecode = self.coder.implement(report, is_init)
            if find_method_name(naivecode):
                code = naivecode

            if code.strip() == "":
                if i == 0:
                    code = self.coder.implement(report, is_init=True)
                else:
                    code = self.session_history['Round_{}'.format(i-1)]["code"]
                break
            
            if i == self.max_round-1:
                self.session_history['Round_{}'.format(i)] = {"code": code}
                break
            tests = self.tester.test(code)
            test_report = code_truncate(tests)
            answer_report = unsafe_execute(self.before_func+code+'\n'+test_report+'\n'+f'check({method_name})', '')
            report = f'The compilation output of the preceding code is: {answer_report}'

            is_init = False
            self.session_history['Round_{}'.format(i)] = {"code": code, "report": report}

            if (code == "error") or (report == "error"):
                code = "error"
                break
            
            if report == "Code Test Passed.":
                break

        self.analyst.itf.clear_history()
        self.coder.itf.clear_history()
        self.tester.itf.clear_history()
        
        return code, self.session_history

    def run_coder_only(self):
        plan = ""
        code = self.coder.implement(plan, is_init=True)
        self.coder.itf.clear_history()
        return code, self.session_history,0,0


import contextlib
import faulthandler
import io
import os
import platform
import signal
import tempfile 

def unsafe_execute(code, report):

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir 

            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            # Construct the check program and run it.
            check_program = (
                code + report
            )

            try:
                exec_globals = {}
                with swallow_io():
                    timeout = 10
                    with time_limit(timeout):
                        exec(check_program, exec_globals)
                result = "Code Test Passed."
            except AssertionError as e:
                result = f"failed with AssertionError. {e}"
            except TimeoutException:
                result = "timed out"
            except BaseException as e:
                result = f"{e}"


            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
            return result


def reliability_guard(maximum_memory_bytes = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the 
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    os.rmdir = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    import sys
    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None
    
@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname
            
class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)