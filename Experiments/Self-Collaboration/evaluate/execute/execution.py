# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
import logging

# from execute._execution import check_correctness, check_correctness_with_test_cases, check_correctness_T

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Dict
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile

def _pack_test_cases(test_cases, timeout):
    blank_4 = ' ' * 4
    blank_8 = ' ' * 8
    blank_12 = ' ' * 12
    result = f'def check():\n    pass_result = []\n'
    for idx, tc in enumerate(test_cases):
        multi_line_assertion = tc.strip().replace('\n', f'\n{blank_12}')
        result += f'\n{blank_4}try:\n{blank_8}with time_limit({timeout}):\n{blank_12}{multi_line_assertion}\
                    \n{blank_12}pass_result.append(True)\n{blank_4}except Exception as e:\n{blank_8}pass_result.append(False)\n'
    result += '\n    return pass_result\n'
    result += f'\nglobal final_result\nfinal_result = check()'
    return result


def check_correctness_with_test_cases(task_id, prompt, completion, test_cases, timeout):
    """
    Evaluates the functional correctness of a solution_content by running the test
    suite provided in the problem. 
    """
    extend_timeout = timeout*len(test_cases)

    def unsafe_execute():

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
                prompt + completion + "\n" +
                _pack_test_cases(test_cases, timeout)
            )

            try:
                exec_globals = {'time_limit': time_limit}
                with swallow_io():
                    exec(check_program, exec_globals)
                result.append(exec_globals['final_result'])
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=extend_timeout + 0.1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=task_id,
        test_cases=test_cases,
        completion=completion,
        passed=(type(result[0]) == list) and len(result[0]) > 0,
        result=result[0]
    )

def check_correctness_T(task_id: str, prompt: str, completion: str, test: list, entry_point: str, timeout: float) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 
    """

    def unsafe_execute():

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
                prompt + completion + "\n" + '\n'.join(test)
            )

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        exec(check_program, exec_globals)
                result.append("passed")
            except AssertionError:
                result.append("assertion")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout+1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=task_id,
        passed=result[0] == "passed",
        result=result[0],
        completion=completion,
    )


def check_correctness(task_id: str, prompt: str, completion: str, test: str, entry_point: str, timeout: float) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 
    """

    def unsafe_execute():

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
                completion + "\n" + test + "\n" + f'check({entry_point})'
            )

            # print(check_program)

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        exec(check_program, exec_globals)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                # print('failed')
                # print(check_program)
                result.append(f"failed: {e}")

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout+1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=task_id,
        passed=result[0] == "passed",
        result=result[0],
        completion=completion,
    )

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


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
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

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
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


def evaluate_with_test_code_one_sample(
    sample,
    timeout
):
    # logger.info(f'Start evaluation with test code, timeout={timeout}')
    # Check the generated samples against test suites.
    

        
    result=False
    score=0

    test_number=len(sample['test'])
    correct=0
    output_results=[]

    task_id = sample["task_id"]
    
    prompt = sample['prompt']
    
    entry_point = sample['entry_point']
    completion = sample["completion"]

    tests = sample['test']
    # print(completion)
    
    # print(completion)
    # print(prompt)
    # 遍历所有的test
    futures = []
    # print(tests)
    with ProcessPoolExecutor() as executor:
        for test in tests:
            # print(test)
            args = (task_id, prompt, completion, test, entry_point, timeout)
            # 将参数输入并运行
            future = executor.submit(check_correctness, *args)
            futures.append(future)
    # logger.info(f'{len(futures)} execution requests are submitted')  
    for idx, future in enumerate(as_completed(futures)):
        output_results.append(future.result())
    # print(len(output_results))
    # print(output_results)
    # print(len(output_results))
    for idx, exe_result in enumerate(output_results):
        # logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
        
        if exe_result['passed']:
            correct+=1
    if correct==test_number:
        result = True
    score = round(correct/test_number ,4)
        # print(round(correct[i]/test_number[i],4))
    # print(scores)
    
        
        # for idx, future in enumerate(as_completed(futures)):
        #     # logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
        #     f=0
        #     for i in range(len(future)):
        #         result = future[i].result()
        #         print("&"*288)
        #         print(result)
        #         if result == 'pass':
        #             f+=1
        #     if f==len(future):
        #         results[result["task_id"]][result["completion"]] = 'passed'
        #     else:
        #         results[result["task_id"]][result["completion"]] = 'failed'
        #     score[result["task_id"]][result["completion"]] = f/len(future)
        #     print(score)


    # logger.info('execution finished! start parsing results')

    sample["pass_results"].append(result)
    sample['scores'].append(score)
    sample['pass_test_cases_num'].append(correct)


    return sample


def evaluate_with_test_code(
    samples,
    timeout
):
    # logger.info(f'Start evaluation with test code, timeout={timeout}')
    # Check the generated samples against test suites.
    

        
    existed_completion = defaultdict(set)
    results = [False for i in range(len(samples))]
    scores = [0 for i in range(len(samples))]

    test_number=[len(sample['test']) for sample in samples]
    correct=[ 0 for i in range(len(samples))]
    output_results=[]

    
    
    for sample in samples:
        task_id = sample["task_id"]
        prompt = sample['prompt']
        
        entry_point = sample['entry_point']
        completion = sample["completion"]

        if completion in existed_completion[task_id]:
            continue
        existed_completion[task_id].add(completion)
        tests = sample['test']
        # print(completion)
        

        # 遍历所有的test
        futures = []
        with ProcessPoolExecutor() as executor:
            for test in tests:
                # print(test)
                args = (task_id, prompt, completion, test, entry_point, timeout)
                # 将参数输入并运行
                future = executor.submit(check_correctness, *args)
                futures.append(future)
        # logger.info(f'{len(futures)} execution requests are submitted')  
        for idx, future in enumerate(as_completed(futures)):
            output_results.append(future.result())
        # print(len(output_results))
    
    # print(len(output_results))
    for idx, result in enumerate(output_results):
        # logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
        task_num = int(result['task_id'].split('/')[-1])
        if result['passed']:
            correct[task_num]+=1
    for i in range(len(samples)):
        if correct[i]==test_number[i]:
            results[i] = True
        scores[i] = round(correct[i]/test_number[i],4)
        # print(round(correct[i]/test_number[i],4))
    # print(scores)
    
        
        # for idx, future in enumerate(as_completed(futures)):
        #     # logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
        #     f=0
        #     for i in range(len(future)):
        #         result = future[i].result()
        #         print("&"*288)
        #         print(result)
        #         if result == 'pass':
        #             f+=1
        #     if f==len(future):
        #         results[result["task_id"]][result["completion"]] = 'passed'
        #     else:
        #         results[result["task_id"]][result["completion"]] = 'failed'
        #     score[result["task_id"]][result["completion"]] = f/len(future)
        #     print(score)


    # logger.info('execution finished! start parsing results')
    samples_with_result = []
    for idx,sample in enumerate(samples):
        task_id = sample["task_id"]
        completion = sample["completion"]
        # result = results[task_id][completion]
        # sample["result"] = result["result"]
        sample["pass_results"].append(results[idx])
        sample['scores'].append(scores[idx])
        sample['pass_test_cases_num'].append(correct[idx])
        samples_with_result.append(sample)

    assert len(samples_with_result) == len(samples), "Some problems are not attempted."

    return samples_with_result

def evaluate_with_test_cases(
    solutions,
    test_cases_dict,
    timeout,
    limit
):
    # logger.info(f'Start evaluation with test cases, timeout={timeout}, limit={limit}')
    # Check the generated solutions against test suites.
    with ProcessPoolExecutor() as executor:
        futures = []
        results_list = []
        existed_completion = defaultdict(set)

        for solution in solutions:
            task_id = solution['task_id']
            prompt = solution['prompt']
            completion = solution['completion']
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            task_test_cases = test_cases_dict[task_id]
            if not task_test_cases:
                continue
            # get limited test cases
            limited_task_test_cases = [cases_per_sample[:limit] for cases_per_sample in task_test_cases]
            limited_task_test_cases = sum(limited_task_test_cases, [])
            
            args = (task_id, prompt, completion, list(set(limited_task_test_cases)), timeout)
            future = executor.submit(check_correctness_with_test_cases, *args)
            futures.append(future)

        # logger.info(f'{len(futures)} execution requests are submitted')
        for idx, future in enumerate(as_completed(futures)):
            # logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results_list.append(result)

    # logger.info('execution finished!')
    return results_list

def evaluate_with_test_code_T(
    samples,
    timeout
):
    # logger.info(f'Start evaluation with test code, timeout={timeout}')
    # Check the generated samples against test suites.
    with ProcessPoolExecutor() as executor:

        futures = []
        existed_completion = defaultdict(set)
        results = defaultdict(defaultdict)

        for sample in samples:
            task_id = sample["task_id"]
            prompt = sample['prompt']
            test = sample['test_case_list']
            entry_point = sample['entry_point']
            completion = sample["completion"]
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            args = (task_id, prompt, completion, test, entry_point, timeout)
            future = executor.submit(check_correctness_T, *args)
            futures.append(future)
        # logger.info(f'{len(futures)} execution requests are submitted')
        
        for idx, future in enumerate(as_completed(futures)):
            # logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results[result["task_id"]][result["completion"]] = result

    # logger.info('execution finished! start parsing results')
    samples_with_result = []
    for sample in samples:
        task_id = sample["task_id"]
        completion = sample["completion"]
        result = results[task_id][completion]
        sample["result"] = result["result"]
        sample["passed"] = result["passed"]
        samples_with_result.append(sample)

    assert len(samples_with_result) == len(samples), "Some problems are not attempted."

    return samples_with_result