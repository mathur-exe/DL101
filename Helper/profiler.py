from pyinstrument import Profiler
from functools import wraps

def profile_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = Profiler()
        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop()
        print("PyInstrument Profiling Result:")
        print(profiler.output_text(unicode=True, color=True))
        return result
    return wrapper