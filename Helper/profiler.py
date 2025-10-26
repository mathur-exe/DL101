import functools
import cProfile, pstats, io
import yappi

def profile(tool = "snakeviz", top_n = 10, filename = "profile.prof"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if tool == "cProfile":
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    return func(*args, **kwargs)
                finally:
                    profiler.disable()
                    s = io.StringIO()
                    stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
                    stats.print_stats(top_n)
                    print(s.getvalue())

            elif tool == "snakeviz":
                cProfile.runctx("result = func(*args, **kwargs)", 
                                globals(), locals(), filename=filename)
                print(f"Run `snakeviz {filename}` to explore results")
                return locals()["result"]
            
            # TODO: understand yappi
            # TODO: creation of a decorator for yappi
            elif tool == "yappi":
                raise NotImplementedError("yappi is not implemented yet")
            
            else:
                raise ValueError(f"Invalid tool: {tool}")
        return wrapper
    return decorator