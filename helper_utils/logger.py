import logging
import functools
import time

# Configure logging to display timestamp, log level, and message.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_decorator(func):
    """
    A decorator that logs the entry, exit, arguments, result, 
    and execution time of the decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Entering: {func.__name__} | args: {args}, kwargs: {kwargs}")
        start_time = time.time()
        
        # Call the wrapped function.
        result = func(*args, **kwargs)
        
        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(f"Exiting: {func.__name__} | returned: {result} in {elapsed:.4f} sec")
        return result
    return wrapper