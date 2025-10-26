### System Functionality
`sys` is one of the useful libraries available by default in python, while some of the vanilla feature like `sys.flags` can come handy but other feature like `sys.setprofile`, `sys.<I/O>` have better replacement which are in used as common practice by experienced dev and architects 

1. `sys.argv` --> `argparse` / `jsonargparse`
    ```
    import argparse

    def main(argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--verbose", "-v", action="store_true")
        args = parser.parse_args(argv)
        ...
    ```

2. `sys.settrace` / `sys.setprofile` --> `SnakeViz`, `cProfile` and `yappi` (multithreading)
```
# SnakeViz : ref 

# CProfile : ref 

# yappi : yet to implement
```

### Dataclasses
