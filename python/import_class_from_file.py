import importlib

def load_module_func(module_name):
    mod = importlib.import_module(module_name)
    print("hello")
    print(mod)
    print(dir(mod))
    print(mod.DEEPING_INFO)
    return mod


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config-json", type=str, default="")
    parser.add_argument("--env-script", type=str, default="")

    args =parser.parse_args()
    script = args.env_script 

    load_module_func(script)