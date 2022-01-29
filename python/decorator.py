def decorator_with_param(*args, **kwargs):
    def deco(self, *args, **kwargs):
        print(self, args, kwargs)
        print("-------param")
        return self
    return deco

def print_config_decorator(function, *de_args, **de_kwargs):
    def print_config(self, *args, **kwargs):
        print("-----------")
        print(self.config)
        print(self, args, de_args, de_kwargs)
        function(self, *args, **kwargs)
        print(self.config)
        print(self, args, de_args, de_kwargs)
        print(de_args)
        print("-----------")
        return None
    return print_config


def print_config_decorator_with_param(function, *de_args, **de_kwargs):
    def print_config(param, self, *args, **kwargs):
        print("-----------")
        print(self.config)
        print(self, args, de_args, de_kwargs)
        function(self, *args, **kwargs)
        print(self.config)
        print(self, args, de_args, de_kwargs)
        print(de_args)
        print("-----------")
        return None
    return print_config


ALL_MODELS = {}
def add_model_name_config(model_name, model_class):
    ALL_MODELS[model_name] = model_class
    print(ALL_MODELS)

class GymEnvironment():
    def __init__(self):
        add_model_name_config("GYM_TEST", GymEnvironment)
        self.config = {"hello":1, "visit":3, 'seed':0}

    @print_config_decorator
    def reset(self, reset_seed=0):
        self.config['seed'] = reset_seed
        print("here")
        print(self.config)
        return None
    
    # @print_config_decorator_with_param(10)
    # def reset_with_time(self, reset_seed=0):
    #     self.config['seed'] = reset_seed
    #     return None

env = GymEnvironment()
env.reset(5)
# env.reset_with_time(100)
print(env.config)