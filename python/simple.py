class Foo():
    def __init__(self):
        print("foo init")

    def bark(self):
        print("bark")


config = {"env_name" :"foo"}

INFO = {
    "env_name" : "foo",
    "env_class" : Foo
}