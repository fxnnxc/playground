def main():
    import argparse 
    from Crawller import Crawller, Macro
    from scripts import ALL_SCRIPTS 
    import json 

    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str)
    parser.add_argument("--config-path", type=str, default="")

    args =parser.parse_args()

    crawller = Crawller()
    macro = Macro()
    assert args.script in ALL_SCRIPTS 
    script = ALL_SCRIPTS[args.script]
    if args.config_path == "":
        config = None 
    else:
        import json 
        with open(args.config_path) as f:
            config = json.load(f)
    result = script(crawller, macro, config=config)
    with open("save.json", "w") as f :
        json.dump(result, f)
    crawller.driver.close()

if __name__ == "__main__":
    main()