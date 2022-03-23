def main():
    import argparse 
    from Crawller import Crawller, Macro
    from scripts import ALL_SCRIPTS 

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
        config = json.load(args.config_path)
    result = script(crawller, macro, config=config)
    print(result)
    crawller.driver.close()

if __name__ == "__main__":
    main()