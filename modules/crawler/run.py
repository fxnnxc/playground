def main():
    import argparse 
    from Crawller import Crawller, Macro
    from scripts import ALL_SCRIPTS 
    import json 

    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="")
    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--html-path", type=str, default="")

    args =parser.parse_args()

    if args.script != "":
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
        conf_name = args.config_path.split("/")[-1]
        result = script(crawller, macro, config=config)
        with open(f"results/{conf_name}", "w") as f :
            json.dump(result, f)
        crawller.driver.close()

    if args.html_path != "":
        from parsing import Parser
        import os 
        with open(os.path.join(args.html_path)) as handle:
            html_doc_dict = json.load(handle)
        
        all_results = {}
        conf_name = args.html_path.split("/")[-1]
        script = conf_name.split("-")[0]

        for keyword, html_doc_list in html_doc_dict.items():
            all_results[keyword] = {}
            for html_doc in html_doc_list:
                result = getattr(Parser(), script)(html_doc)
                all_results[keyword].update(result)

        with open(f'results/parsed_{conf_name}', "w", encoding='UTF-8-sig') as handle:
            json.dump(all_results, handle, indent=4, ensure_ascii = False)
        
    


if __name__ == "__main__":
    main()