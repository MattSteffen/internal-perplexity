import json
import os
import argparse

from crawler import Crawler

  

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run the document crawler and processor')
    parser.add_argument('--config', '-c', type=str, help='Path to your configuration file')
    parser.add_argument('--directory', '-d', type=str, help='Directory of files to import')
    args = parser.parse_args()
    
    # Load config file if provided
    config_dict = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
    
    # Override path with directory arg if provided
    if args.directory:
        config_dict["path"] = args.directory
    
    # Validate we have a path
    if "path" not in config_dict:
        parser.error("Must specify directory path either via --directory arg or in config file")
    
    # Create and run crawler
    crawler = Crawler(config_dict)
    
    # Process all documents
    with crawler._setup_vector_db() as db:
        for data in crawler.run():
            db.insert_data(data)

    print("complete")

if __name__ == "__main__":
    main()