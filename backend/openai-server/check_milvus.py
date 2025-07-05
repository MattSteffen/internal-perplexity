#!/usr/bin/env python3
"""
Milvus Collection Inspector
This script connects to a Milvus database and lists all collections
along with the number of items (entities) in each collection.
"""

from pymilvus import connections, utility, Collection
import sys
from typing import List, Tuple


def connect_to_milvus(host: str = "localhost", port: str = "19530", alias: str = "default") -> bool:
    """
    Connect to Milvus server
    
    Args:
        host: Milvus server host (default: localhost)
        port: Milvus server port (default: 19530)
        alias: Connection alias (default: default)
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        connections.connect(
            alias=alias,
            host=host,
            port=port
        )
        print(f"✅ Connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False


def get_collections_info() -> List[Tuple[str, int]]:
    """
    Get information about all collections in the database
    
    Returns:
        List of tuples containing (collection_name, entity_count)
    """
    collections_info = []
    
    try:
        # Get all collection names
        collection_names = utility.list_collections()
        
        if not collection_names:
            print("📭 No collections found in the database")
            return collections_info
        
        print(f"🔍 Found {len(collection_names)} collection(s)")
        
        for collection_name in collection_names:
            try:
                # Create collection object
                collection = Collection(collection_name)
                
                # Load collection to get accurate count
                collection.load()
                
                # Get entity count
                entity_count = collection.num_entities
                
                collections_info.append((collection_name, entity_count))
                
                # Release collection from memory
                collection.release()
                
            except Exception as e:
                print(f"⚠️  Error processing collection '{collection_name}': {e}")
                collections_info.append((collection_name, -1))  # -1 indicates error
                
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        
    return collections_info


def display_collections_info(collections_info: List[Tuple[str, int]]):
    """
    Display collections information in a formatted table
    
    Args:
        collections_info: List of tuples containing (collection_name, entity_count)
    """
    if not collections_info:
        print("📭 No collections to display")
        return
    
    # Calculate column widths
    max_name_width = max(len(name) for name, _ in collections_info)
    max_name_width = max(max_name_width, len("Collection Name"))
    
    # Print header
    print("\n" + "="*60)
    print(f"{'Collection Name':<{max_name_width}} | {'Entity Count':>12}")
    print("="*60)
    
    # Print collection info
    total_entities = 0
    for collection_name, entity_count in collections_info:
        if entity_count >= 0:
            print(f"{collection_name:<{max_name_width}} | {entity_count:>12,}")
            total_entities += entity_count
        else:
            print(f"{collection_name:<{max_name_width}} | {'Error':>12}")
    
    # Print summary
    print("="*60)
    print(f"{'Total Collections:':<{max_name_width}} | {len(collections_info):>12}")
    print(f"{'Total Entities:':<{max_name_width}} | {total_entities:>12,}")
    print("="*60)


def main():
    """
    Main function to run the Milvus collection inspector
    """
    print("🚀 Milvus Collection Inspector")
    print("-" * 30)
    
    # Configuration - you can modify these values
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    
    # Connect to Milvus
    if not connect_to_milvus(MILVUS_HOST, MILVUS_PORT):
        sys.exit(1)
    
    try:
        # Get collections information
        collections_info = get_collections_info()
        
        # Display the information
        display_collections_info(collections_info)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    
    finally:
        # Disconnect from Milvus
        connections.disconnect("default")
        print("\n👋 Disconnected from Milvus")


if __name__ == "__main__":
    main()